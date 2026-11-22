from __future__ import annotations
import hashlib
import logging
import numpy as np
import os
import pathlib
import re
import time
from typing import NamedTuple

from fastembed import TextEmbedding

from ._md import parse_frontmatter

log = logging.getLogger(__name__)

CHUNK_WORDS = 750       # ~1000 tokens
OVERLAP_WORDS = 75
# Bump when _chunks() changes how it splits or joins text: it is part of the
# content hash, so every transcript is re-embedded on the next build.
CHUNKER_VERSION = 1

OFFSET_DTYPE = np.dtype([
    ("video_id", "U16"),
    ("chunk_index", "i4"),
    ("start_sec", "i4"),
    ("end_sec", "i4"),
    ("char_start", "i4"),
    ("char_end", "i4"),
])

HASHES_DTYPE = np.dtype([("video_id", "U16"), ("sha256", "U64")])


def _parse_md(md_path: pathlib.Path):
    """Path-taking wrapper around parse_frontmatter for back-compat."""
    return parse_frontmatter(md_path.read_text())


def _chunks(body: str) -> list[dict]:
    """Sliding window over the [HH:MM:SS] body lines."""
    lines = []
    for line in body.splitlines():
        m = re.match(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.+)$", line)
        if not m:
            continue
        h, m_, s, t = m.groups()
        sec = int(h) * 3600 + int(m_) * 60 + int(s)
        lines.append((sec, t))

    out: list[dict] = []
    words: list[str] = []
    start_sec: int | None = None
    cursor = 0
    last_sec = 0
    for sec, t in lines:
        if start_sec is None:
            start_sec = sec
        last_sec = sec
        words.extend(t.split())
        while len(words) >= CHUNK_WORDS:
            chunk_text = " ".join(words[:CHUNK_WORDS])
            out.append({
                "start_sec": start_sec,
                "end_sec": last_sec,
                "char_start": cursor,
                "char_end": cursor + len(chunk_text),
                "text": chunk_text,
            })
            cursor += len(chunk_text) + 1
            words = words[CHUNK_WORDS - OVERLAP_WORDS:]
            start_sec = last_sec
    if words:
        chunk_text = " ".join(words)
        out.append({
            "start_sec": start_sec if start_sec is not None else 0,
            "end_sec": last_sec,
            "char_start": cursor,
            "char_end": cursor + len(chunk_text),
            "text": chunk_text,
        })
    return out


class _Transcript(NamedTuple):
    path: pathlib.Path
    sha256: str
    chunks: list[dict]


def _content_hash(body: str) -> str:
    """Hash of what the embeddings are computed from: the transcript body and
    the chunking parameters. A `touch` or a git checkout keeps it; an edit to
    the body, a new CHUNK_WORDS / OVERLAP_WORDS or a CHUNKER_VERSION bump
    changes it."""
    h = hashlib.sha256(f"chunks:v{CHUNKER_VERSION}:{CHUNK_WORDS}:{OVERLAP_WORDS}\n".encode())
    h.update(body.encode("utf-8"))
    return h.hexdigest()


def _scan(transcripts_dir: pathlib.Path) -> dict[str, _Transcript]:
    """video_id -> (path, content hash, chunks), for every transcript with a
    video_id. Each file is read once, so the hash describes exactly the text
    that is chunked and embedded, even if the file changes during the build."""
    out: dict[str, _Transcript] = {}
    for md in sorted(transcripts_dir.glob("*.md")):
        fm, body = parse_frontmatter(md.read_text())
        vid = fm.get("video_id")
        if vid:
            out[vid] = _Transcript(md, _content_hash(body), _chunks(body))
    return out


def _load_existing(out: pathlib.Path, model_name: str):
    """Read the stored vectors, offsets, built_at and content hashes.

    Returns (vectors, offsets, built_at, hashes), or None when the file must
    be rebuilt from scratch: it was written with another model, or it can't
    be read. npz members load lazily, so a damaged member only fails when it
    is read; the reads therefore sit inside the try as well.

    built_at is when the previous build started reading transcripts. Files
    written before `built_at` was stored fall back to the npz file's mtime.
    hashes is {video_id: sha256} for every transcript of that build, or None
    for a file written before hashes were stored (legacy).
    """
    try:
        with np.load(out, allow_pickle=False) as npz:
            # A different model means the stored vectors live in another
            # embedding space: drop them rather than mix incompatible vectors.
            if str(npz["model"]) != model_name:
                return None
            vecs, offs = npz["vectors"], npz["offsets"]
            built_at = (float(npz["built_at"]) if "built_at" in npz.files
                        else out.stat().st_mtime)
            hashes = None
            if "hashes" in npz.files:
                h = npz["hashes"]
                if h.dtype != HASHES_DTYPE:
                    raise ValueError(f"unexpected hashes dtype {h.dtype}")
                hashes = dict(zip(h["video_id"].tolist(), h["sha256"].tolist()))
        if vecs.ndim != 2 or len(vecs) != len(offs) or offs.dtype != OFFSET_DTYPE:
            raise ValueError(f"inconsistent arrays: vectors {vecs.shape}, "
                             f"offsets {offs.shape} {offs.dtype}")
    except Exception as e:
        log.warning("Could not read %s (%s); rebuilding all embeddings.", out, e)
        return None
    return vecs, offs, built_at, hashes


def _unchanged_ids(on_disk: dict[str, _Transcript], offsets: np.ndarray,
                   built_at: float, hashes: dict[str, str] | None) -> set[str]:
    """Transcripts whose stored rows can be reused as they are.

    A transcript qualifies only if the npz holds exactly one row per chunk it
    has now (so an id missing from the npz, e.g. a file restored with
    `cp -p`, is never trusted), and its content is unchanged:
    - npz with hashes: the stored hash equals the current one; mtimes are
      ignored, so a `touch` or a git checkout re-embeds nothing.
    - legacy npz without hashes: its mtime is older than the stored built_at
      (strict `<`: on a coarse-mtime filesystem a tie means "maybe changed").
      The build then writes hashes, so this fallback runs once.
    """
    ids, counts = np.unique(offsets["video_id"], return_counts=True)
    stored_rows = dict(zip(ids.tolist(), counts.tolist()))
    keep: set[str] = set()
    for vid, tr in on_disk.items():
        if stored_rows.get(vid, 0) != len(tr.chunks):
            continue
        if hashes is not None:
            if hashes.get(vid) == tr.sha256:
                keep.add(vid)
        elif tr.path.stat().st_mtime < built_at:
            keep.add(vid)
    return keep


def _hashes_array(on_disk: dict[str, _Transcript]) -> np.ndarray:
    return np.array([(vid, tr.sha256) for vid, tr in on_disk.items()],
                    dtype=HASHES_DTYPE)


def _save_atomic(out: pathlib.Path, **arrays) -> None:
    """Write the npz to a temp file next to `out`, then swap it in.

    A crash or Ctrl-C during the write leaves the old file untouched. The temp
    file is opened by us and passed to np.savez as a file object, so numpy does
    not append its own `.npz` suffix to the name. Temp files left by a
    process that was killed outright (no `finally`) are removed first.
    """
    for stale in out.parent.glob(f".{out.stem}.*.tmp{out.suffix}"):
        stale.unlink(missing_ok=True)
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp{out.suffix}")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()


def build_embeddings(transcripts_dir: pathlib.Path, refs_dir: pathlib.Path,
                     model_name: str = "BAAI/bge-small-en-v1.5",
                     incremental: bool = False) -> pathlib.Path:
    refs_dir.mkdir(parents=True, exist_ok=True)
    out = refs_dir / "embeddings.npz"

    # Taken before any transcript is read. Only legacy files (no hashes) use
    # it for change detection; it is still stored for older readers.
    started = time.time()
    on_disk = _scan(transcripts_dir)

    existing = _load_existing(out, model_name) if out.exists() else None

    # Fast path for a full build: if no transcript changed AND the stored
    # model matches the requested one, skip. When the caller changes
    # --multilingual (bge-m3, 1024-d) vs default (bge-small, 384-d), the
    # stored npz is stale and we MUST rebuild, otherwise search.py reads
    # vectors that don't match corpus_meta.json's embedding_dim
    # (_load_existing returns None for another model). A transcript deleted
    # since the last build (e.g. an update rollback), one changed, or one on
    # disk whose rows are missing from the npz also forces a rebuild.
    if existing is not None and not incremental:
        vecs, offs, built_at, hashes = existing
        stored_ids = set(offs["video_id"].tolist())
        if (stored_ids <= on_disk.keys()
                and _unchanged_ids(on_disk, offs, built_at, hashes) == on_disk.keys()):
            current = {vid: tr.sha256 for vid, tr in on_disk.items()}
            if hashes != current:
                # Legacy file (or one listing a deleted transcript without
                # rows): record the current hashes, keep the vectors as they
                # are, and don't load the model.
                _save_atomic(out, vectors=vecs, offsets=offs,
                             model=np.array(model_name),
                             built_at=np.array(built_at),
                             hashes=_hashes_array(on_disk))
            return out

    model = TextEmbedding(model_name=model_name)

    existing_vec, existing_off = None, None
    keep_ids: set[str] = set()
    if incremental and existing is not None:
        existing_vec, existing_off, built_at, hashes = existing
        # Keep a transcript's stored rows only if it is still on disk and
        # unchanged. Everything else is re-embedded below (new or changed)
        # or dropped (deleted).
        keep_ids = _unchanged_ids(on_disk, existing_off, built_at, hashes)
        mask = np.isin(existing_off["video_id"], list(keep_ids))
        existing_vec, existing_off = existing_vec[mask], existing_off[mask]

    new_texts: list[str] = []
    new_offsets: list[tuple] = []
    for vid, tr in on_disk.items():
        if vid in keep_ids:
            continue
        for i, ch in enumerate(tr.chunks):
            new_texts.append(ch["text"])
            new_offsets.append((vid, i, ch["start_sec"], ch["end_sec"],
                                ch["char_start"], ch["char_end"]))

    if new_texts:
        vecs = np.array(list(model.embed(new_texts)), dtype=np.float16)
        offs = np.array(new_offsets, dtype=OFFSET_DTYPE)
        if existing_vec is not None:
            vecs = np.vstack([existing_vec, vecs])
            offs = np.concatenate([existing_off, offs])
    elif existing_vec is not None:
        # Nothing new; may be empty (every stored transcript deleted) but
        # still keeps the stored (0, dim) shape.
        vecs, offs = existing_vec, existing_off
    else:
        # Determine dim from a single-token embed to get the right shape.
        probe = np.array(list(model.embed(["probe"])), dtype=np.float16)
        vecs = np.zeros((0, probe.shape[1]), dtype=np.float16)
        offs = np.zeros((0,), dtype=OFFSET_DTYPE)

    _save_atomic(out, vectors=vecs, offsets=offs, model=np.array(model_name),
                 built_at=np.array(started), hashes=_hashes_array(on_disk))
    return out
