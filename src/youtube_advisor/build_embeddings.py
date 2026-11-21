from __future__ import annotations
import logging
import numpy as np
import os
import pathlib
import re
from fastembed import TextEmbedding

from ._md import parse_frontmatter

log = logging.getLogger(__name__)

CHUNK_WORDS = 750       # ~1000 tokens
OVERLAP_WORDS = 75

OFFSET_DTYPE = np.dtype([
    ("video_id", "U16"),
    ("chunk_index", "i4"),
    ("start_sec", "i4"),
    ("end_sec", "i4"),
    ("char_start", "i4"),
    ("char_end", "i4"),
])


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


def _transcript_ids(transcripts_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """video_id -> transcript path, for every transcript with a video_id."""
    ids: dict[str, pathlib.Path] = {}
    for md in sorted(transcripts_dir.glob("*.md")):
        fm, _ = _parse_md(md)
        vid = fm.get("video_id")
        if vid:
            ids[vid] = md
    return ids


def _load_existing(out: pathlib.Path, model_name: str):
    """Read the stored vectors and offsets for an incremental build.

    Returns (vectors, offsets), or (None, None) when the file must be rebuilt
    from scratch: it was written with another model, or it can't be read.
    npz members load lazily, so a damaged member only fails when it is read;
    the reads therefore sit inside the try as well.
    """
    try:
        with np.load(out, allow_pickle=False) as npz:
            # A different model means the stored vectors live in another
            # embedding space: drop them rather than mix incompatible vectors.
            if str(npz["model"]) != model_name:
                return None, None
            vecs, offs = npz["vectors"], npz["offsets"]
        if vecs.ndim != 2 or len(vecs) != len(offs) or "video_id" not in (offs.dtype.names or ()):
            raise ValueError(f"inconsistent arrays: vectors {vecs.shape}, offsets {offs.shape}")
    except Exception as e:
        log.warning("Could not read %s (%s); rebuilding all embeddings.", out, e)
        return None, None
    return vecs, offs


def _save_atomic(out: pathlib.Path, **arrays) -> None:
    """Write the npz to a temp file next to `out`, then swap it in.

    A crash or Ctrl-C during the write leaves the old file untouched. The temp
    file is opened by us and passed to np.savez as a file object, so numpy does
    not append its own `.npz` suffix to the name.
    """
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
    on_disk = _transcript_ids(transcripts_dir)

    # Fast path: if nothing's changed AND the stored model matches the
    # requested one, skip. When the caller changes --multilingual (bge-m3,
    # 1024-d) vs default (bge-small, 384-d), the stored npz is stale and we
    # MUST rebuild — otherwise search.py reads vectors that don't match
    # corpus_meta.json's embedding_dim and shape errors at query time.
    # A transcript deleted since the last build (e.g. an update rollback)
    # also forces a rebuild, so its rows don't linger in search results.
    if out.exists() and not incremental:
        emb_mtime = out.stat().st_mtime
        any_newer = any(p.stat().st_mtime > emb_mtime for p in transcripts_dir.glob("*.md"))
        if not any_newer:
            try:
                with np.load(out, allow_pickle=False) as npz_check:
                    stored_model = str(npz_check["model"])
                    # Shape sanity only: a non-2-D `vectors` array raises here
                    # and forces a rebuild. The stored dim is NOT compared with
                    # the requested model's dim.
                    _ = npz_check["vectors"].shape[1] if npz_check["vectors"].size else None
                    stored_ids = set(npz_check["offsets"]["video_id"].tolist())
            except Exception:
                stored_model, stored_ids = None, set()
            # Subset, not equality: a transcript without timestamped lines
            # has no rows, and that must not force a rebuild every time.
            if stored_model == model_name and stored_ids <= on_disk.keys():
                return out

    model = TextEmbedding(model_name=model_name)

    existing_vec, existing_off = None, None
    keep_ids: set[str] = set()
    if incremental and out.exists():
        emb_mtime = out.stat().st_mtime
        existing_vec, existing_off = _load_existing(out, model_name)
        if existing_off is not None:
            stored_ids = set(existing_off["video_id"].tolist())
            # Keep a transcript's stored rows only if it is still on disk and
            # has not changed since the last build. Everything else is either
            # re-embedded below (new or changed) or dropped (deleted).
            keep_ids = {vid for vid, md in on_disk.items()
                        if vid in stored_ids and md.stat().st_mtime <= emb_mtime}
            mask = np.isin(existing_off["video_id"], list(keep_ids))
            existing_vec, existing_off = existing_vec[mask], existing_off[mask]

    new_texts: list[str] = []
    new_offsets: list[tuple] = []
    for vid, md in on_disk.items():
        if vid in keep_ids:
            continue
        _, body = _parse_md(md)
        chunks = _chunks(body)
        for i, ch in enumerate(chunks):
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

    _save_atomic(out, vectors=vecs, offsets=offs, model=np.array(model_name))
    return out
