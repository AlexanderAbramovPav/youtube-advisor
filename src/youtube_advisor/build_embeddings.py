from __future__ import annotations
import numpy as np
import pathlib
import re
from fastembed import TextEmbedding

from ._md import parse_frontmatter

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


def build_embeddings(transcripts_dir: pathlib.Path, refs_dir: pathlib.Path,
                     model_name: str = "BAAI/bge-small-en-v1.5",
                     incremental: bool = False) -> pathlib.Path:
    refs_dir.mkdir(parents=True, exist_ok=True)
    out = refs_dir / "embeddings.npz"

    # Fast path: if nothing's changed AND the stored model matches the
    # requested one, skip. When the caller changes --multilingual (bge-m3,
    # 1024-d) vs default (bge-small, 384-d), the stored npz is stale and we
    # MUST rebuild — otherwise search.py reads vectors that don't match
    # corpus_meta.json's embedding_dim and shape errors at query time.
    if out.exists() and not incremental:
        emb_mtime = out.stat().st_mtime
        any_newer = any(p.stat().st_mtime > emb_mtime for p in transcripts_dir.glob("*.md"))
        if not any_newer:
            try:
                npz_check = np.load(out, allow_pickle=False)
                stored_model = str(npz_check["model"])
                stored_dim = npz_check["vectors"].shape[1] if npz_check["vectors"].size else None
            except Exception:
                stored_model, stored_dim = None, None
            if stored_model == model_name:
                # Optional dim sanity: only re-run if a stored dim contradicts
                # the requested model — caught later by FastEmbed on probe.
                return out

    model = TextEmbedding(model_name=model_name)

    existing_ids: set[str] = set()
    existing_vec, existing_off = None, None
    if incremental and out.exists():
        npz = np.load(out, allow_pickle=False)
        # If the stored model differs from the requested one, the existing
        # vectors live in the wrong embedding space — drop them and rebuild
        # everything rather than mixing incompatible vectors.
        if str(npz["model"]) == model_name:
            existing_vec, existing_off = npz["vectors"], npz["offsets"]
            existing_ids = set(existing_off["video_id"].tolist())

    new_texts: list[str] = []
    new_offsets: list[tuple] = []
    for md in sorted(transcripts_dir.glob("*.md")):
        fm, body = _parse_md(md)
        vid = fm.get("video_id")
        if not vid or vid in existing_ids:
            continue
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
        vecs, offs = existing_vec, existing_off
    else:
        # Determine dim from a single-token embed to get the right shape.
        probe = np.array(list(model.embed(["probe"])), dtype=np.float16)
        vecs = np.zeros((0, probe.shape[1]), dtype=np.float16)
        offs = np.zeros((0,), dtype=OFFSET_DTYPE)

    np.savez(out, vectors=vecs, offsets=offs, model=np.array(model_name))
    return out
