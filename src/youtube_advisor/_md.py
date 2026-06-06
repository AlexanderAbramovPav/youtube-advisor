"""Markdown / transcript helpers shared across modules."""
from __future__ import annotations
import os
import pathlib
import random
import tempfile
from typing import Any

import yaml


def atomic_write_text(path: pathlib.Path, content: str) -> None:
    """Atomically write ``content`` to ``path`` via a same-directory tempfile.

    Cleans up the temp file on any write/replace failure so we never leak
    `.tmp-*` debris in transcripts/ or alongside .progress.json.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-",
                                suffix=path.suffix or ".tmp")
    os.close(fd)
    try:
        pathlib.Path(tmp).write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            pathlib.Path(tmp).unlink()
        except OSError:
            pass
        raise


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_text). Empty dict + original text if no fence."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    return yaml.safe_load(text[4:end]) or {}, text[end + 5:]


def video_id_from_filename(p: pathlib.Path) -> str:
    """Extract videoId from YYYY-MM-DD-{videoId}.md filename."""
    parts = p.stem.split("-")
    return "-".join(parts[3:]) if len(parts) >= 4 else p.stem


def dump_yaml(obj: Any) -> str:
    """Project-standard YAML dump: unicode-preserving, key-order-preserving."""
    return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)


def sample_transcripts(advisor_dir: pathlib.Path, n: int, *,
                       seed_salt: int = 0,
                       body_chars: int = 3500,
                       with_video_id: bool = False) -> list:
    """Deterministic transcript sampler. Returns list[str] of body excerpts,
    or list[(video_id, body)] when with_video_id=True."""
    files = sorted((advisor_dir / "transcripts").glob("*.md"))
    if not files:
        return []
    random.seed(len(files) if seed_salt == 0 else len(files) * 7 + seed_salt)
    picked = random.sample(files, min(n, len(files)))
    out = []
    for f in picked:
        text = f.read_text()
        body = text.split("\n---\n", 1)[1] if "\n---\n" in text else text
        excerpt = body[:body_chars]
        if with_video_id:
            out.append((video_id_from_filename(f), excerpt))
        else:
            out.append(excerpt)
    return out
