from __future__ import annotations
import pathlib
from datetime import datetime, timezone

from .._md import atomic_write_text, dump_yaml


def _hms(sec: float) -> str:
    s = int(sec)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def write_transcript(out_dir: pathlib.Path, meta: dict, segs: list[dict],
                     transcript_source: str, lang: str) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fm = {
        "title": meta.get("title", ""),
        "video_url": f"https://www.youtube.com/watch?v={meta['video_id']}",
        "video_id": meta["video_id"],
        "channel": meta.get("channel", ""),
        "channel_id": meta.get("channel_id", ""),
        "published_date": meta["published_date"],
        "length_seconds": int(meta.get("length_seconds") or 0),
        "length_minutes": int((meta.get("length_seconds") or 0) // 60),
        "views": int(meta.get("views") or 0),
        "description": meta.get("description", "")[:500],
        "keywords": meta.get("keywords", []),
        "transcript_source": transcript_source,
        "transcript_language": lang,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chapters": meta.get("chapters", []),
    }
    body_lines = [f"[{_hms(s['start'])}] {s['text']}" for s in segs if s.get("text")]
    content = (
        "---\n"
        + dump_yaml(fm)
        + "---\n\n"
        + "\n".join(body_lines)
        + "\n"
    )

    fname = f"{meta['published_date']}-{meta['video_id']}.md"
    final = out_dir / fname
    atomic_write_text(final, content)
    return final
