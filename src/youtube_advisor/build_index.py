from __future__ import annotations
import json, pathlib

from ._md import parse_frontmatter

# Back-compat alias for callers/tests that import the underscore name.
_parse_frontmatter = parse_frontmatter

def build_index(transcripts_dir: pathlib.Path, refs_dir: pathlib.Path,
                incremental: bool = False) -> pathlib.Path:
    refs_dir.mkdir(parents=True, exist_ok=True)
    out = refs_dir / "index.json"
    existing: dict[str, dict] = {}
    if incremental and out.exists():
        existing = {v["video_id"]: v for v in json.loads(out.read_text())["videos"]}

    videos: list[dict] = []
    out_mtime = out.stat().st_mtime if (incremental and out.exists()) else 0.0
    for md in sorted(transcripts_dir.glob("*.md")):
        fm, body = parse_frontmatter(md.read_text())
        vid = fm.get("video_id")
        if not vid:
            continue
        if incremental and vid in existing and md.stat().st_mtime <= out_mtime:
            videos.append(existing[vid])
            continue
        videos.append({
            "video_id": vid,
            "title": fm.get("title", ""),
            "video_url": fm.get("video_url"),
            "channel": fm.get("channel"),
            "published_date": fm.get("published_date"),
            "length_minutes": fm.get("length_minutes", 0),
            "views": fm.get("views", 0),
            "keywords": fm.get("keywords", []),
            "transcript_source": fm.get("transcript_source"),
            "summary": body.strip()[:200],
        })
    videos.sort(key=lambda v: v["published_date"] or "", reverse=True)
    payload = {"version": 1, "n_videos": len(videos), "videos": videos}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out
