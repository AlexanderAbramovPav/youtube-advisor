from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
import logging
import re
from typing import Iterable

DEFAULT_TITLE_EXCLUDE = r"#?shorts\b"

log = logging.getLogger(__name__)


def _compile_or_literal(pattern: str | None) -> re.Pattern | None:
    """Compile a user-supplied title pattern; fall back to literal-substring
    matching when the input isn't a valid regex. This prevents bootstrap from
    crashing after the most expensive step on inputs like `C++ tricks`."""
    if not pattern:
        return None
    try:
        return re.compile(pattern, re.I)
    except re.error as e:
        log.warning("Invalid title regex %r (%s); treating as literal substring.",
                    pattern, e)
        return re.compile(re.escape(pattern), re.I)


@dataclass
class FilterConfig:
    since: date | None = None
    until: date | None = None
    max: int = 500
    title_include: str | None = None
    title_exclude: str | None = DEFAULT_TITLE_EXCLUDE
    playlist_ids: set[str] = field(default_factory=set)
    manual_ids: set[str] = field(default_factory=set)

def apply_filters(candidates: Iterable[dict], cfg: FilterConfig) -> list[dict]:
    manual_kept: list[dict] = []
    rest: list[dict] = []
    for v in candidates:
        if v["video_id"] in cfg.manual_ids:
            manual_kept.append(v)
        else:
            rest.append(v)

    inc_re = _compile_or_literal(cfg.title_include)
    exc_re = _compile_or_literal(cfg.title_exclude)

    def keep(v: dict) -> bool:
        # Malformed published_date is a system-boundary failure (yt-dlp
        # response, manual --ids feed, etc.) — drop the video with a warning
        # rather than crashing the entire run.
        try:
            pub = date.fromisoformat(v["published_date"])
        except (ValueError, TypeError, KeyError):
            log.warning("Dropping video %s: malformed published_date %r",
                        v.get("video_id"), v.get("published_date"))
            return False
        if cfg.since and pub < cfg.since: return False
        if cfg.until and pub > cfg.until: return False
        if inc_re and not inc_re.search(v["title"]): return False
        if exc_re and exc_re.search(v["title"]): return False
        if cfg.playlist_ids and not (cfg.playlist_ids & v.get("playlist_ids", set())):
            return False
        return True

    kept = [v for v in rest if keep(v)]
    kept.sort(key=lambda v: v["published_date"], reverse=True)
    kept = kept[: cfg.max]

    seen = {v["video_id"] for v in kept}
    out = kept + [m for m in manual_kept if m["video_id"] not in seen]
    return out
