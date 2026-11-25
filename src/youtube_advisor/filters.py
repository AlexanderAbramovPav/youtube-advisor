from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
import logging
import math
import re
from typing import Iterable

DEFAULT_TITLE_EXCLUDE = r"#?shorts\b"

log = logging.getLogger(__name__)

_DURATION_HMS_RE = re.compile(
    r"^(?:(?P<h>[0-9]+)h)?(?:(?P<m>[0-9]+)m)?(?:(?P<s>[0-9]+)s)?$",
    re.IGNORECASE,
)
_DURATION_MMSS_RE = re.compile(r"^(?P<m>[0-9]+):(?P<s>[0-9]{2})$")
_DURATION_HHMMSS_RE = re.compile(r"^(?P<h>[0-9]+):(?P<m>[0-9]{2}):(?P<s>[0-9]{2})$")


def parse_duration(value: str) -> int:
    """Parse a duration string into non-negative whole seconds.

    Accepted formats: plain seconds ('3600'), units ('90m', '2h', '1h30m',
    '45s', '1h30m15s'), 'MM:SS', and 'H:MM:SS'.
    """
    if not isinstance(value, str):
        raise ValueError(f"Invalid duration {value!r}: expected a string.")
    s = value.strip()
    if not s:
        raise ValueError("Invalid duration '': value cannot be empty.")
    if s.startswith("-"):
        raise ValueError(f"Invalid duration {value!r}: duration cannot be negative.")
    if re.fullmatch(r"[0-9]+", s):
        return int(s)
    m = _DURATION_HHMMSS_RE.fullmatch(s)
    if m:
        hours, mins, secs = int(m["h"]), int(m["m"]), int(m["s"])
        if mins >= 60 or secs >= 60:
            raise ValueError(
                f"Invalid duration {value!r}: minutes and seconds in H:MM:SS must be < 60."
            )
        return hours * 3600 + mins * 60 + secs
    m = _DURATION_MMSS_RE.fullmatch(s)
    if m:
        mins, secs = int(m["m"]), int(m["s"])
        if secs >= 60:
            raise ValueError(
                f"Invalid duration {value!r}: seconds in MM:SS must be < 60."
            )
        return mins * 60 + secs
    m = _DURATION_HMS_RE.fullmatch(s)
    if m and any(m.group(k) is not None for k in ("h", "m", "s")):
        hours = int(m["h"] or 0)
        mins = int(m["m"] or 0)
        secs = int(m["s"] or 0)
        return hours * 3600 + mins * 60 + secs
    raise ValueError(
        f"Invalid duration {value!r}: use seconds (3600), units (90m, 2h, 1h30m), MM:SS, or H:MM:SS."
    )


def format_duration(seconds: int) -> str:
    """Format whole seconds compactly (e.g. '1h', '20m', '1h30m', '45s')."""
    if seconds == 0:
        return "0s"
    hours, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if mins:
        parts.append(f"{mins}m")
    if secs:
        parts.append(f"{secs}s")
    return "".join(parts)


def format_duration_bounds(min_duration: int | None, max_duration: int | None) -> str | None:
    """Format optional duration bounds like '≥ 1h', '≤ 20m', or '≥ 20m, ≤ 1h'."""
    parts: list[str] = []
    if min_duration is not None:
        parts.append(f"≥ {format_duration(min_duration)}")
    if max_duration is not None:
        parts.append(f"≤ {format_duration(max_duration)}")
    return ", ".join(parts) if parts else None


def _known_length(v: dict) -> float | None:
    raw = v.get("length_seconds")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    if not math.isfinite(raw) or raw <= 0:
        return None
    return raw


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
    min_duration: int | None = None
    max_duration: int | None = None
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
    has_duration_bound = cfg.min_duration is not None or cfg.max_duration is not None
    unknown_duration_count = 0

    def keep(v: dict) -> bool:
        nonlocal unknown_duration_count
        # Malformed published_date is a system-boundary failure (yt-dlp
        # response, manual --ids feed, etc.) — drop the video with a warning
        # rather than crashing the entire run.
        try:
            pub = date.fromisoformat(v["published_date"])
        except (ValueError, TypeError, KeyError):
            log.warning("Dropping video %s: malformed published_date %r",
                        v.get("video_id"), v.get("published_date"))
            return False
        if cfg.since and pub < cfg.since:
            return False
        if cfg.until and pub > cfg.until:
            return False
        if inc_re and not inc_re.search(v["title"]):
            return False
        if exc_re and exc_re.search(v["title"]):
            return False
        if cfg.playlist_ids and not (cfg.playlist_ids & v.get("playlist_ids", set())):
            return False
        if has_duration_bound:
            length = _known_length(v)
            if length is None:
                unknown_duration_count += 1
            else:
                if cfg.min_duration is not None and length < cfg.min_duration:
                    return False
                if cfg.max_duration is not None and length > cfg.max_duration:
                    return False
        return True

    kept = [v for v in rest if keep(v)]
    if unknown_duration_count:
        log.warning(
            "Kept %d video(s) with unknown length (cannot check duration bounds).",
            unknown_duration_count,
        )
    kept.sort(key=lambda v: v["published_date"], reverse=True)
    kept = kept[: cfg.max]

    seen = {v["video_id"] for v in kept}
    out = kept + [m for m in manual_kept if m["video_id"] not in seen]
    return out
