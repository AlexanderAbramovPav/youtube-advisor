from __future__ import annotations
import concurrent.futures as cf
import subprocess, json, re
from datetime import datetime

from ._cookies import ytdlp_args


def _fetch_upload_date(video_id: str, cookies: dict | None = None) -> str | None:
    """Second-pass per-video metadata fetch for entries that `--flat-playlist`
    returned without a usable date. Returns YYYY-MM-DD or None on failure.

    --flat-playlist is fast but omits `upload_date` / `timestamp` for most
    channel entries. Without this fallback, `--since` filtering breaks and
    every transcript filename gets today's date.
    """
    proc = subprocess.run(
        ["yt-dlp", *ytdlp_args(cookies), "--skip-download",
         "--print", "%(upload_date)s",
         f"https://youtu.be/{video_id}"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    if len(out) == 8 and out.isdigit():
        return f"{out[:4]}-{out[4:6]}-{out[6:8]}"
    return None


def resolve_video(video_id: str, cookies: dict | None = None) -> dict | None:
    """Fetch metadata for a single video by id.

    Used for manual `--ids` entries that fall outside a capped channel
    listing (older videos, or videos from other channels). Returns the same
    entry shape as resolve_channel(), or None when the fetch fails or the
    video has no resolvable date.
    """
    proc = subprocess.run(
        ["yt-dlp", *ytdlp_args(cookies), "--skip-download", "-J",
         f"https://youtu.be/{video_id}"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        e = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    pub = e.get("upload_date")
    if not (pub and len(pub) == 8):
        return None
    return {
        "video_id": e.get("id") or video_id,
        "title": e.get("title") or "",
        "published_date": f"{pub[:4]}-{pub[4:6]}-{pub[6:8]}",
        "length_seconds": int(e.get("duration") or 0),
        "views": e.get("view_count") or 0,
        "description": e.get("description") or "",
        "channel": e.get("channel") or e.get("uploader") or "",
        "channel_id": e.get("channel_id") or "",
        "playlist_ids": set(),
    }


def _channel_url(handle_or_url: str) -> str:
    s = handle_or_url.strip()
    if s.startswith("http"):
        return s
    if s.startswith("@"):
        return f"https://www.youtube.com/{s}/videos"
    return f"https://www.youtube.com/@{s}/videos"


def resolve_channel(
    handle_or_url: str,
    limit: int | None = None,
    cookies: dict | None = None,
) -> list[dict]:
    url = _channel_url(handle_or_url)
    cmd = ["yt-dlp", *ytdlp_args(cookies), "--flat-playlist", "-J", url]
    if limit is not None:
        cmd += ["--playlist-end", str(limit)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(
            f"Could not resolve channel '{handle_or_url}': {proc.stderr.strip()[:200]}"
        )
    data = json.loads(proc.stdout)
    entries = data.get("entries") or []
    if not entries:
        raise ValueError(
            f"Could not resolve channel '{handle_or_url}': no videos returned"
        )

    # Channel-level fields live on the top-level data object for flat-playlist output
    top_channel = data.get("channel") or data.get("uploader") or ""
    top_channel_id = data.get("channel_id") or ""

    is_playlist = "playlist?list=" in url
    playlist_id = re.search(r"list=([\w-]+)", url).group(1) if is_playlist else None

    # First pass: assemble entries; mark those needing a date lookup.
    out = []
    needs_date = []  # references back into out[] by index
    for e in entries:
        if not e or e.get("_type") not in (None, "url"):
            continue
        vid = e.get("id")
        if not vid:
            continue

        # Date: prefer upload_date (YYYYMMDD string), fall back to timestamp
        # (unix epoch). If neither is present, defer to a parallel second pass.
        pub = e.get("upload_date")
        if pub and len(pub) == 8:
            published: str | None = f"{pub[:4]}-{pub[4:6]}-{pub[6:8]}"
        else:
            ts = e.get("timestamp")
            if ts:
                published = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            else:
                published = None  # resolved in second pass

        entry = {
            "video_id": vid,
            "title": e.get("title") or "",
            "published_date": published,
            "length_seconds": int(e.get("duration") or 0),
            "views": e.get("view_count") or 0,
            "description": e.get("description") or "",
            "channel": e.get("channel") or e.get("uploader") or top_channel,
            "channel_id": e.get("channel_id") or top_channel_id,
            "playlist_ids": {playlist_id} if playlist_id else set(),
        }
        out.append(entry)
        if published is None:
            needs_date.append(entry)

    # Second pass: parallel per-video date resolution (~8x faster than serial).
    if needs_date:
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            futures = {
                ex.submit(_fetch_upload_date, e["video_id"], cookies): e
                for e in needs_date
            }
            for f in cf.as_completed(futures):
                e = futures[f]
                try:
                    e["published_date"] = f.result()
                except Exception:
                    e["published_date"] = None

    # Drop entries we still couldn't date — better than mislabelling with today.
    return [e for e in out if e["published_date"]]
