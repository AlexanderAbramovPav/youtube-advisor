from __future__ import annotations
import pathlib
import re
import subprocess
import tempfile
from .captions_api import CaptionsBlocked, NoCaptionsAvailable
from .._cookies import ytdlp_args

# yt-dlp stderr fragments that mean YouTube refused this client, not that the
# video lacks subtitles. Matched case-insensitively. "not a bot" comes from
# "Sign in to confirm you're not a bot" (the apostrophe may be curly, so it is
# left out); "Sign in to confirm your age" is an age gate and must not match.
_BLOCK_MARKERS = (
    "http error 429",
    "too many requests",
    "not a bot",
    "rate-limited by youtube",
    "try again later",
    "captcha challenge",
)


class YtDlpFailed(RuntimeError):
    """yt-dlp exited non-zero for a reason other than a block (video private
    or unavailable, age gate, network error). Not evidence of missing captions,
    so the orchestrator records it as failed instead of escalating to Whisper."""


def block_reason(stderr: str | None) -> str | None:
    """Return the yt-dlp error line that shows a YouTube block, else None."""
    for line in (stderr or "").splitlines():
        low = line.lower()
        if any(m in low for m in _BLOCK_MARKERS):
            return line.strip()
    return None


def _last_error_line(stderr: str | None) -> str:
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    errors = [ln for ln in lines if ln.startswith("ERROR")]
    return (errors or lines or ["no stderr"])[-1]


_TS = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})")


def _to_sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_vtt(text: str) -> list[dict]:
    segs, lines = [], text.splitlines()
    i = 0
    while i < len(lines):
        m = _TS.match(lines[i])
        if not m:
            i += 1
            continue
        start = _to_sec(*m.group(1, 2, 3, 4))
        end = _to_sec(*m.group(5, 6, 7, 8))
        i += 1
        body = []
        while i < len(lines) and lines[i].strip() and not _TS.match(lines[i]):
            body.append(lines[i])
            i += 1
        cleaned = re.sub(r"<[^>]+>", "", " ".join(body)).strip()
        if cleaned:
            segs.append({"start": start, "duration": end - start, "text": cleaned})
    return segs


def fetch_captions(
    video_id: str,
    lang_pref: str = "en",
    cookies: dict | None = None,
) -> tuple[list[dict], str]:
    with tempfile.TemporaryDirectory() as td:
        url = f"https://www.youtube.com/watch?v={video_id}"
        out_tpl = str(pathlib.Path(td) / "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp", *ytdlp_args(cookies),
            "--write-auto-sub", "--write-sub",
            "--sub-lang", f"{lang_pref}.*,en.*", "--sub-format", "vtt",
            "--skip-download", "-o", out_tpl, url,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        vtts = list(pathlib.Path(td).glob("*.vtt"))
        if not vtts:
            # Order matters: a block can come with exit code 0 (yt-dlp may
            # only warn about a failed subtitle download), and a non-zero
            # exit without a block is some other failure. Only a clean run
            # that wrote nothing means the video has no subtitles.
            reason = block_reason(r.stderr)
            if reason:
                raise CaptionsBlocked(video_id, reason, authenticated=bool(cookies))
            if r.returncode != 0:
                raise YtDlpFailed(f"yt-dlp exit {r.returncode}: {_last_error_line(r.stderr)}")
            raise NoCaptionsAvailable(video_id)

        # Prefer language match first, then larger size as tiebreaker. yt-dlp
        # may write several .vtt files (one per matching sub-lang plus auto);
        # naive size-only sort picks the wrong language when the preferred
        # one happens to be sparser than an alternative.
        def _lang_match(p: pathlib.Path) -> int:
            parts = p.stem.split(".")
            lang_tag = parts[-1] if len(parts) >= 2 else ""
            return 0 if lang_tag.startswith(lang_pref) else 1

        vtts.sort(key=lambda p: (_lang_match(p), -p.stat().st_size))
        segs = _parse_vtt(vtts[0].read_text())
        parts = vtts[0].stem.split(".")
        lang = parts[-1] if len(parts) >= 2 else "en"
        if not segs:
            raise NoCaptionsAvailable(video_id)
        return segs, lang
