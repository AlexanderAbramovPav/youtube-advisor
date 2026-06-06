from __future__ import annotations
import subprocess, tempfile, pathlib, re
from .captions_api import NoCaptionsAvailable
from .._cookies import ytdlp_args

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
        subprocess.run(cmd, capture_output=True, text=True)
        vtts = list(pathlib.Path(td).glob("*.vtt"))
        if not vtts:
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
