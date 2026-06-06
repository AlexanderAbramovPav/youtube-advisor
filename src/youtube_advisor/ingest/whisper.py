from __future__ import annotations
import subprocess, os, pathlib
from .captions_api import NoCaptionsAvailable
from .._cookies import ytdlp_args

AUDIO_CACHE = pathlib.Path.home() / ".cache" / "youtube-advisor" / "audio"

def _download_audio(video_id: str, cookies: dict | None = None) -> pathlib.Path:
    AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
    out = AUDIO_CACHE / f"{video_id}.wav"
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = ["yt-dlp", *ytdlp_args(cookies),
           "-x", "--audio-format", "wav", "--audio-quality", "0",
           "-o", str(AUDIO_CACHE / f"{video_id}.%(ext)s"), url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise NoCaptionsAvailable(video_id)
    if not out.exists():
        raise NoCaptionsAvailable(video_id)
    return out

def _transcribe_via_scriba(wav: pathlib.Path) -> tuple[list[dict], str]:
    scriba = os.getenv("SCRIBA_DIR") or str(pathlib.Path.home() / ".claude/skills/scriba")
    script = pathlib.Path(scriba) / "scripts/transcribe.sh"
    if not script.exists():
        raise RuntimeError("WHISPER_VIA_SCRIBA=1 but scriba transcribe.sh not found")
    # Compute the expected output path without relying on .with_suffix's
    # behaviour for multi-dot suffixes (which differs across Python versions).
    out_md = wav.with_name(wav.stem + ".transcript.md")
    subprocess.run(["bash", str(script), str(wav)], check=True)
    segs = []
    for line in out_md.read_text().splitlines():
        if line.startswith("[") and "]" in line:
            ts, text = line.split("]", 1)
            try:
                h, m, s = ts[1:].split(":")
                start = int(h) * 3600 + int(m) * 60 + int(s)
            except (ValueError, AttributeError):
                continue
            segs.append({"start": start, "duration": 0, "text": text.strip()})
    if not segs:
        # No parseable [HH:MM:SS] lines means the scriba output was malformed
        # (or empty). Raising lets the orchestrator classify the video as
        # failed:NoCaptionsAvailable rather than silently writing an empty
        # transcript.
        raise NoCaptionsAvailable(wav.stem)
    return segs, "auto"

def _transcribe_via_faster_whisper(wav: pathlib.Path) -> tuple[list[dict], str]:
    from faster_whisper import WhisperModel
    model = WhisperModel("medium", device="auto", compute_type="int8")
    segments_iter, info = model.transcribe(str(wav), beam_size=1)
    segs = [{"start": s.start, "duration": s.end - s.start, "text": s.text.strip()}
            for s in segments_iter]
    return segs, info.language

def transcribe_audio(
    video_id: str, cookies: dict | None = None
) -> tuple[list[dict], str]:
    wav = _download_audio(video_id, cookies=cookies)
    if os.getenv("WHISPER_VIA_SCRIBA") == "1":
        return _transcribe_via_scriba(wav)
    return _transcribe_via_faster_whisper(wav)
