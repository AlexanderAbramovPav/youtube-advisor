import os
import pathlib
import subprocess
import pytest
from unittest.mock import patch, MagicMock
from youtube_advisor.ingest import whisper as wh
from youtube_advisor.ingest.captions_api import NoCaptionsAvailable

# ----- _download_audio -----

def test_download_audio_returns_cached_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(wh, "AUDIO_CACHE", tmp_path)
    cached = tmp_path / "abc.wav"
    cached.write_bytes(b"fake wav data")
    result = wh._download_audio("abc")
    assert result == cached

def test_download_audio_invokes_yt_dlp_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(wh, "AUDIO_CACHE", tmp_path)
    def fake_run(cmd, *a, **kw):
        # Simulate yt-dlp writing the wav file
        out_path = tmp_path / "abc.wav"
        out_path.write_bytes(b"fake wav")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr("youtube_advisor.ingest.whisper.subprocess.run", fake_run)
    result = wh._download_audio("abc")
    assert result.exists()
    assert result.name == "abc.wav"

def test_download_audio_raises_on_yt_dlp_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(wh, "AUDIO_CACHE", tmp_path)
    def fake_run(*a, **kw):
        return subprocess.CompletedProcess(args=a, returncode=1, stdout="", stderr="error")
    monkeypatch.setattr("youtube_advisor.ingest.whisper.subprocess.run", fake_run)
    with pytest.raises(NoCaptionsAvailable):
        wh._download_audio("abc")

def test_download_audio_raises_when_file_not_written(tmp_path, monkeypatch):
    monkeypatch.setattr(wh, "AUDIO_CACHE", tmp_path)
    def fake_run(*a, **kw):
        return subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr="")
    monkeypatch.setattr("youtube_advisor.ingest.whisper.subprocess.run", fake_run)
    with pytest.raises(NoCaptionsAvailable):
        wh._download_audio("abc")

# ----- _transcribe_via_scriba -----

def test_scriba_parses_timestamped_md(tmp_path, monkeypatch):
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"")
    transcript_md = tmp_path / "test.transcript.md"
    transcript_md.write_text("[00:00:00] Hello world\n[00:00:05] Second segment\n")

    fake_scriba = tmp_path / "scriba"
    (fake_scriba / "scripts").mkdir(parents=True)
    (fake_scriba / "scripts/transcribe.sh").write_text("#!/bin/bash\n")
    monkeypatch.setenv("SCRIBA_DIR", str(fake_scriba))
    monkeypatch.setattr("youtube_advisor.ingest.whisper.subprocess.run",
                        lambda *a, **kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr=""))
    segs, lang = wh._transcribe_via_scriba(wav)
    assert lang == "auto"
    assert len(segs) == 2
    assert segs[0]["start"] == 0
    assert segs[0]["text"] == "Hello world"
    assert segs[1]["start"] == 5
    assert segs[1]["text"] == "Second segment"

def test_scriba_raises_when_transcribe_sh_missing(tmp_path, monkeypatch):
    wav = tmp_path / "test.wav"
    monkeypatch.setenv("SCRIBA_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(RuntimeError, match="scriba transcribe.sh not found"):
        wh._transcribe_via_scriba(wav)

def test_scriba_raises_when_no_parseable_lines(tmp_path, monkeypatch):
    """Bug #11: if scriba's output has no parseable [HH:MM:SS] lines, the
    parser must raise NoCaptionsAvailable so the orchestrator classifies the
    video as failed — not silently store an empty transcript."""
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"")
    transcript_md = tmp_path / "test.transcript.md"
    transcript_md.write_text("garbage line one\ngarbage line two\n")
    fake_scriba = tmp_path / "scriba"
    (fake_scriba / "scripts").mkdir(parents=True)
    (fake_scriba / "scripts/transcribe.sh").write_text("#!/bin/bash\n")
    monkeypatch.setenv("SCRIBA_DIR", str(fake_scriba))
    monkeypatch.setattr("youtube_advisor.ingest.whisper.subprocess.run",
                        lambda *a, **kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr=""))
    with pytest.raises(NoCaptionsAvailable):
        wh._transcribe_via_scriba(wav)


def test_scriba_transcript_path_computed_without_raising(tmp_path, monkeypatch):
    """Bug #1 (false-positive verification): the output path computation
    must work for arbitrary wav stems without raising. Earlier suggestion
    flagged `with_suffix('.transcript.md')` as a ValueError source — we use
    `with_name(stem + '.transcript.md')` to be portable across Pythons."""
    wav = tmp_path / "video.id.with.dots.wav"
    wav.write_bytes(b"")
    transcript_md = tmp_path / "video.id.with.dots.transcript.md"
    transcript_md.write_text("[00:00:00] ok\n")
    fake_scriba = tmp_path / "scriba"
    (fake_scriba / "scripts").mkdir(parents=True)
    (fake_scriba / "scripts/transcribe.sh").write_text("#!/bin/bash\n")
    monkeypatch.setenv("SCRIBA_DIR", str(fake_scriba))
    monkeypatch.setattr("youtube_advisor.ingest.whisper.subprocess.run",
                        lambda *a, **kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr=""))
    segs, _ = wh._transcribe_via_scriba(wav)
    assert len(segs) == 1


def test_scriba_skips_malformed_lines(tmp_path, monkeypatch):
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"")
    transcript_md = tmp_path / "test.transcript.md"
    transcript_md.write_text(
        "[bad] Malformed timestamp\n"
        "[00:00:10] Valid segment\n"
        "No timestamp at all\n"
    )
    fake_scriba = tmp_path / "scriba"
    (fake_scriba / "scripts").mkdir(parents=True)
    (fake_scriba / "scripts/transcribe.sh").write_text("#!/bin/bash\n")
    monkeypatch.setenv("SCRIBA_DIR", str(fake_scriba))
    monkeypatch.setattr("youtube_advisor.ingest.whisper.subprocess.run",
                        lambda *a, **kw: subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr=""))
    segs, _ = wh._transcribe_via_scriba(wav)
    assert len(segs) == 1
    assert segs[0]["start"] == 10

# ----- transcribe_audio dispatch -----

def test_transcribe_audio_dispatches_to_scriba_when_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_VIA_SCRIBA", "1")
    monkeypatch.setattr(wh, "_download_audio", lambda vid, cookies=None: tmp_path / "fake.wav")
    monkeypatch.setattr(wh, "_transcribe_via_scriba", lambda wav: ([{"start":0,"duration":0,"text":"scriba"}], "auto"))
    monkeypatch.setattr(wh, "_transcribe_via_faster_whisper",
                        lambda wav: pytest.fail("should not call faster-whisper"))
    segs, lang = wh.transcribe_audio("abc")
    assert lang == "auto"
    assert segs[0]["text"] == "scriba"

def test_transcribe_audio_dispatches_to_faster_whisper_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("WHISPER_VIA_SCRIBA", raising=False)
    monkeypatch.setattr(wh, "_download_audio", lambda vid, cookies=None: tmp_path / "fake.wav")
    monkeypatch.setattr(wh, "_transcribe_via_scriba", lambda wav: pytest.fail("should not call scriba"))
    monkeypatch.setattr(wh, "_transcribe_via_faster_whisper",
                        lambda wav: ([{"start":0,"duration":0,"text":"fw"}], "en"))
    segs, lang = wh.transcribe_audio("abc")
    assert lang == "en"
    assert segs[0]["text"] == "fw"

# ----- live smoke (skipped by default) -----

pytestmark_slow = pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="network + Whisper model")

@pytest.mark.slow
@pytestmark_slow
def test_transcribe_real_short_video(tmp_path):
    # Replace with a short permissive video ID when running RUN_SLOW=1
    segs, lang = wh.transcribe_audio("dQw4w9WgXcQ")
    assert len(segs) > 0
    assert lang
