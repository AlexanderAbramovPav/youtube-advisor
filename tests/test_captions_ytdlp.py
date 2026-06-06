import pathlib
import subprocess
import pytest
from unittest.mock import patch
from youtube_advisor.ingest.captions_ytdlp import _parse_vtt, fetch_captions
from youtube_advisor.ingest.captions_api import NoCaptionsAvailable


def test_parses_basic_vtt(fixtures_dir):
    segs = _parse_vtt((fixtures_dir / "sample.vtt").read_text())
    assert len(segs) == 3
    assert segs[0]["start"] == 0.0
    assert segs[0]["duration"] == 2.0
    assert segs[0]["text"] == "Hello world"
    assert "<c>" not in segs[1]["text"]
    assert segs[1]["text"] == "Welcome back"
    assert segs[2]["text"] == "Today we discuss"


def test_parse_vtt_empty_returns_empty():
    assert _parse_vtt("WEBVTT\n\n") == []


def test_parse_vtt_skips_empty_cues():
    text = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n\n00:00:01.000 --> 00:00:02.000\nHi\n"
    segs = _parse_vtt(text)
    assert len(segs) == 1
    assert segs[0]["text"] == "Hi"


def test_fetch_captions_raises_when_no_vtt_produced(tmp_path):
    """yt-dlp 'succeeded' but emitted no .vtt files -> NoCaptionsAvailable."""
    def fake_run(*a, **kw):
        return subprocess.CompletedProcess(args=a, returncode=0, stdout="", stderr="")
    with patch("youtube_advisor.ingest.captions_ytdlp.subprocess.run", side_effect=fake_run):
        with pytest.raises(NoCaptionsAvailable):
            fetch_captions("abc123")


def test_fetch_captions_parses_written_vtt(monkeypatch, fixtures_dir):
    """Simulate yt-dlp writing a VTT into its tempdir; verify parsing pipeline."""
    sample = (fixtures_dir / "sample.vtt").read_text()

    def fake_run(cmd, *a, **kw):
        out_tpl = cmd[cmd.index("-o") + 1]
        td = pathlib.Path(out_tpl).parent
        (td / "abc123.en.vtt").write_text(sample)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr("youtube_advisor.ingest.captions_ytdlp.subprocess.run", fake_run)

    segs, lang = fetch_captions("abc123")
    assert lang == "en"
    assert len(segs) == 3
    assert segs[0]["text"] == "Hello world"


def test_fetch_captions_prefers_lang_match_over_size(monkeypatch):
    """When yt-dlp writes VTTs in multiple languages, the requested language
    must win even when a non-matching language has a smaller (or larger) file.
    Earlier behaviour sorted by size only and silently picked Spanish when the
    user wanted English."""
    spanish_short = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHola\n"
    english_long = "WEBVTT\n\n" + "\n\n".join(
        f"00:00:{i:02d}.000 --> 00:00:{i+1:02d}.000\nLine {i}" for i in range(30))

    def fake_run(cmd, *a, **kw):
        out_tpl = cmd[cmd.index("-o") + 1]
        td = pathlib.Path(out_tpl).parent
        (td / "abc.es.vtt").write_text(spanish_short)
        (td / "abc.en.vtt").write_text(english_long)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr("youtube_advisor.ingest.captions_ytdlp.subprocess.run", fake_run)

    segs, lang = fetch_captions("abc", lang_pref="en")
    assert lang == "en"
    # English long file has 30 cues.
    assert len(segs) == 30


def test_fetch_captions_picks_larger_within_same_language(monkeypatch):
    """Within the same language, prefer the larger VTT — usually the
    fuller/auto track wins because it contains the actual transcript body."""
    short = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nFragment\n"
    longer = "WEBVTT\n\n" + "\n\n".join(
        f"00:00:{i:02d}.000 --> 00:00:{i+1:02d}.000\nFull cue {i}" for i in range(5))

    def fake_run(cmd, *a, **kw):
        out_tpl = cmd[cmd.index("-o") + 1]
        td = pathlib.Path(out_tpl).parent
        (td / "abc.en.vtt").write_text(short)
        (td / "abc.en-US.vtt").write_text(longer)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    monkeypatch.setattr("youtube_advisor.ingest.captions_ytdlp.subprocess.run", fake_run)

    segs, _ = fetch_captions("abc", lang_pref="en")
    assert len(segs) == 5
