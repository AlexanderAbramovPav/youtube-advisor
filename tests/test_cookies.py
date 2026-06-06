"""Tests for the cookies-resolution helper used to bypass YT anti-bot."""
from __future__ import annotations
import os
import pathlib
import pytest

from youtube_advisor import _cookies as ck


# ----- resolve_cookies precedence -----

def test_explicit_browser_wins(monkeypatch):
    monkeypatch.setenv("YOUTUBE_ADVISOR_COOKIES_BROWSER", "firefox")
    monkeypatch.setenv("YOUTUBE_ADVISOR_COOKIES_FILE", "/nope.txt")
    out = ck.resolve_cookies(from_browser="chrome")
    assert out == {"browser": "chrome"}


def test_explicit_file_wins_when_no_browser(tmp_path, monkeypatch):
    cf = tmp_path / "c.txt"
    cf.write_text("# Netscape\n")
    monkeypatch.setenv("YOUTUBE_ADVISOR_COOKIES_BROWSER", "firefox")
    out = ck.resolve_cookies(cookies_file=str(cf))
    # Explicit file wins even though env browser is set, because we passed an
    # explicit kwarg.
    assert out == {"file": cf}


def test_env_browser_used_when_no_explicit(monkeypatch, tmp_path):
    # Disable auto-discovery so it doesn't shadow.
    monkeypatch.setattr(ck, "DEFAULT_COOKIES_PATH", tmp_path / "nope.txt")
    monkeypatch.setenv("YOUTUBE_ADVISOR_COOKIES_BROWSER", "safari")
    monkeypatch.delenv("YOUTUBE_ADVISOR_COOKIES_FILE", raising=False)
    out = ck.resolve_cookies()
    assert out == {"browser": "safari"}


def test_env_file_used_when_no_explicit_no_env_browser(tmp_path, monkeypatch):
    cf = tmp_path / "c.txt"
    cf.write_text("# Netscape\n")
    monkeypatch.setattr(ck, "DEFAULT_COOKIES_PATH", tmp_path / "nope.txt")
    monkeypatch.delenv("YOUTUBE_ADVISOR_COOKIES_BROWSER", raising=False)
    monkeypatch.setenv("YOUTUBE_ADVISOR_COOKIES_FILE", str(cf))
    out = ck.resolve_cookies()
    assert out == {"file": cf}


def test_auto_discovery_last(tmp_path, monkeypatch):
    cf = tmp_path / "auto.txt"
    cf.write_text("# Netscape\n")
    monkeypatch.setattr(ck, "DEFAULT_COOKIES_PATH", cf)
    monkeypatch.delenv("YOUTUBE_ADVISOR_COOKIES_BROWSER", raising=False)
    monkeypatch.delenv("YOUTUBE_ADVISOR_COOKIES_FILE", raising=False)
    out = ck.resolve_cookies()
    assert out == {"file": cf}


def test_empty_when_nothing_set(tmp_path, monkeypatch):
    monkeypatch.setattr(ck, "DEFAULT_COOKIES_PATH", tmp_path / "nope.txt")
    monkeypatch.delenv("YOUTUBE_ADVISOR_COOKIES_BROWSER", raising=False)
    monkeypatch.delenv("YOUTUBE_ADVISOR_COOKIES_FILE", raising=False)
    assert ck.resolve_cookies() == {}


def test_unknown_browser_raises():
    with pytest.raises(ValueError, match="Unknown browser"):
        ck.resolve_cookies(from_browser="netscape")


def test_missing_cookies_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ck.resolve_cookies(cookies_file=str(tmp_path / "absent.txt"))


# ----- arg builders -----

def test_ytdlp_cookie_args_browser():
    assert ck.ytdlp_cookie_args({"browser": "chrome"}) == \
        ["--cookies-from-browser", "chrome"]


def test_ytdlp_cookie_args_file(tmp_path):
    f = tmp_path / "c.txt"
    assert ck.ytdlp_cookie_args({"file": f}) == ["--cookies", str(f)]


def test_ytdlp_cookie_args_empty():
    assert ck.ytdlp_cookie_args({}) == []


def test_ytdlp_args_includes_antibot_when_no_cookies():
    args = ck.ytdlp_args(None)
    assert "--sleep-interval" in args
    assert "--max-sleep-interval" in args
    assert "--retries" in args
    assert "--retry-sleep" in args


def test_ytdlp_args_combines_cookies_and_antibot():
    args = ck.ytdlp_args({"browser": "firefox"})
    assert args[:2] == ["--cookies-from-browser", "firefox"]
    assert "--sleep-interval" in args


# ----- JS-challenge solver + PO-token tolerance -----

def test_antibot_args_include_extractor_args_for_pot_tolerance():
    """`--extractor-args youtube:formats=missing_pot` makes yt-dlp tolerate
    missing PO tokens (a common 'Sign in to confirm you're not a bot' trigger).
    Must always be present regardless of installed yt-dlp version."""
    assert "--extractor-args" in ck.YTDLP_ANTIBOT_ARGS
    idx = ck.YTDLP_ANTIBOT_ARGS.index("--extractor-args")
    assert ck.YTDLP_ANTIBOT_ARGS[idx + 1] == "youtube:formats=missing_pot"


def test_antibot_args_include_jsc_solver_when_supported():
    """When the installed yt-dlp recognises `--remote-components`, the antibot
    list must include `--remote-components ejs:github` to bypass YouTube's JS
    challenge. On older yt-dlp builds without the flag, the args are silently
    omitted (degraded but functional)."""
    if ck._HAS_REMOTE_COMPONENTS:
        assert "--remote-components" in ck.YTDLP_ANTIBOT_ARGS
        idx = ck.YTDLP_ANTIBOT_ARGS.index("--remote-components")
        assert ck.YTDLP_ANTIBOT_ARGS[idx + 1] == "ejs:github"
    else:
        assert "--remote-components" not in ck.YTDLP_ANTIBOT_ARGS
