import json
import pathlib
import pytest
import subprocess
from unittest.mock import patch
from youtube_advisor.fetch_channel import resolve_channel, _channel_url

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _mock_run(stdout: str, returncode: int = 0, stderr: str = ""):
    def fake_run(*a, **kw):
        return subprocess.CompletedProcess(args=a, returncode=returncode, stdout=stdout, stderr=stderr)
    return fake_run


def test_channel_url_normalization():
    assert _channel_url("@ycombinator") == "https://www.youtube.com/@ycombinator/videos"
    assert _channel_url("https://youtube.com/@x") == "https://youtube.com/@x"
    assert _channel_url("ycombinator") == "https://www.youtube.com/@ycombinator/videos"


def test_resolve_channel_parses_fixture():
    fixture = (FIXTURES / "yc_flat.json").read_text()
    with patch("youtube_advisor.fetch_channel.subprocess.run", _mock_run(fixture)):
        videos = resolve_channel("@ycombinator", limit=5)
    assert len(videos) >= 1
    v = videos[0]
    assert {"video_id", "title", "published_date", "channel", "length_seconds",
            "views", "description", "channel_id", "playlist_ids"} <= set(v.keys())
    # published_date format
    assert len(v["published_date"]) == 10 and v["published_date"][4] == "-"
    # playlist_ids is a set, empty for channel scan
    assert isinstance(v["playlist_ids"], set)
    assert v["playlist_ids"] == set()


def test_resolve_playlist_populates_playlist_ids():
    fixture = (FIXTURES / "yc_flat.json").read_text()  # any entries shape works
    with patch("youtube_advisor.fetch_channel.subprocess.run", _mock_run(fixture)):
        videos = resolve_channel("https://www.youtube.com/playlist?list=PLTESTID", limit=5)
    assert all("PLTESTID" in v["playlist_ids"] for v in videos)


def test_invalid_channel_raises():
    err = "ERROR: [generic] Unable to download webpage: HTTP Error 404"
    with patch("youtube_advisor.fetch_channel.subprocess.run", _mock_run("", returncode=1, stderr=err)):
        with pytest.raises(ValueError, match="Could not resolve"):
            resolve_channel("@nonexistent")


def test_empty_entries_raises():
    with patch("youtube_advisor.fetch_channel.subprocess.run", _mock_run('{"entries": []}', returncode=0)):
        with pytest.raises(ValueError, match="Could not resolve"):
            resolve_channel("@empty")


def test_resolve_channel_passes_antibot_args():
    """yt-dlp invocations must always include the anti-bot sleep/retry flags."""
    fixture = (FIXTURES / "yc_flat.json").read_text()
    seen = {}

    def fake_run(cmd, *a, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=fixture, stderr="")

    with patch("youtube_advisor.fetch_channel.subprocess.run", fake_run):
        resolve_channel("@yc", limit=5)

    cmd = seen["cmd"]
    assert "--sleep-interval" in cmd
    assert "--max-sleep-interval" in cmd
    assert "--retries" in cmd
    assert "--retry-sleep" in cmd


def test_resolve_channel_with_cookies_from_browser_adds_flag():
    fixture = (FIXTURES / "yc_flat.json").read_text()
    seen = {}

    def fake_run(cmd, *a, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=fixture, stderr="")

    with patch("youtube_advisor.fetch_channel.subprocess.run", fake_run):
        resolve_channel("@yc", limit=5, cookies={"browser": "chrome"})

    cmd = seen["cmd"]
    assert "--cookies-from-browser" in cmd
    idx = cmd.index("--cookies-from-browser")
    assert cmd[idx + 1] == "chrome"


def test_resolve_channel_with_cookies_file_adds_flag(tmp_path):
    fixture = (FIXTURES / "yc_flat.json").read_text()
    cookies_path = tmp_path / "c.txt"
    cookies_path.write_text("# Netscape\n")
    seen = {}

    def fake_run(cmd, *a, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=fixture, stderr="")

    with patch("youtube_advisor.fetch_channel.subprocess.run", fake_run):
        resolve_channel("@yc", limit=5, cookies={"file": cookies_path})

    cmd = seen["cmd"]
    assert "--cookies" in cmd
    idx = cmd.index("--cookies")
    assert cmd[idx + 1] == str(cookies_path)
