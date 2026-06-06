import pathlib
import pytest
from unittest.mock import patch, MagicMock
from youtube_advisor.ingest.orchestrator import run_ingest, _existing
from youtube_advisor.ingest.captions_api import NoCaptionsAvailable

def _vid(id_, pub="2024-01-01"):
    return {"video_id": id_, "title": f"v{id_}", "published_date": pub, "channel": "@x"}

# ----- dedup -----

def test_existing_returns_true_when_file_exists(tmp_path):
    (tmp_path / "2024-01-01-abc.md").write_text("hi")
    assert _existing(tmp_path, "abc")

def test_existing_returns_false_when_no_file(tmp_path):
    assert not _existing(tmp_path, "abc")

def test_dedup_skips_existing(tmp_path):
    (tmp_path / "2024-01-01-a.md").write_text("---\nvideo_id: a\n---\n")
    summary = run_ingest([_vid("a")], tmp_path)
    assert summary["skipped"] == 1
    assert summary["ingested"] == 0

# ----- cascade -----

def test_cascade_uses_tier1_when_available(tmp_path):
    segs = [{"start": 0, "duration": 1, "text": "hi"}]
    with patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions",
               return_value=(segs, "en")):
        summary = run_ingest([_vid("a")], tmp_path)
    assert summary["by_source"]["captions"] == 1
    assert summary["ingested"] == 1
    assert (tmp_path / "2024-01-01-a.md").exists()

def test_falls_back_to_tier2_when_tier1_raises(tmp_path):
    segs = [{"start": 0, "duration": 1, "text": "hi"}]
    with patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions",
               side_effect=NoCaptionsAvailable("a")), \
         patch("youtube_advisor.ingest.orchestrator.tier2.fetch_captions",
               return_value=(segs, "en")):
        summary = run_ingest([_vid("a")], tmp_path)
    assert summary["by_source"]["captions"] == 1
    assert summary["ingested"] == 1

def test_falls_back_to_tier3_when_tier1_and_tier2_raise(tmp_path):
    segs = [{"start": 0, "duration": 1, "text": "hi"}]
    with patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions",
               side_effect=NoCaptionsAvailable("a")), \
         patch("youtube_advisor.ingest.orchestrator.tier2.fetch_captions",
               side_effect=NoCaptionsAvailable("a")), \
         patch("youtube_advisor.ingest.orchestrator.tier3.transcribe_audio",
               return_value=(segs, "en")):
        summary = run_ingest([_vid("a")], tmp_path)
    assert summary["by_source"]["whisper-v3"] == 1
    assert summary["ingested"] == 1

def test_skip_whisper_marks_no_captions_as_failed(tmp_path):
    with patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions",
               side_effect=NoCaptionsAvailable("a")), \
         patch("youtube_advisor.ingest.orchestrator.tier2.fetch_captions",
               side_effect=NoCaptionsAvailable("a")):
        summary = run_ingest([_vid("a")], tmp_path, skip_whisper=True)
    assert summary["failed"] == 1
    assert summary["by_source"]["failed:no-captions"] == 1
    log_file = tmp_path.parent / "ingest.errors.log"
    assert log_file.exists()
    assert "a" in log_file.read_text()

def test_tier1_non_caption_exception_does_not_fall_through(tmp_path):
    """A non-NoCaptionsAvailable Tier 1 exception (network / parse) must NOT
    escalate to Tier 2 or Tier 3 — a transient ConnectionError isn't evidence
    the video has no captions, and burning Whisper compute on it is wrong.
    The video is recorded as failed:<ExceptionName> for later --reingest."""
    with patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions",
               side_effect=ConnectionError("network down")), \
         patch("youtube_advisor.ingest.orchestrator.tier2.fetch_captions") as t2, \
         patch("youtube_advisor.ingest.orchestrator.tier3.transcribe_audio") as t3:
        summary = run_ingest([_vid("a")], tmp_path)
    t2.assert_not_called()
    t3.assert_not_called()
    assert summary["failed"] == 1
    assert summary["by_source"]["failed:ConnectionError"] == 1


def test_tier1_and_tier2_network_errors_do_not_call_whisper(tmp_path):
    """Same guarantee but with Tier 2 also raising a non-NoCaptions exception."""
    with patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions",
               side_effect=ConnectionError("network down")), \
         patch("youtube_advisor.ingest.orchestrator.tier2.fetch_captions",
               side_effect=ConnectionError("still down")), \
         patch("youtube_advisor.ingest.orchestrator.tier3.transcribe_audio") as t3:
        summary = run_ingest([_vid("a")], tmp_path, skip_whisper=False)
    t3.assert_not_called()
    assert "failed:ConnectionError" in summary["by_source"]

def test_tier3_exception_logs_and_reports_failed(tmp_path):
    with patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions",
               side_effect=NoCaptionsAvailable("a")), \
         patch("youtube_advisor.ingest.orchestrator.tier2.fetch_captions",
               side_effect=NoCaptionsAvailable("a")), \
         patch("youtube_advisor.ingest.orchestrator.tier3.transcribe_audio",
               side_effect=RuntimeError("whisper crash")):
        summary = run_ingest([_vid("a")], tmp_path)
    assert summary["failed"] == 1
    assert "failed:RuntimeError" in summary["by_source"]
    log_file = tmp_path.parent / "ingest.errors.log"
    assert "RuntimeError" in log_file.read_text()

# ----- multi-video summary -----

def test_mixed_outcomes_summary(tmp_path):
    segs = [{"start": 0, "duration": 1, "text": "x"}]
    def tier1_side(vid):
        if vid == "a": return (segs, "en")
        raise NoCaptionsAvailable(vid)
    def tier2_side(vid):
        if vid == "b": return (segs, "en")
        raise NoCaptionsAvailable(vid)
    def tier3_side(vid):
        if vid == "c": return (segs, "en")
        raise RuntimeError("crash")
    with patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions", side_effect=tier1_side), \
         patch("youtube_advisor.ingest.orchestrator.tier2.fetch_captions", side_effect=tier2_side), \
         patch("youtube_advisor.ingest.orchestrator.tier3.transcribe_audio", side_effect=tier3_side):
        summary = run_ingest([_vid("a"), _vid("b"), _vid("c"), _vid("d")], tmp_path)
    assert summary["ingested"] == 3  # a, b, c
    assert summary["failed"] == 1    # d
    assert summary["by_source"]["captions"] == 2  # a (tier1), b (tier2)
    assert summary["by_source"]["whisper-v3"] == 1  # c
    assert summary["by_source"]["failed:RuntimeError"] == 1  # d

def test_empty_videos_returns_zeros(tmp_path):
    summary = run_ingest([], tmp_path)
    assert summary == {"ingested": 0, "skipped": 0, "failed": 0, "by_source": {}}


# ----- concurrency drops without cookies -----

def test_concurrency_drops_to_2_when_no_cookies(tmp_path):
    """When cookies are unset/empty, the captions pool max_workers must be 2."""
    seen = {}
    segs = [{"start": 0, "duration": 1, "text": "hi"}]

    real_executor = __import__("concurrent.futures").futures.ThreadPoolExecutor

    def capturing_executor(max_workers=None, *a, **kw):
        seen["max_workers"] = max_workers
        return real_executor(max_workers=max_workers, *a, **kw)

    with patch("youtube_advisor.ingest.orchestrator.cf.ThreadPoolExecutor",
               side_effect=capturing_executor), \
         patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions",
               return_value=(segs, "en")):
        run_ingest([_vid("a"), _vid("b"), _vid("c"), _vid("d"),
                    _vid("e"), _vid("f")], tmp_path)
    # 6 pending videos but cap is 2 when no cookies.
    assert seen["max_workers"] == 2


def test_concurrency_stays_8_when_cookies_provided(tmp_path):
    seen = {}
    segs = [{"start": 0, "duration": 1, "text": "hi"}]
    real_executor = __import__("concurrent.futures").futures.ThreadPoolExecutor

    def capturing_executor(max_workers=None, *a, **kw):
        seen["max_workers"] = max_workers
        return real_executor(max_workers=max_workers, *a, **kw)

    with patch("youtube_advisor.ingest.orchestrator.cf.ThreadPoolExecutor",
               side_effect=capturing_executor), \
         patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions",
               return_value=(segs, "en")):
        run_ingest([_vid("a"), _vid("b"), _vid("c"), _vid("d"),
                    _vid("e"), _vid("f"),
                    _vid("g"), _vid("h"), _vid("i"), _vid("j")],
                   tmp_path, cookies={"browser": "chrome"})
    assert seen["max_workers"] == 8


def test_progress_callback_called_per_video(tmp_path):
    """run_ingest must call progress.tick once per pending video so the
    sibling status reader (status.sh) can show live counters."""
    prog = MagicMock()
    videos = [{"video_id": f"v{i}", "title": f"t{i}",
               "published_date": "2024-01-01", "channel": "@y"}
              for i in range(3)]
    segs = [{"start": 0, "duration": 1, "text": "hi"}]
    with patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions",
               return_value=(segs, "en")):
        run_ingest(videos, tmp_path, progress=prog)
    assert prog.tick.call_count == 3
    # And ticks carry the video_id and source kwargs.
    seen_ids = sorted(
        call.kwargs.get("video_id") for call in prog.tick.call_args_list
    )
    assert seen_ids == ["v0", "v1", "v2"]


def test_progress_callback_default_none_is_safe(tmp_path):
    """Calling run_ingest without progress= must remain a no-op (no crash)."""
    segs = [{"start": 0, "duration": 1, "text": "hi"}]
    with patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions",
               return_value=(segs, "en")):
        summary = run_ingest(
            [{"video_id": "a", "title": "t", "published_date": "2024-01-01",
              "channel": "@y"}],
            tmp_path,
        )
    assert summary["ingested"] == 1


def test_cookies_passed_through_to_tier1(tmp_path):
    """When cookies are provided, fetch_captions receives them as kwarg."""
    segs = [{"start": 0, "duration": 1, "text": "hi"}]
    captured = []

    def fake(vid, cookies=None):
        captured.append(cookies)
        return (segs, "en")

    with patch("youtube_advisor.ingest.orchestrator.tier1.fetch_captions",
               side_effect=fake):
        run_ingest([_vid("a")], tmp_path, cookies={"browser": "firefox"})
    assert captured == [{"browser": "firefox"}]
