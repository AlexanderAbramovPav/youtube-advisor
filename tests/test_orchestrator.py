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
    The video is recorded as failed:<ExceptionName>; the next update retries it."""
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
        if vid == "a":
            return (segs, "en")
        raise NoCaptionsAvailable(vid)
    def tier2_side(vid):
        if vid == "b":
            return (segs, "en")
        raise NoCaptionsAvailable(vid)
    def tier3_side(vid):
        if vid == "c":
            return (segs, "en")
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


# ----- YouTube blocks (429 / IpBlocked / bot check) -----
#
# A block is not "no captions": it must never reach Whisper, it stops the run
# (no further YouTube requests), and every video it did not reach stays
# pending so the next `update` retries it.

from youtube_advisor.ingest.captions_api import CaptionsBlocked  # noqa: E402

_SEGS = [{"start": 0, "duration": 1, "text": "hi"}]
_T1 = "youtube_advisor.ingest.orchestrator.tier1.fetch_captions"
_T2 = "youtube_advisor.ingest.orchestrator.tier2.fetch_captions"
_T3 = "youtube_advisor.ingest.orchestrator.tier3.transcribe_audio"
_SLEEP = "youtube_advisor.ingest.orchestrator.time.sleep"


def _ids(mock):
    return [c.args[0] for c in mock.call_args_list]


def _errors_log(tmp_path):
    # run_ingest writes the log next to out_dir; the block tests use
    # tmp_path/"tx" as out_dir so the log is per-test, not tmp_path.parent.
    p = tmp_path / "ingest.errors.log"
    return p.read_text() if p.exists() else ""


def test_tier1_block_stops_run_and_skips_tier2_and_whisper(tmp_path):
    with patch(_SLEEP), \
         patch(_T1, side_effect=CaptionsBlocked("a", "IpBlocked")) as t1, \
         patch(_T2) as t2, patch(_T3) as t3:
        summary = run_ingest([_vid("a"), _vid("b"), _vid("c")], tmp_path / 'tx',
                             max_concurrent=1)
    assert _ids(t1) == ["a"]          # b and c never reach YouTube
    t2.assert_not_called()            # no fall-through inside the blocked video
    t3.assert_not_called()            # no Whisper audio download
    assert summary["by_source"] == {"failed:blocked": 1}
    assert summary["failed"] == 1 and summary["ingested"] == 0
    assert summary["blocked"] is True
    assert summary["not_attempted"] == 2
    assert "IpBlocked" in summary["block_detail"]
    assert _errors_log(tmp_path) == "a\tfailed:blocked\n"
    assert list((tmp_path / "tx").glob("*.md")) == []


def test_block_mid_run_keeps_done_videos_and_leaves_rest_pending(tmp_path):
    def t1_side(vid):
        if vid == "b":
            raise CaptionsBlocked(vid, "IpBlocked")
        return (_SEGS, "en")
    with patch(_SLEEP), patch(_T1, side_effect=t1_side) as t1, patch(_T3) as t3:
        summary = run_ingest([_vid("a"), _vid("b"), _vid("c"), _vid("d")],
                             tmp_path / "tx", max_concurrent=1)
    assert _ids(t1) == ["a", "b"]
    t3.assert_not_called()
    assert summary["by_source"] == {"captions": 1, "failed:blocked": 1}
    assert summary["not_attempted"] == 2
    assert (tmp_path / "tx" / "2024-01-01-a.md").exists()
    assert _errors_log(tmp_path) == "b\tfailed:blocked\n"


def test_tier2_block_does_not_escalate_to_whisper(tmp_path):
    with patch(_SLEEP), \
         patch(_T1, side_effect=NoCaptionsAvailable("a")) as t1, \
         patch(_T2, side_effect=CaptionsBlocked("a", "HTTP Error 429")) as t2, \
         patch(_T3) as t3:
        summary = run_ingest([_vid("a"), _vid("b")], tmp_path / 'tx', max_concurrent=1)
    assert _ids(t1) == ["a"] and _ids(t2) == ["a"]
    t3.assert_not_called()
    assert summary["by_source"] == {"failed:blocked": 1}
    assert summary["not_attempted"] == 1


def test_real_no_captions_still_goes_to_whisper(tmp_path):
    with patch(_SLEEP), \
         patch(_T1, side_effect=NoCaptionsAvailable("a")), \
         patch(_T2, side_effect=NoCaptionsAvailable("a")), \
         patch(_T3, return_value=(_SEGS, "en")) as t3:
        summary = run_ingest([_vid("a")], tmp_path / 'tx')
    assert _ids(t3) == ["a"]
    assert summary["by_source"] == {"whisper-v3": 1}
    assert "blocked" not in summary and "not_attempted" not in summary


def test_block_in_captions_pass_cancels_whisper_pass(tmp_path):
    """a has no captions (would go to Whisper), b is blocked: the Whisper pass
    must not run, because it downloads audio from YouTube."""
    def t1_side(vid):
        if vid == "b":
            raise CaptionsBlocked(vid, "IpBlocked")
        raise NoCaptionsAvailable(vid)
    with patch(_SLEEP), patch(_T1, side_effect=t1_side) as t1, \
         patch(_T2, side_effect=NoCaptionsAvailable("a")), patch(_T3) as t3:
        summary = run_ingest([_vid("a"), _vid("b")], tmp_path / 'tx', max_concurrent=1)
    t3.assert_not_called()
    assert _ids(t1) == ["a", "b"]
    assert summary["by_source"] == {"failed:blocked": 1}
    assert summary["not_attempted"] == 1   # a: pending, retried next run
    assert _errors_log(tmp_path) == "b\tfailed:blocked\n"


def test_whisper_download_block_stops_remaining_whisper_videos(tmp_path):
    with patch(_SLEEP), \
         patch(_T1, side_effect=NoCaptionsAvailable("x")) as t1, \
         patch(_T2, side_effect=NoCaptionsAvailable("x")), \
         patch(_T3, side_effect=CaptionsBlocked("a", "HTTP Error 429")) as t3:
        summary = run_ingest([_vid("a"), _vid("b")], tmp_path / 'tx', max_concurrent=1)
    assert _ids(t3) == ["a"]
    # Pass 1 tried a and b; pass 2 re-tried a's captions, then stopped.
    assert _ids(t1) == ["a", "b", "a"]
    assert summary["by_source"] == {"failed:blocked": 1}
    assert summary["not_attempted"] == 1


def test_blocked_and_untried_videos_are_retried_by_next_run(tmp_path):
    """There is no --reingest flag: a video without a transcript file is
    picked up again by the next run (update selects it the same way)."""
    with patch(_SLEEP), patch(_T1, side_effect=CaptionsBlocked("a", "IpBlocked")):
        run_ingest([_vid("a"), _vid("b")], tmp_path / 'tx', max_concurrent=1)
    with patch(_SLEEP), patch(_T1, return_value=(_SEGS, "en")) as t1:
        summary = run_ingest([_vid("a"), _vid("b")], tmp_path / 'tx', max_concurrent=1)
    assert _ids(t1) == ["a", "b"]
    assert summary["by_source"] == {"captions": 2}
    assert summary["skipped"] == 0


def test_unauthenticated_tier1_block_falls_through_to_cookied_tier2(tmp_path):
    """Tier 1 cannot use browser cookies. With --cookies-from-browser, its
    block is worked around by yt-dlp with the cookies (the documented fix);
    tier 1 is then skipped for the rest of the run."""
    with patch(_T1, side_effect=CaptionsBlocked("a", "IpBlocked", authenticated=False)) as t1, \
         patch(_T2, return_value=(_SEGS, "en")) as t2, patch(_T3) as t3:
        summary = run_ingest([_vid("a"), _vid("b"), _vid("c")], tmp_path / 'tx',
                             max_concurrent=1, cookies={"browser": "chrome"})
    assert _ids(t1) == ["a"]
    assert _ids(t2) == ["a", "b", "c"]
    t3.assert_not_called()
    assert summary["by_source"] == {"captions": 3}
    assert "blocked" not in summary


def test_cookied_tier2_block_after_tier1_block_stops_run(tmp_path):
    with patch(_T1, side_effect=CaptionsBlocked("a", "IpBlocked", authenticated=False)), \
         patch(_T2, side_effect=CaptionsBlocked("a", "not a bot", authenticated=True)) as t2, \
         patch(_T3) as t3:
        summary = run_ingest([_vid("a"), _vid("b")], tmp_path / 'tx',
                             max_concurrent=1, cookies={"browser": "chrome"})
    assert _ids(t2) == ["a"]
    t3.assert_not_called()
    assert summary["blocked"] is True and summary["not_attempted"] == 1


def test_authenticated_tier1_block_stops_run(tmp_path):
    """A cookie-file session was blocked too: yt-dlp would send the same
    cookies, so stop instead of falling through."""
    with patch(_T1, side_effect=CaptionsBlocked("a", "IpBlocked", authenticated=True)), \
         patch(_T2) as t2:
        summary = run_ingest([_vid("a"), _vid("b")], tmp_path / 'tx',
                             max_concurrent=1, cookies={"file": "/x/cookies.txt"})
    t2.assert_not_called()
    assert summary["blocked"] is True and summary["not_attempted"] == 1


def test_block_with_parallel_workers_sends_no_new_requests(tmp_path):
    """With several workers, a request already past the stop check may still
    go out, but once the block is recorded no worker starts another one."""
    import threading
    from youtube_advisor.ingest import orchestrator as orch
    started, at_block = [], []
    lock = threading.Lock()
    real_block = orch._RunState.block

    def recording_block(self, tier_name, e):
        real_block(self, tier_name, e)   # stop flag is set from here on
        with lock:
            at_block.append(len(started))

    def t1_side(vid):
        with lock:
            started.append(vid)
        if vid == "v5":
            raise CaptionsBlocked(vid, "IpBlocked")
        return (_SEGS, "en")
    videos = [_vid(f"v{i}") for i in range(30)]
    with patch(_SLEEP), patch(_T1, side_effect=t1_side), \
         patch.object(orch._RunState, "block", recording_block):
        summary = run_ingest(videos, tmp_path / "tx", max_concurrent=2)
    assert summary["blocked"] is True
    assert len(at_block) == 1
    # Only the other worker's in-flight request may start after the block.
    assert len(started) <= at_block[0] + 1
    attempted = summary["ingested"] + summary["failed"]
    assert attempted == len(started)
    assert attempted + summary["not_attempted"] == 30


def test_no_polite_sleep_for_videos_queued_after_block(tmp_path):
    """Without cookies each request is preceded by a polite sleep; after a
    block the queued videos must return at once, not sleep first."""
    with patch(_SLEEP) as sleep, \
         patch(_T1, side_effect=CaptionsBlocked("a", "IpBlocked")):
        summary = run_ingest([_vid(f"v{i}") for i in range(10)], tmp_path / "tx",
                             max_concurrent=1)
    assert sleep.call_count == 1
    assert summary["not_attempted"] == 9


# ----- captions-pass concurrency and polite sleep, per tier -----

def _peak_tracker(result=None, exc=None, hold=0.05):
    """Fake tier fn that records the peak number of concurrent calls."""
    import threading
    lock = threading.Lock()
    state = {"now": 0, "peak": 0, "calls": 0}

    def fn(vid, cookies=None):
        with lock:
            state["now"] += 1
            state["calls"] += 1
            state["peak"] = max(state["peak"], state["now"])
        threading.Event().wait(hold)  # time.sleep is patched in tests
        with lock:
            state["now"] -= 1
        if exc is not None:
            raise exc(vid)
        return result
    return fn, state


def _pool_sizes():
    import concurrent.futures as cf
    sizes = []

    class Recording(cf.ThreadPoolExecutor):
        def __init__(self, max_workers=None, *a, **kw):
            sizes.append(max_workers)
            super().__init__(max_workers, *a, **kw)
    return Recording, sizes


def _cookie_file(tmp_path):
    p = tmp_path / "cookies.txt"
    p.write_text("# Netscape HTTP Cookie File\n"
                 ".youtube.com\tTRUE\t/\tTRUE\t0\tSID\tx\n")
    return p


def _run_tiers(tmp_path, cookies, t1_fn, t2_fn, n=12):
    pool_cls, sizes = _pool_sizes()
    with patch(_SLEEP) as sleep, patch(_T1, side_effect=t1_fn), \
         patch(_T2, side_effect=t2_fn), \
         patch("youtube_advisor.ingest.orchestrator.cf.ThreadPoolExecutor", pool_cls):
        summary = run_ingest([_vid(f"v{i}") for i in range(n)], tmp_path / "tx",
                             cookies=cookies, skip_whisper=True)
    return summary, sizes, sleep


def test_browser_cookies_keep_tier1_at_unauthenticated_setting(tmp_path):
    """Browser cookies never reach the captions API: tier 1 runs at most 2
    calls at once, each after a polite sleep, inside the 8-worker pool."""
    t1, s1 = _peak_tracker(result=(_SEGS, "en"))
    summary, sizes, sleep = _run_tiers(tmp_path, {"browser": "chrome"}, t1, None)
    assert summary["ingested"] == 12
    assert sizes == [8]
    assert s1["peak"] == 2
    assert sleep.call_count == 12


def test_browser_cookies_keep_tier2_at_authenticated_setting(tmp_path):
    """yt-dlp sends the browser cookies: tier 2 is not held to tier 1's cap
    and gets no polite sleep."""
    t1, s1 = _peak_tracker(exc=NoCaptionsAvailable, hold=0)
    t2, s2 = _peak_tracker(result=(_SEGS, "en"), hold=0.1)
    summary, sizes, sleep = _run_tiers(tmp_path, {"browser": "chrome"}, t1, t2)
    assert summary["ingested"] == 12
    assert s1["peak"] <= 2
    assert 2 < s2["peak"] <= 8
    assert sleep.call_count == 12  # tier 1 only


def test_cookie_file_authenticates_tier1(tmp_path):
    t1, s1 = _peak_tracker(result=(_SEGS, "en"))
    summary, sizes, sleep = _run_tiers(tmp_path, {"file": _cookie_file(tmp_path)}, t1, None)
    assert summary["ingested"] == 12
    assert sizes == [8]
    assert 2 < s1["peak"] <= 8
    sleep.assert_not_called()


def test_unreadable_cookie_file_leaves_tier1_unauthenticated(tmp_path):
    t1, s1 = _peak_tracker(result=(_SEGS, "en"))
    _, sizes, sleep = _run_tiers(tmp_path, {"file": tmp_path / "missing.txt"}, t1, None)
    assert sizes == [8]
    assert s1["peak"] == 2
    assert sleep.call_count == 12


def test_no_cookies_two_workers_and_sleep_before_every_call(tmp_path):
    t1, s1 = _peak_tracker(exc=NoCaptionsAvailable, hold=0)
    t2, s2 = _peak_tracker(result=(_SEGS, "en"))
    summary, sizes, sleep = _run_tiers(tmp_path, None, t1, t2)
    assert summary["ingested"] == 12
    assert sizes == [2]
    assert s1["peak"] <= 2 and s2["peak"] <= 2
    assert sleep.call_count == 24  # before tier 1 and tier 2


def test_explicit_max_concurrent_lifts_tier1_cap(tmp_path):
    t1, s1 = _peak_tracker(result=(_SEGS, "en"))
    pool_cls, sizes = _pool_sizes()
    with patch(_SLEEP), patch(_T1, side_effect=t1), \
         patch("youtube_advisor.ingest.orchestrator.cf.ThreadPoolExecutor", pool_cls):
        run_ingest([_vid(f"v{i}") for i in range(12)], tmp_path / "tx",
                   cookies={"browser": "chrome"}, max_concurrent=6)
    assert sizes == [6]
    assert 2 < s1["peak"] <= 6
