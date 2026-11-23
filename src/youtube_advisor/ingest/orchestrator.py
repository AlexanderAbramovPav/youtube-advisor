from __future__ import annotations
import concurrent.futures as cf
import logging
import pathlib
import random
import threading
import time
from . import captions_api as tier1
from . import captions_ytdlp as tier2
from . import whisper as tier3
from .captions_api import CaptionsBlocked, NoCaptionsAvailable
from .normalize import write_transcript

log = logging.getLogger(__name__)

BLOCKED = "failed:blocked"
# Internal sentinel: the run stopped on a block before this video was tried.
# Such a video is neither counted nor logged as failed; it stays pending (no
# transcript file), so the next `update` run picks it up.
NOT_ATTEMPTED = "not-attempted"


class _RunState:
    """Block bookkeeping shared by the worker threads of one run_ingest call."""

    def __init__(self) -> None:
        self.stop = threading.Event()        # no more YouTube requests this run
        self.skip_tier1 = threading.Event()  # tier 1 is blocked; cookied tier 2 is not
        self.block_detail: str | None = None
        self._lock = threading.Lock()

    def block(self, tier_name: str, e: CaptionsBlocked) -> None:
        with self._lock:
            first = not self.stop.is_set()
            self.stop.set()
            if first:
                self.block_detail = f"{tier_name}: {e}"
        if first:
            log.error(
                "YouTube blocked requests (%s: %s). Stopping: no further "
                "YouTube requests in this run. Videos not fetched yet stay "
                "pending; re-run `update` later or pass --cookies-from-browser.",
                tier_name, e,
            )

    def tier1_blocked(self, e: CaptionsBlocked) -> None:
        with self._lock:
            first = not self.skip_tier1.is_set()
            self.skip_tier1.set()
        if first:
            log.warning(
                "tier1 blocked without cookies (%s); using cookied yt-dlp only "
                "for the rest of this run.", e,
            )


def _existing(out_dir: pathlib.Path, video_id: str) -> bool:
    return any(out_dir.glob(f"*-{video_id}.md"))

def _ingest_one(
    meta: dict,
    out_dir: pathlib.Path,
    skip_whisper: bool,
    cookies: dict | None = None,
    polite_sleep: bool = False,
    state: _RunState | None = None,
) -> str:
    vid = meta["video_id"]
    state = state or _RunState()
    for tier_name, src, fn in [("tier1", "captions", tier1.fetch_captions),
                                ("tier2", "captions", tier2.fetch_captions)]:
        if tier_name == "tier1" and state.skip_tier1.is_set():
            continue
        if state.stop.is_set():
            return NOT_ATTEMPTED
        try:
            if polite_sleep:
                # Stagger captions API calls to avoid IP-level rate-limit when
                # the caller is unauthenticated.
                time.sleep(random.uniform(0.5, 2.0))
            if state.stop.is_set():
                return NOT_ATTEMPTED
            segs, lang = fn(vid, cookies=cookies) if cookies is not None else fn(vid)
            write_transcript(out_dir, meta, segs, src, lang)
            return src
        except CaptionsBlocked as e:
            # A block says nothing about captions: never escalate to Whisper.
            # One exception: tier 1 cannot use browser cookies, so when it is
            # blocked unauthenticated and the user gave cookies, tier 2
            # (yt-dlp with those cookies) is the documented workaround.
            if tier_name == "tier1" and not e.authenticated and cookies:
                state.tier1_blocked(e)
                continue
            state.block(tier_name, e)
            return BLOCKED
        except NoCaptionsAvailable:
            continue
        except Exception as e:
            # Non-NoCaptionsAvailable exceptions (network errors, parse
            # failures, etc.) should fail THIS video instead of escalating to
            # the expensive Whisper tier — a transient ConnectionError isn't
            # evidence that the video has no captions. The video has no
            # transcript file, so the next `update` run retries it.
            log.warning("%s failed for %s: %s", tier_name, vid, e)
            return f"failed:{type(e).__name__}"
    if skip_whisper:
        return "failed:no-captions"
    if state.stop.is_set():
        return NOT_ATTEMPTED
    try:
        # Whisper transcribe accepts cookies for the audio-download yt-dlp call.
        if cookies is not None:
            segs, lang = tier3.transcribe_audio(vid, cookies=cookies)
        else:
            segs, lang = tier3.transcribe_audio(vid)
        write_transcript(out_dir, meta, segs, "whisper-v3", lang)
        return "whisper-v3"
    except CaptionsBlocked as e:
        state.block("tier3", e)
        return BLOCKED
    except Exception as e:
        log.error("tier3 failed for %s: %s", vid, e)
        return f"failed:{type(e).__name__}"


def blocked_message(summary: dict, advisor_dir: pathlib.Path | str) -> str:
    """User-facing text for a run that stopped on a YouTube block."""
    n_blocked = summary.get("by_source", {}).get(BLOCKED, 0)
    n_pending = summary.get("not_attempted", 0)
    return (
        f"YouTube blocked requests ({summary.get('block_detail') or 'HTTP 429 / bot check'}).\n"
        f"Ingest stopped: {n_blocked} video(s) blocked, {n_pending} not tried yet; "
        "no Whisper fallback was run for them.\n"
        "They are retried by the next update. Wait (up to an hour), or pass "
        "--cookies-from-browser <browser>, then run:\n"
        f"  youtube-advisor update --advisor {advisor_dir} --yes"
    )


def run_ingest(
    videos: list[dict],
    out_dir: pathlib.Path,
    skip_whisper: bool = False,
    max_concurrent: int | None = None,
    cookies: dict | None = None,
    progress=None,
) -> dict:
    """Ingest videos through the Tier 1 → 2 → 3 cascade.

    When ``cookies`` is unset/empty we lower captions-pass concurrency to 2
    AND add a 0.5–2.0s random sleep per call to reduce IP-level rate-limit
    risk. When cookies are present we default to 8-way concurrency. Pass an
    explicit ``max_concurrent`` to override either default.

    Optional ``progress``: a ``_progress.Progress``-shaped object; we call
    ``progress.tick(video_id, title, source)`` after each video lands so a
    sibling reader (status.sh) can show live counters. Default ``None`` is a
    no-op — unit tests don't need to pass it.

    A YouTube block (CaptionsBlocked from any tier) stops the run: in-flight
    workers finish, no new YouTube request is sent, and Whisper is skipped.
    The blocked video is recorded as ``failed:blocked``; untried videos are
    left pending. The summary then carries ``blocked=True``,
    ``not_attempted`` and ``block_detail``; these keys are absent otherwise.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_log = out_dir.parent / "ingest.errors.log"
    summary = {"ingested": 0, "skipped": 0, "failed": 0, "by_source": {}}

    authenticated = bool(cookies)
    if max_concurrent is None:
        max_concurrent = 8 if authenticated else 2
    polite_sleep = not authenticated
    state = _RunState()

    pending = [v for v in videos if not _existing(out_dir, v["video_id"])]
    summary["skipped"] = len(videos) - len(pending)

    # Pass 1: Tier 1 + Tier 2 in parallel pool (skip_whisper=True keeps the pool network-only).
    def network_only(v):
        return _ingest_one(v, out_dir, skip_whisper=True,
                           cookies=cookies, polite_sleep=polite_sleep, state=state)
    if pending:
        with cf.ThreadPoolExecutor(max_workers=min(max_concurrent, max(1, len(pending)))) as ex:
            results = list(ex.map(network_only, pending))
    else:
        results = []

    need_whisper, network_ok, not_attempted = [], [], []
    for v, r in zip(pending, results):
        # Only "true no-captions" videos escalate to Whisper. Other failures
        # (transient network errors, parse errors, blocks) get reported as-is
        # so the next `update` run can retry them cheaply.
        if r == NOT_ATTEMPTED:
            not_attempted.append(v)
        elif r == "failed:no-captions":
            need_whisper.append(v)
        else:
            network_ok.append((v, r))
            if progress is not None:
                try:
                    progress.tick(video_id=v["video_id"], title=v.get("title"), source=r)
                except Exception:
                    log.warning("progress.tick failed (non-fatal)", exc_info=True)

    # Pass 2: Tier 3 serial (Whisper saturates CPU/MLX itself).
    if not skip_whisper and need_whisper:
        for v in need_whisper:
            # After a block, Whisper would download audio from YouTube too.
            r = (NOT_ATTEMPTED if state.stop.is_set() else
                 _ingest_one(v, out_dir, skip_whisper=False,
                             cookies=cookies, polite_sleep=False, state=state))
            if r == NOT_ATTEMPTED:
                not_attempted.append(v)
                continue
            network_ok.append((v, r))
            if progress is not None:
                try:
                    progress.tick(video_id=v["video_id"], title=v.get("title"), source=r)
                except Exception:
                    log.warning("progress.tick failed (non-fatal)", exc_info=True)
    elif skip_whisper:
        # When skip_whisper, no-captions stays as "failed:no-captions" — keep that classification.
        for v in need_whisper:
            network_ok.append((v, "failed:no-captions"))
            if progress is not None:
                try:
                    progress.tick(video_id=v["video_id"], title=v.get("title"),
                                  source="failed:no-captions")
                except Exception:
                    log.warning("progress.tick failed (non-fatal)", exc_info=True)

    for v, r in network_ok:
        if r.startswith("failed"):
            summary["failed"] += 1
            with errors_log.open("a") as f:
                f.write(f"{v['video_id']}\t{r}\n")
        else:
            summary["ingested"] += 1
        summary["by_source"][r] = summary["by_source"].get(r, 0) + 1
    if state.stop.is_set():
        summary["blocked"] = True
        summary["not_attempted"] = len(not_attempted)
        summary["block_detail"] = state.block_detail
    return summary
