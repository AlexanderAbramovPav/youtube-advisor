from __future__ import annotations
import pathlib, logging, random, time, concurrent.futures as cf
from . import captions_api as tier1
from . import captions_ytdlp as tier2
from . import whisper as tier3
from .captions_api import NoCaptionsAvailable
from .normalize import write_transcript

log = logging.getLogger(__name__)

def _existing(out_dir: pathlib.Path, video_id: str) -> bool:
    return any(out_dir.glob(f"*-{video_id}.md"))

def _ingest_one(
    meta: dict,
    out_dir: pathlib.Path,
    skip_whisper: bool,
    cookies: dict | None = None,
    polite_sleep: bool = False,
) -> str:
    vid = meta["video_id"]
    for tier_name, src, fn in [("tier1", "captions", tier1.fetch_captions),
                                ("tier2", "captions", tier2.fetch_captions)]:
        try:
            if polite_sleep:
                # Stagger captions API calls to avoid IP-level rate-limit when
                # the caller is unauthenticated.
                time.sleep(random.uniform(0.5, 2.0))
            segs, lang = fn(vid, cookies=cookies) if cookies is not None else fn(vid)
            write_transcript(out_dir, meta, segs, src, lang)
            return src
        except NoCaptionsAvailable:
            continue
        except Exception as e:
            # Non-NoCaptionsAvailable exceptions (network errors, parse
            # failures, etc.) should fail THIS video instead of escalating to
            # the expensive Whisper tier — a transient ConnectionError isn't
            # evidence that the video has no captions. The video can be
            # retried later via `update --reingest <id>`.
            log.warning("%s failed for %s: %s", tier_name, vid, e)
            return f"failed:{type(e).__name__}"
    if skip_whisper:
        return "failed:no-captions"
    try:
        # Whisper transcribe accepts cookies for the audio-download yt-dlp call.
        if cookies is not None:
            segs, lang = tier3.transcribe_audio(vid, cookies=cookies)
        else:
            segs, lang = tier3.transcribe_audio(vid)
        write_transcript(out_dir, meta, segs, "whisper-v3", lang)
        return "whisper-v3"
    except Exception as e:
        log.error("tier3 failed for %s: %s", vid, e)
        return f"failed:{type(e).__name__}"


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
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_log = out_dir.parent / "ingest.errors.log"
    summary = {"ingested": 0, "skipped": 0, "failed": 0, "by_source": {}}

    authenticated = bool(cookies)
    if max_concurrent is None:
        max_concurrent = 8 if authenticated else 2
    polite_sleep = not authenticated

    pending = [v for v in videos if not _existing(out_dir, v["video_id"])]
    summary["skipped"] = len(videos) - len(pending)

    # Pass 1: Tier 1 + Tier 2 in parallel pool (skip_whisper=True keeps the pool network-only).
    def network_only(v):
        return _ingest_one(v, out_dir, skip_whisper=True,
                           cookies=cookies, polite_sleep=polite_sleep)
    if pending:
        with cf.ThreadPoolExecutor(max_workers=min(max_concurrent, max(1, len(pending)))) as ex:
            results = list(ex.map(network_only, pending))
    else:
        results = []

    need_whisper, network_ok = [], []
    for v, r in zip(pending, results):
        # Only "true no-captions" videos escalate to Whisper. Other failures
        # (transient network errors, parse errors) get reported as-is so a
        # later `--reingest` retry can resolve them cheaply.
        if r == "failed:no-captions":
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
            r = _ingest_one(v, out_dir, skip_whisper=False,
                            cookies=cookies, polite_sleep=False)
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
    return summary
