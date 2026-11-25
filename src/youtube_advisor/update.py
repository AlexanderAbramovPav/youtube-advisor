from __future__ import annotations
import json
import logging
import pathlib
from datetime import date, datetime, timezone

import click
import yaml

from ._cookies import resolve_cookies
from .filters import FilterConfig, apply_filters
from .fetch_channel import resolve_channel, resolve_video
from .ingest.orchestrator import blocked_message, run_ingest
from .build_index import build_index
from .build_embeddings import build_embeddings
from .eval_runner import run as run_evals
from .gen_benchmark import generate as gen_bench
from .postrun_guide import update_guide
from ._md import video_id_from_filename, dump_yaml
from ._llm import should_skip_llm

log = logging.getLogger(__name__)

# Back-compat alias for tests that import the underscore name.
_video_id_from_filename = video_id_from_filename


def _load_state(advisor_dir: pathlib.Path) -> tuple[dict, dict]:
    refs = advisor_dir / "references"
    filt = yaml.safe_load((refs / "selection_filter.yaml").read_text())
    meta = json.loads((refs / "corpus_meta.json").read_text())
    return filt, meta


def _save_state(
    advisor_dir: pathlib.Path,
    filt: dict,
    meta: dict,
    *,
    save_filter: bool = True,
) -> None:
    refs = advisor_dir / "references"
    refs.mkdir(parents=True, exist_ok=True)
    if save_filter:
        (refs / "selection_filter.yaml").write_text(dump_yaml(filt))
    (refs / "corpus_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def _playlist_ids_from_urls(urls: list[str]) -> set[str]:
    """Extract `list=<ID>` from playlist URLs; non-URLs treated as bare IDs."""
    import re as _re
    ids: set[str] = set()
    for u in urls or []:
        m = _re.search(r"list=([\w-]+)", u)
        if m:
            ids.add(m.group(1))
        elif u:
            ids.add(u)
    return ids


def _filter_from_yaml(filt: dict) -> FilterConfig:
    return FilterConfig(
        since=date.fromisoformat(filt["since"]) if filt.get("since") else None,
        until=date.fromisoformat(filt["until"]) if filt.get("until") else None,
        max=filt.get("max", 500),
        title_include=filt.get("title_include"),
        title_exclude=filt.get("title_exclude"),
        min_duration=filt.get("min_duration"),
        max_duration=filt.get("max_duration"),
        playlist_ids=_playlist_ids_from_urls(filt.get("playlists", [])),
        manual_ids=set(filt.get("extra_ids", [])),
    )


def _language_was_detected(meta: dict) -> bool:
    """Whether the stored corpus_language came from real transcripts.

    Advisors written before the marker existed: their language is treated as
    a placeholder only when the stored corpus was empty (n_videos == 0), the
    one case where it provably was not detected.
    """
    if "corpus_language_detected" in meta:
        return bool(meta["corpus_language_detected"])
    return meta.get("n_videos", 0) > 0


def _requested_quote_style(meta: dict) -> str:
    """The quote style the user asked for: "auto" or a fixed style.

    Advisors written before it was stored cannot tell an explicit choice
    from "auto", so their stored style is kept as a fixed choice.
    """
    return meta.get("quote_style_requested") or meta.get("quote_style", "translation-first")


def _refresh_corpus_language(out: pathlib.Path, meta: dict) -> bool:
    """Detect corpus_language again when the stored one is a placeholder.

    A bootstrap that got no transcripts (e.g. YouTube blocked the first
    video) stores "en" without looking at any text. Once transcripts exist,
    detect the language from them and resolve an "auto" quote style again; a
    fixed quote style is kept. Updates ``meta`` in place; returns True when
    it detected the language (even when it is the "en" placeholder again).
    """
    from .bootstrap import _detect_corpus_language, _has_transcripts, _resolve_quote_style
    if _language_was_detected(meta) or not _has_transcripts(out / "transcripts"):
        return False
    requested = _requested_quote_style(meta)
    lang = _detect_corpus_language(out / "transcripts")
    meta["corpus_language"] = lang
    meta["corpus_language_detected"] = True
    meta["quote_style_requested"] = requested
    meta["quote_style"] = _resolve_quote_style(
        requested, meta.get("answer_language", "auto"), lang)
    return True


def _rewrite_stub(out: pathlib.Path, meta: dict) -> None:
    """Re-render the stub SKILL.md and the AI hand-off with the current meta
    and transcript samples (a blocked bootstrap wrote them with none).

    Only while ``.pending-llm-draft.json`` exists: then SKILL.md is still the
    stub (the AI deletes the hand-off once it drafts its own).
    """
    if not (out / ".pending-llm-draft.json").exists():
        return
    from .bootstrap import _write_pending_llm_draft, _write_stub_skill
    kw = dict(
        meta=meta,
        user_intent=meta.get("user_intent", ""),
        answer_language=meta.get("answer_language", "auto"),
        quote_style=meta.get("quote_style", "translation-first"),
        channel_display_name=", ".join(meta.get("channels", [])),
    )
    _write_stub_skill(out, **kw)
    _write_pending_llm_draft(out, **kw)


def _generate_pending_evals(out: pathlib.Path, filt: dict,
                            meta: dict) -> tuple[bool, dict | None]:
    """Generate the benchmark a blocked bootstrap skipped, then run it.

    Returns ``(attempted, eval_summary)``; the summary is None when nothing
    was pending or the attempt failed. The caller checks that this run was
    not blocked and may use the LLM. An existing ``evals/benchmark.json``
    (written by the calling AI, or by an earlier attempt whose eval run
    failed) is kept and only run. A failure is logged and leaves the evals
    pending: the corpus update itself has succeeded by then.
    """
    from .bootstrap import _has_transcripts
    if not meta.get("evals_pending") or not _has_transcripts(out / "transcripts"):
        return False, None
    try:
        if not (out / "evals" / "benchmark.json").exists():
            gen_bench(out)
        summary = run_evals(out)
    except Exception as e:
        log.warning("Could not generate or run the pending evals (%s); the "
                    "next update tries again.", e)
        return True, None
    meta.pop("evals_pending", None)
    _save_state(out, filt, meta, save_filter=False)
    return True, summary


def run(*, advisor_dir, since=None, max_n=None, ids=None,
        reindex_only=False, regen_skill_md=False, run_evals_=False,
        prune=False, yes=False, no_llm=None,
        cookies_from_browser: str | None = None,
        cookies_file: str | None = None,
        whisper_skip_if_no_captions: bool | None = None,
        no_evals: bool = False,
        **_) -> dict:
    """Ingest new videos for the saved filter and refresh the index.

    Videos that failed or were blocked on an earlier run have no transcript
    file, so this run selects and fetches them again. ``None`` for
    ``whisper_skip_if_no_captions`` keeps the value saved at bootstrap.
    ``no_evals`` drops evals a blocked bootstrap left pending.
    """

    out = pathlib.Path(advisor_dir).expanduser()
    cookies = resolve_cookies(
        from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )
    filt, meta = _load_state(out)
    # Decide before this run changes n_videos (the legacy rule reads it).
    meta["corpus_language_detected"] = _language_was_detected(meta)

    # CLI overrides — normalise dates at the boundary so downstream
    # `date.fromisoformat()` never sees a datetime-shaped string.
    filt_changed = False
    if since is not None:
        d = since.date() if hasattr(since, "date") else since
        filt["since"] = d.isoformat() if hasattr(d, "isoformat") else d
        filt_changed = True
    if max_n is not None:
        filt["max"] = max_n
        filt_changed = True
    if ids is not None:
        ids_path = pathlib.Path(ids)
        extra = {line.strip() for line in ids_path.read_text().splitlines() if line.strip()}
        filt["extra_ids"] = sorted(set(filt.get("extra_ids", [])) | extra)
        filt_changed = True
    if no_evals:
        meta.pop("evals_pending", None)
    if whisper_skip_if_no_captions is not None:
        filt["whisper_skip_if_no_captions"] = bool(whisper_skip_if_no_captions)
        filt_changed = True

    # Resolve LLM mode up-front so the early-return path can gate evals correctly.
    effective_no_llm = should_skip_llm(no_llm)

    # Keep vendored/symlinked runtime fresh on every update (idempotent) —
    # without this, `pip install -U youtube-advisor` doesn't propagate to
    # existing advisors that vendored an old copy.
    from .bootstrap import _install_runtime
    _install_runtime(out)

    # Reindex-only short-circuit
    if reindex_only:
        build_index(out / "transcripts", out / "references", incremental=False)
        build_embeddings(out / "transcripts", out / "references",
                         model_name=meta["embedding_model"], incremental=False)
        _save_state(out, filt, meta, save_filter=filt_changed)
        return {"guide": update_guide(out, 0, 0), "evals": None}

    # Re-resolve channels. Same capped-listing heuristic as bootstrap: when
    # only `max` narrows selection (no date/title/duration filters), the newest-first
    # /videos tab top max*3 is guaranteed to contain the max newest videos —
    # avoids the per-video date-resolution pass over the entire channel.
    # Manual extra_ids are resolved individually below, so the cap never
    # hides them.
    listing_limit = None
    if (
        filt.get("max")
        and not filt.get("since")
        and not filt.get("until")
        and not filt.get("title_include")
        and filt.get("min_duration") is None
        and filt.get("max_duration") is None
    ):
        listing_limit = filt["max"] * 3
    candidates = []
    for ch in filt["channels"]:
        candidates.extend(resolve_channel(ch, limit=listing_limit, cookies=cookies))
    for pl in filt.get("playlists", []):
        candidates.extend(resolve_channel(pl, limit=listing_limit, cookies=cookies))

    # Manual ids outside the (possibly capped) listing get their metadata
    # fetched individually — they must be present in candidates to survive
    # apply_filters().
    have = {v["video_id"] for v in candidates}
    for vid in filt.get("extra_ids", []):
        if vid not in have:
            v = resolve_video(vid, cookies=cookies)
            if v:
                candidates.append(v)

    cfg = _filter_from_yaml(filt)
    selected = apply_filters(candidates, cfg)

    existing_ids = {video_id_from_filename(p) for p in (out / "transcripts").glob("*.md")}
    new_videos = [v for v in selected if v["video_id"] not in existing_ids]
    on_channel = {v["video_id"] for v in selected}
    # A capped listing cannot prove a video left the channel UNLESS the listing
    # came back smaller than the cap — in that case we've seen the whole channel.
    # Otherwise (listing hit the cap), refuse to compute "missing".
    full_listing = listing_limit is None or len(candidates) < listing_limit
    missing = existing_ids - on_channel if full_listing else set()

    if not new_videos and not missing:
        # Persist any CLI overrides even when corpus didn't change.
        if _refresh_corpus_language(out, meta):
            _rewrite_stub(out, meta)
        _save_state(out, filt, meta, save_filter=filt_changed)
        attempted, eval_summary = (_generate_pending_evals(out, filt, meta)
                                   if not effective_no_llm else (False, None))
        if not attempted and run_evals_ and not effective_no_llm:
            eval_summary = run_evals(out)
        return {
            "guide": f"✓ {out.name}: 0 new, 0 removed, index unchanged.\n",
            "evals": eval_summary,
        }

    if not yes:
        click.confirm(
            f"Found {len(new_videos)} new videos, {len(missing)} no longer on channel. Continue?",
            abort=True,
        )

    # Prune missing
    if missing and prune:
        for vid in missing:
            for p in (out / "transcripts").glob(f"*-{vid}.md"):
                p.unlink()

    # Ingest new
    summary = (run_ingest(new_videos, out / "transcripts", cookies=cookies,
                          skip_whisper=bool(filt.get("whisper_skip_if_no_captions")))
               if new_videos
               else {"by_source": {}, "ingested": 0, "skipped": 0, "failed": 0})

    # Incremental index + embeddings
    build_index(out / "transcripts", out / "references", incremental=True)
    build_embeddings(out / "transcripts", out / "references",
                     model_name=meta["embedding_model"], incremental=True)

    # Update meta
    n_videos = len(list((out / "transcripts").glob("*.md")))
    meta["n_videos"] = n_videos
    meta["corpus_version"] = f"{date.today().isoformat()}/{n_videos}"
    meta["last_update"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for src, c in summary.get("by_source", {}).items():
        if src.startswith("failed"):
            continue  # not in the corpus; retried runs would count them again
        meta.setdefault("transcript_source_counts", {})
        meta["transcript_source_counts"][src] = meta["transcript_source_counts"].get(src, 0) + c

    # Refresh latest_video_date from the just-rebuilt index.
    index_path = out / "references" / "index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text())
        dates = [v.get("published_date") for v in idx.get("videos", []) if v.get("published_date")]
        if dates:
            meta["latest_video_date"] = max(dates)

    if _refresh_corpus_language(out, meta):
        _rewrite_stub(out, meta)
    _save_state(out, filt, meta, save_filter=filt_changed)

    blocked = bool(summary.get("blocked"))
    if blocked and (regen_skill_md or run_evals_):
        log.warning("YouTube blocked this run: skipping SKILL.md regen and evals "
                    "on the partial corpus; re-run update when unblocked.")
    user_intent = meta.get("user_intent") or "(regenerated)"

    # SKILL.md regen — opt-in only
    if regen_skill_md and not blocked:
        if effective_no_llm:
            # Re-emit the hand-off file so the calling AI can re-draft SKILL.md
            # from the (now-updated) corpus samples. Leave the existing SKILL.md
            # in place until the AI overwrites it.
            from .bootstrap import _write_pending_llm_draft
            _write_pending_llm_draft(
                out,
                meta=meta,
                user_intent=user_intent,
                answer_language=meta.get("answer_language", "auto"),
                quote_style=meta.get("quote_style", "translation-first"),
                channel_display_name=", ".join(meta.get("channels", [])),
            )
        else:
            from .gen_skill_md import generate as gen_skill
            gen_skill(out, user_intent=user_intent,
                      answer_language=meta.get("answer_language", "auto"),
                      quote_style=meta.get("quote_style", "translation-first"),
                      channel_display_name=", ".join(meta.get("channels", [])))
            # The LLM draft replaces any AI hand-off left by --no-llm or a
            # blocked bootstrap; a stale one would invite a second redraft.
            (out / ".pending-llm-draft.json").unlink(missing_ok=True)

    # Evals a blocked bootstrap skipped: generated by the first update that
    # finishes without a block (the LLM writes the benchmark).
    attempted, eval_summary = False, None
    if not effective_no_llm and not blocked:
        attempted, eval_summary = _generate_pending_evals(out, filt, meta)

    # Evals (opt-in); already run above when evals were pending.
    if not attempted and run_evals_ and not effective_no_llm and not blocked:
        eval_summary = run_evals(out)
        delta = eval_summary.get("delta_passed")
        if delta is not None and delta < 0 and not yes:
            if not click.confirm(
                f"Eval pass rate dropped by {abs(delta)}. Continue without rollback?",
                default=True,
            ):
                # Rollback: delete just-ingested transcripts and rebuild.
                for v in new_videos:
                    for p in (out / "transcripts").glob(f"*-{v['video_id']}.md"):
                        p.unlink()
                build_index(out / "transcripts", out / "references", incremental=False)
                # Incremental: drops the rolled-back rows, re-embeds nothing.
                build_embeddings(out / "transcripts", out / "references",
                                 model_name=meta["embedding_model"], incremental=True)
                return {
                    "guide": f"⟲ Rolled back {len(new_videos)} new videos due to eval regression.\n",
                    "evals": eval_summary,
                    "summary": summary,
                }

    # Count what actually landed: failed or blocked videos are not in the corpus.
    guide = update_guide(out, summary.get("ingested", 0),
                         len(missing) if prune else 0, eval_summary)
    if summary.get("blocked"):
        # Carry the flags this run skipped into the printed retry command.
        skipped = [f for f, on in (("--regen-skill-md", regen_skill_md),
                                   ("--run-evals", run_evals_)) if on]
        guide += "\n" + " ".join([blocked_message(summary, out), *skipped]) + "\n"
    return {
        "guide": guide,
        "evals": eval_summary,
        "summary": summary,
    }
