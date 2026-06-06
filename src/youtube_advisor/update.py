from __future__ import annotations
import json
import pathlib
from datetime import date, datetime, timezone

import click
import yaml

from ._cookies import resolve_cookies
from .filters import FilterConfig, apply_filters
from .fetch_channel import resolve_channel, resolve_video
from .ingest.orchestrator import run_ingest
from .build_index import build_index
from .build_embeddings import build_embeddings
from .eval_runner import run as run_evals
from .postrun_guide import update_guide
from ._md import video_id_from_filename, dump_yaml
from ._llm import should_skip_llm

# Back-compat alias for tests that import the underscore name.
_video_id_from_filename = video_id_from_filename


def _load_state(advisor_dir: pathlib.Path) -> tuple[dict, dict]:
    refs = advisor_dir / "references"
    filt = yaml.safe_load((refs / "selection_filter.yaml").read_text())
    meta = json.loads((refs / "corpus_meta.json").read_text())
    return filt, meta


def _save_state(advisor_dir: pathlib.Path, filt: dict, meta: dict) -> None:
    refs = advisor_dir / "references"
    refs.mkdir(parents=True, exist_ok=True)
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
        playlist_ids=_playlist_ids_from_urls(filt.get("playlists", [])),
        manual_ids=set(filt.get("extra_ids", [])),
    )


def run(*, advisor_dir, since=None, max_n=None, ids=None,
        reindex_only=False, regen_skill_md=False, run_evals_=False,
        prune=False, yes=False, no_llm=None,
        cookies_from_browser: str | None = None,
        cookies_file: str | None = None,
        **_) -> dict:

    out = pathlib.Path(advisor_dir).expanduser()
    cookies = resolve_cookies(
        from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )
    filt, meta = _load_state(out)

    # CLI overrides — normalise dates at the boundary so downstream
    # `date.fromisoformat()` never sees a datetime-shaped string.
    if since is not None:
        d = since.date() if hasattr(since, "date") else since
        filt["since"] = d.isoformat() if hasattr(d, "isoformat") else d
    if max_n is not None:
        filt["max"] = max_n
    if ids is not None:
        ids_path = pathlib.Path(ids)
        extra = {l.strip() for l in ids_path.read_text().splitlines() if l.strip()}
        filt["extra_ids"] = sorted(set(filt.get("extra_ids", [])) | extra)

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
        _save_state(out, filt, meta)
        return {"guide": update_guide(out, 0, 0), "evals": None}

    # Re-resolve channels. Same capped-listing heuristic as bootstrap: when
    # only `max` narrows selection (no date/title filters), the newest-first
    # /videos tab top max*3 is guaranteed to contain the max newest videos —
    # avoids the per-video date-resolution pass over the entire channel.
    # Manual extra_ids are resolved individually below, so the cap never
    # hides them.
    listing_limit = None
    if filt.get("max") and not filt.get("since") and not filt.get("until") \
            and not filt.get("title_include"):
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
        _save_state(out, filt, meta)
        eval_summary = run_evals(out) if (run_evals_ and not effective_no_llm) else None
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
    summary = (run_ingest(new_videos, out / "transcripts", cookies=cookies)
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
        meta.setdefault("transcript_source_counts", {})
        meta["transcript_source_counts"][src] = meta["transcript_source_counts"].get(src, 0) + c

    # Refresh latest_video_date from the just-rebuilt index.
    index_path = out / "references" / "index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text())
        dates = [v.get("published_date") for v in idx.get("videos", []) if v.get("published_date")]
        if dates:
            meta["latest_video_date"] = max(dates)

    _save_state(out, filt, meta)

    # SKILL.md regen — opt-in only
    if regen_skill_md:
        if effective_no_llm:
            # Re-emit the hand-off file so the calling AI can re-draft SKILL.md
            # from the (now-updated) corpus samples. Leave the existing SKILL.md
            # in place until the AI overwrites it.
            from .bootstrap import _write_pending_llm_draft
            _write_pending_llm_draft(
                out,
                meta=meta,
                user_intent="(regenerated)",
                answer_language=meta.get("answer_language", "auto"),
                quote_style=meta.get("quote_style", "translation-first"),
                channel_display_name=", ".join(meta.get("channels", [])),
            )
        else:
            from .gen_skill_md import generate as gen_skill
            gen_skill(out, user_intent="(regenerated)",
                      answer_language=meta.get("answer_language", "auto"),
                      quote_style=meta.get("quote_style", "translation-first"),
                      channel_display_name=", ".join(meta.get("channels", [])))

    # Evals (opt-in)
    eval_summary = None
    if run_evals_ and not effective_no_llm:
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
                build_embeddings(out / "transcripts", out / "references",
                                 model_name=meta["embedding_model"], incremental=False)
                return {
                    "guide": f"⟲ Rolled back {len(new_videos)} new videos due to eval regression.\n",
                    "evals": eval_summary,
                }

    return {
        "guide": update_guide(out, len(new_videos), len(missing) if prune else 0, eval_summary),
        "evals": eval_summary,
    }
