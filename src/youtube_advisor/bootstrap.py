from __future__ import annotations
import json
import os
import pathlib
import shutil
from datetime import date, datetime, timezone

import click
from jinja2 import Environment, FileSystemLoader

from ._cookies import resolve_cookies
from ._progress import Progress
from .filters import FilterConfig, apply_filters
from .fetch_channel import resolve_channel
from .ingest.orchestrator import run_ingest
from .build_index import build_index
from .build_embeddings import build_embeddings
from .gen_skill_md import generate as gen_skill
from ._paths import TEMPLATES_DIR
from .gen_benchmark import generate as gen_bench
from .eval_runner import run as run_evals
from .postrun_guide import bootstrap_guide
from ._md import parse_frontmatter, dump_yaml, sample_transcripts
from ._naming import short_slug


def _resolve_quote_style(qs: str, answer_language: str, corpus_lang: str) -> str:
    """Resolve auto quote-style based on answer/corpus languages.

    Explicit user choice wins. When auto:
      - answer_language == "auto" → translation-first (safe default).
      - answer_language matches corpus_lang → quote-only.
      - otherwise → translation-first.
    """
    if qs != "auto":
        return qs
    if answer_language == "auto":
        return "translation-first"
    return "quote-only" if answer_language == corpus_lang else "translation-first"


def _detect_corpus_language(transcripts_dir: pathlib.Path) -> str:
    """Sample up to 3 transcripts' frontmatter `transcript_language`; default 'en'."""
    if not transcripts_dir.exists():
        return "en"
    for md in sorted(transcripts_dir.glob("*.md"))[:3]:
        fm, _ = parse_frontmatter(md.read_text())
        lang = fm.get("transcript_language")
        if lang:
            return str(lang)[:2]
    return "en"


def _vendor_or_symlink(out: pathlib.Path, vendor: bool) -> None:
    """Place `youtube_advisor` importable from <out>/scripts/_lib.

    vendor=True: copy the package into <out>/scripts/_lib/youtube_advisor.
    vendor=False: symlink <out>/scripts/_lib → parent dir of the installed
                  youtube_advisor package (so `from youtube_advisor...` works).
    """
    lib_dir = out / "scripts" / "_lib"
    pkg_root = pathlib.Path(__file__).resolve().parent  # .../youtube_advisor/
    pkg_parent = pkg_root.parent                         # parent containing youtube_advisor/

    if lib_dir.is_symlink() or lib_dir.exists():
        if lib_dir.is_symlink():
            lib_dir.unlink()
        else:
            shutil.rmtree(lib_dir)

    if vendor:
        lib_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(pkg_root, lib_dir / "youtube_advisor", dirs_exist_ok=True)
    else:
        lib_dir.parent.mkdir(parents=True, exist_ok=True)
        lib_dir.symlink_to(pkg_parent)


_BUILD_INDEX_WRAPPER = '''#!/usr/bin/env python3
"""Rebuild the advisor's index.json + embeddings.npz (no fetch).

Usage: python3 scripts/build_index.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "_lib"))

from youtube_advisor.build_index import build_index
from youtube_advisor.build_embeddings import build_embeddings

root = pathlib.Path(__file__).parent.parent
meta = json.loads((root / "references" / "corpus_meta.json").read_text())
build_index(root / "transcripts", root / "references")
build_embeddings(
    root / "transcripts",
    root / "references",
    model_name=meta["embedding_model"],
)
print(f"Rebuilt index + embeddings for {meta['n_videos']} videos.")
'''


def _install_runtime(out: pathlib.Path, vendor: bool = False) -> None:
    """Refresh the vendored/symlinked ``_lib`` and the scripts wrappers.

    Idempotent — safe to call from both bootstrap and update. After
    ``pip install -U youtube-advisor`` an existing advisor will still call
    the old code via its vendored copy unless this is re-run.
    """
    _vendor_or_symlink(out, vendor)
    _install_search_wrapper(out)


def _install_search_wrapper(out: pathlib.Path) -> None:
    """Install both scripts/search.py and scripts/build_index.py wrappers."""
    tmpl_src = TEMPLATES_DIR / "search.py.tmpl"
    scripts_dir = out / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    search_dst = scripts_dir / "search.py"
    search_dst.write_text(tmpl_src.read_text())
    search_dst.chmod(0o755)

    build_dst = scripts_dir / "build_index.py"
    build_dst.write_text(_BUILD_INDEX_WRAPPER)
    build_dst.chmod(0o755)


_STUB_ANTI_PATTERNS = [
    "Don't fabricate quotes; only quote what you've actually read in the transcripts.",
    "Don't answer outside the corpus without explicitly saying so and offering a fallback.",
    "Don't paraphrase when a verbatim quote is available — quote beats paraphrase.",
]


def _write_stub_skill(
    advisor_dir: pathlib.Path,
    *,
    meta: dict,
    user_intent: str,
    answer_language: str,
    quote_style: str,
    channel_display_name: str,
) -> None:
    """Render a stub SKILL.md / AGENTS.md / README.md / workflow.md / answer_template
    using generic-but-honest defaults. The calling AI is expected to polish SKILL.md
    after reading the .pending-llm-draft.json hand-off."""
    slug = short_slug(advisor_dir.name)
    description = (
        f"Advisor over {channel_display_name} ({meta['n_videos']} videos). "
        f"Answer questions about this channel's content with verbatim quotes from the corpus."
    )
    if user_intent:
        description += f" Intent: {user_intent}"
    purpose = (
        f"Answer questions about {channel_display_name} content with verbatim quotes "
        f"linked to source video timestamps. Built from {meta['n_videos']} transcripts "
        f"({meta.get('date_range', '')})."
    )
    example_queries = [
        f"What does {channel_display_name} say about X?",
        f"How does {channel_display_name} approach Y?",
        f"Find quotes from {channel_display_name} about Z.",
        f"Has {channel_display_name} discussed W?",
        f"What advice has {channel_display_name} given on V?",
    ]

    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), keep_trailing_newline=True)
    ctx = {
        "channel_slug": slug,
        "advisor_title": f"{channel_display_name} advisor",
        "channels_list": ", ".join(meta.get("channels", [])),
        "n_videos": meta["n_videos"],
        "date_range": meta.get("date_range", ""),
        "corpus_version": meta["corpus_version"],
        "embedding_model": meta.get("embedding_model", ""),
        "embedding_dim": meta.get("embedding_dim", 0),
        "quote_style": quote_style,
        "answer_language": answer_language,
        "answer_language_rule": (
            "Answer in Russian; quote verbatim in the corpus language"
            if answer_language == "ru"
            else "Answer in the user's chat language; quote verbatim in the corpus language"
        ),
        "channel_display_name": channel_display_name,
        "corpus_language": meta.get("corpus_language", "en"),
        "generated_description": description,
        "generated_purpose": purpose,
        "generated_anti_patterns": "\n".join(f"- {a}" for a in _STUB_ANTI_PATTERNS),
        "generated_example_queries": "\n".join(f"- {q}" for q in example_queries),
        "generator_version": "0.1.0",
        "repo_url": "https://github.com/AlexanderAbramovPav/youtube-advisor",
        "created_at": meta.get("last_bootstrap", ""),
    }

    (advisor_dir / "SKILL.md").write_text(env.get_template("advisor.SKILL.md.tmpl").render(**ctx))
    (advisor_dir / "AGENTS.md").write_text(env.get_template("advisor.AGENTS.md.tmpl").render(**ctx))
    (advisor_dir / "README.md").write_text(env.get_template("advisor.README.md.tmpl").render(**ctx))
    (advisor_dir / "references" / "workflow.md").write_text(
        env.get_template("workflow.md.tmpl").render(**ctx))
    (advisor_dir / "assets").mkdir(exist_ok=True)
    (advisor_dir / "assets" / "answer_template.md").write_text(
        env.get_template("answer_template.md.tmpl").render(**ctx))


def _write_pending_llm_draft(
    advisor_dir: pathlib.Path,
    *,
    meta: dict,
    user_intent: str,
    answer_language: str,
    quote_style: str,
    channel_display_name: str,
) -> None:
    """Write .pending-llm-draft.json — the hand-off file that the calling AI reads
    to draft a polished SKILL.md (and optionally evals/benchmark.json)."""
    samples = sample_transcripts(advisor_dir, 8, body_chars=3500, with_video_id=True)
    # Convert tuples to dicts for JSON-serialisability.
    samples_json = [{"video_id": vid, "excerpt": body} for (vid, body) in samples]

    payload = {
        "task": (
            "Polish this advisor's SKILL.md. Read the samples below, then render "
            "SKILL.md, AGENTS.md, and (optionally) evals/benchmark.json using the "
            "schema. Anti-patterns and example_queries MUST be specific to this "
            "channel's content (no generic clichés). When done, delete this file."
        ),
        "samples": samples_json,
        "schema": {
            "description": "One-paragraph SKILL.md frontmatter description that triggers the skill on relevant user queries.",
            "purpose": "2-3 sentence purpose statement.",
            "anti_patterns": "List of 3-6 channel-specific failure modes to avoid (strings).",
            "example_queries": "List of 5-7 real example questions this advisor handles well (strings).",
        },
        "output_paths": {
            "skill_md": str(advisor_dir / "SKILL.md"),
            "agents_md": str(advisor_dir / "AGENTS.md"),
            "benchmark_json": str(advisor_dir / "evals" / "benchmark.json"),
            "workflow_md": str(advisor_dir / "references" / "workflow.md"),
        },
        "corpus_meta": meta,
        "intent": user_intent,
        "answer_language": answer_language,
        "quote_style": quote_style,
        "channel_display_name": channel_display_name,
    }
    (advisor_dir / ".pending-llm-draft.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )


def _no_llm_postrun(advisor_dir: pathlib.Path) -> str:
    return (
        f"✓ Data pipeline complete: {advisor_dir}\n"
        f"✓ Stub SKILL.md written at: {advisor_dir}/SKILL.md\n"
        f"✓ AI hand-off file: {advisor_dir}/.pending-llm-draft.json\n"
        "\n"
        "For Claude Code: I (the AI in chat) will read .pending-llm-draft.json,\n"
        "draft the polished SKILL.md / AGENTS.md / evals/benchmark.json myself,\n"
        "and remove the .pending-llm-draft.json marker.\n"
    )


def run(*, channels, since, until, max_n, playlists, title_include, title_exclude,
        ids, out_dir, whisper_skip_if_no_captions, multilingual, vendor, no_evals,
        from_natural_language, intent, answer_language, quote_style, yes,
        no_llm=None,
        cookies_from_browser: str | None = None,
        cookies_file: str | None = None) -> dict:
    """End-to-end advisor bootstrap.

    Returns a dict {guide, meta, summary}. The CLI prints `guide`.
    """
    out = pathlib.Path(out_dir).expanduser()
    cookies = resolve_cookies(
        from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )

    # 1) Handle pre-existing non-empty output dir.
    if out.exists() and any(out.iterdir()):
        # Update mode requires references/selection_filter.yaml to drive the
        # refresh. If it's missing, the dir is partially populated and we
        # cannot safely delegate without crashing opaquely.
        if not (out / "references" / "selection_filter.yaml").exists():
            raise click.UsageError(
                f"{out} appears partially populated (no references/selection_filter.yaml). "
                f"To restart from scratch: rm -rf {out}. To force update from current state, "
                "add the missing files manually."
            )
        if not yes:
            if not click.confirm(f"{out} is not empty. Switch to update mode?", default=True):
                raise click.Abort()
        from .update import run as _update
        return _update(advisor_dir=str(out), yes=True)

    out.mkdir(parents=True, exist_ok=True)
    prog = Progress(out)

    # 2) Resolve candidate videos from channels + playlists.
    prog.stage("resolving",
               message=f"Resolving {len(channels) + len(playlists)} channel(s)/playlist(s)...")
    # When only --max limits selection (no date/title narrowing, no manual
    # ids), cap the channel listing itself: the /videos tab is newest-first,
    # so the top N×3 entries are guaranteed to contain the N most recent
    # videos. Without the cap, the per-video date-resolution second pass in
    # resolve_channel() runs over the ENTIRE channel (1000+ yt-dlp calls on
    # large channels) just to throw away all but N.
    listing_limit = None
    if max_n and not since and not until and not title_include and not ids:
        listing_limit = max_n * 3  # headroom for title_exclude (shorts) drops
    candidates: list[dict] = []
    for ch in channels:
        candidates.extend(resolve_channel(ch, limit=listing_limit, cookies=cookies))
    for pl in playlists:
        candidates.extend(resolve_channel(pl, limit=listing_limit, cookies=cookies))

    # 3) Build filter and apply.
    manual_ids: set[str] = set()
    if ids:
        manual_ids = {
            line.strip()
            for line in pathlib.Path(ids).read_text().splitlines()
            if line.strip()
        }
    # When the user passes --playlist, scope the final selection to videos
    # in those playlists (intersection with channel results). An empty set
    # disables playlist scoping entirely.
    from .update import _playlist_ids_from_urls
    cfg = FilterConfig(
        since=since.date() if since else None,
        until=until.date() if until else None,
        max=max_n,
        title_include=title_include,
        title_exclude=title_exclude,
        playlist_ids=_playlist_ids_from_urls(list(playlists)),
        manual_ids=manual_ids,
    )
    prog.stage("filtering", message="Applying selection filters")
    selected = apply_filters(candidates, cfg)
    if not selected:
        raise click.UsageError("Found 0 videos. Loosen filters or check dates.")
    prog.videos(len(selected))

    # 4) Confirm (skipped when --yes or --from-natural-language).
    n = len(selected)
    if not yes and not from_natural_language:
        click.echo(
            f"Found {n} videos. Estimated ingest: ~{n * 5}s captions + "
            "Whisper fallback for any without subtitles."
        )
        click.confirm("Continue?", abort=True)

    # 5) Ingest cascade (captions API → yt-dlp captions → Whisper).
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    prog.stage("ingesting",
               message=f"Fetching transcripts for {len(selected)} videos")
    summary = run_ingest(
        selected, out / "transcripts",
        skip_whisper=whisper_skip_if_no_captions,
        cookies=cookies,
        progress=prog,
    )

    # 6) Keyword index.
    prog.stage("indexing", message="Building BM25 keyword index")
    build_index(out / "transcripts", out / "references")

    # 7) Semantic embeddings.
    prog.stage("embedding", message="Building dense embeddings")
    model_name = "BAAI/bge-m3" if multilingual else "BAAI/bge-small-en-v1.5"
    build_embeddings(out / "transcripts", out / "references", model_name=model_name)

    # 8) Detect corpus language from sampled transcripts.
    corpus_lang = _detect_corpus_language(out / "transcripts")

    # 9) Resolve quote style.
    qs = _resolve_quote_style(quote_style, answer_language, corpus_lang)

    # 10) corpus_meta.json (spec §4.2).
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pub_dates = [v["published_date"] for v in selected if v.get("published_date")]
    date_range = (
        f"{min(pub_dates)} → {max(pub_dates)}" if pub_dates else ""
    )
    meta = {
        "corpus_version": f"{date.today().isoformat()}/{n}",
        "n_videos": n,
        "channels": list(channels) + list(playlists),
        "date_range": date_range,
        "corpus_language": corpus_lang,
        "answer_language": answer_language,
        "quote_style": qs,
        "embedding_model": model_name,
        "embedding_dim": 1024 if multilingual else 384,
        "transcript_source_counts": summary.get("by_source", {}),
        "last_bootstrap": now_iso,
        "last_update": now_iso,
        "latest_video_date": max(pub_dates) if pub_dates else "",
        "generator_version": "0.1.0",
    }
    (out / "references").mkdir(parents=True, exist_ok=True)
    (out / "references" / "corpus_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )

    # 11) selection_filter.yaml snapshot.
    (out / "references" / "selection_filter.yaml").write_text(dump_yaml({
        "version": 1,
        "channels": list(channels),
        "since": cfg.since.isoformat() if cfg.since else None,
        "until": cfg.until.isoformat() if cfg.until else None,
        "max": cfg.max,
        "title_include": cfg.title_include,
        "title_exclude": cfg.title_exclude,
        "playlists": list(playlists),
        "extra_ids": sorted(cfg.manual_ids),
        "created_at": now_iso,
    }))

    # 12) Vendor/symlink _lib + install scripts wrappers.
    _install_runtime(out, vendor)

    # 14) Decide LLM mode.
    # Explicit no_llm wins; if None, auto-detect from ANTHROPIC_API_KEY.
    from ._llm import should_skip_llm
    effective_no_llm = should_skip_llm(no_llm)

    channel_display = ", ".join(list(channels) + list(playlists))

    if effective_no_llm:
        # Skip LLM-driven steps. Write stub SKILL.md + hand-off file for the
        # calling AI (which IS the LLM when invoked from Claude Code).
        prog.stage("drafting",
                   message="Writing stub SKILL.md for AI hand-off")
        _write_stub_skill(
            out,
            meta=meta,
            user_intent=intent or "",
            answer_language=answer_language,
            quote_style=qs,
            channel_display_name=channel_display,
        )
        _write_pending_llm_draft(
            out,
            meta=meta,
            user_intent=intent or "",
            answer_language=answer_language,
            quote_style=qs,
            channel_display_name=channel_display,
        )
        # No evals in --no-llm mode (benchmark generation needs an LLM).
        prog.done(message="Advisor ready (AI hand-off pending).")
        return {
            "guide": _no_llm_postrun(out),
            "meta": meta,
            "summary": summary,
        }

    # LLM-driven SKILL.md + scaffold.
    prog.stage("drafting", message="LLM drafting SKILL.md")
    gen_skill(
        out,
        user_intent=intent or "",
        answer_language=answer_language,
        quote_style=qs,
        channel_display_name=channel_display,
    )

    # 15) Evals.
    if not no_evals:
        prog.stage("evals", message="Running benchmark evals")
        gen_bench(out)
        run_evals(out)

    # 16) Postrun guide.
    prog.done(message="Advisor ready.")
    return {
        "guide": bootstrap_guide(out),
        "meta": meta,
        "summary": summary,
    }
