from __future__ import annotations
import json
import pathlib

from ._naming import short_slug

# Back-compat alias for tests that import the underscore name.
_short_slug = short_slug


def _latest_eval_summary(advisor_dir: pathlib.Path) -> dict | None:
    results_dir = advisor_dir / "evals" / "results"
    if not results_dir.exists():
        return None
    files = sorted(results_dir.glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def _source_counts_text(by_source: dict) -> str:
    """Format like '184 via captions, 3 via Whisper'."""
    parts = []
    if by_source.get("captions"):
        parts.append(f"{by_source['captions']} via captions")
    whisper_total = sum(v for k, v in by_source.items() if k.startswith("whisper"))
    if whisper_total:
        parts.append(f"{whisper_total} via Whisper")
    failed_total = sum(v for k, v in by_source.items() if k.startswith("failed"))
    if failed_total:
        parts.append(f"{failed_total} failed")
    return ", ".join(parts) if parts else "all transcripts"


def bootstrap_guide(advisor_dir: pathlib.Path) -> str:
    meta = json.loads((advisor_dir / "references" / "corpus_meta.json").read_text())
    eval_sum = _latest_eval_summary(advisor_dir)
    eval_line = (
        f"Evals: {eval_sum['passed']}/{eval_sum['total']} passing."
        if eval_sum
        else "Evals: not run."
    )
    sources_text = _source_counts_text(meta.get("transcript_source_counts", {}))
    slug_short = short_slug(advisor_dir.name)

    return f"""✓ Advisor ready: {advisor_dir}

Corpus: {meta['n_videos']} videos from {', '.join(meta.get('channels', []))} ({meta.get('date_range', '')}), {sources_text}.
Languages: corpus {meta.get('corpus_language', '?')}, answers {meta.get('answer_language', '?')} ({meta.get('quote_style', '?')} quote format).
Indices: keyword + semantic embeddings ({meta.get('embedding_model', '?')}).
{eval_line}

═══  How to use  ═══

In Claude Code:
   /{slug_short}-advisor   — then ask anything related to the corpus.

The advisor returns answers with verbatim quotes from source videos, each linked
to the exact timestamp.

═══  Keeping it fresh  ═══

When new videos appear on the channel:
   /youtube-advisor update {slug_short}-advisor

That's it — same filters, automatic incremental fetch + re-index.

═══  Editing the advisor  ═══

I drafted {advisor_dir}/SKILL.md describing this advisor's
personality and anti-patterns. If the answers feel off, open that file
and tweak the "Anti-patterns" or "Example queries" sections — your edits
are preserved across updates.
"""


def update_guide(
    advisor_dir: pathlib.Path,
    n_new: int,
    n_removed: int,
    eval_summary: dict | None = None,
) -> str:
    channel_slug = advisor_dir.name
    slug_short = short_slug(channel_slug)
    if n_new == 0 and n_removed == 0:
        return f"✓ {channel_slug}: 0 new, 0 removed, index unchanged.\n"

    meta = json.loads((advisor_dir / "references" / "corpus_meta.json").read_text())
    n_videos = meta.get("n_videos", "?")
    prev = n_videos - n_new + n_removed
    latest = (
        meta.get("latest_video_date")
        or meta.get("date_range", "").split("→")[-1].strip()
        or "?"
    )

    lines = [
        f"✓ {channel_slug} updated: +{n_new} new videos, {n_removed} removed.",
        f"   Corpus: {n_videos} videos (was {prev}). Latest: {latest}.",
    ]

    if eval_summary is not None:
        delta = eval_summary.get("delta_passed")
        if delta is None or delta == 0:
            delta_text = "(no change)"
        elif delta > 0:
            delta_text = f"(+{delta})"
        else:
            delta_text = f"({delta}, regression!)"
        lines.append(
            f"   Evals: {eval_summary['passed']}/{eval_summary['total']} passing {delta_text}."
        )

    lines.append(
        f"   Next: ask /{slug_short}-advisor anything, or run another update when more videos drop."
    )
    return "\n".join(lines) + "\n"
