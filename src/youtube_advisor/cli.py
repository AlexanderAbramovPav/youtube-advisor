from __future__ import annotations
import click

from . import bootstrap as bs_mod


@click.group()
def main():
    """youtube-advisor: build & update Claude Code advisor skills from YouTube channels."""


@main.command()
@click.option("--channel", "channels", multiple=True, required=True,
              help="Channel URL or @handle (repeatable).")
@click.option("--since", type=click.DateTime(["%Y-%m-%d"]), default=None,
              help="Only include videos published on/after this date (YYYY-MM-DD).")
@click.option("--until", type=click.DateTime(["%Y-%m-%d"]), default=None,
              help="Only include videos published on/before this date (YYYY-MM-DD).")
@click.option("--max", "max_n", type=int, default=500,
              help="Maximum number of videos to ingest (most recent first).")
@click.option("--playlist", "playlists", multiple=True,
              help="Playlist URL to also include (repeatable).")
@click.option("--title-include", default=None,
              help="Regex; only keep videos whose title matches.")
@click.option("--title-exclude", default=r"#?shorts\b",
              help="Regex; drop videos whose title matches (defaults to shorts).")
@click.option("--ids", type=click.Path(exists=True), default=None,
              help="Path to a file with extra video IDs (one per line) to always include.")
@click.option("--out", "out_dir", type=click.Path(), required=True,
              help="Output advisor directory.")
@click.option("--whisper-skip-if-no-captions", is_flag=True,
              help="Skip Whisper fallback for videos without captions.")
@click.option("--multilingual", is_flag=True,
              help="Use BAAI/bge-m3 multilingual embeddings (1024-d).")
@click.option("--vendor", is_flag=True,
              help="Copy youtube_advisor pkg into <out>/scripts/_lib instead of symlinking.")
@click.option("--no-evals", is_flag=True,
              help="Skip benchmark generation and eval run.")
@click.option("--from-natural-language", is_flag=True,
              help="Skip interactive confirm (the calling agent confirmed in chat).")
@click.option("--intent", default=None,
              help="Free-text intent passed to the SKILL.md drafter.")
@click.option("--answer-language", type=click.Choice(["auto", "en", "ru"]), default="auto")
@click.option("--quote-style", type=click.Choice(["auto", "translation-first", "quote-only"]), default="auto")
@click.option("--no-llm", is_flag=True, default=None,
              help="Skip LLM steps (SKILL.md drafting + eval generation). "
                   "Auto-detected when ANTHROPIC_API_KEY is unset. The calling "
                   "Claude Code agent drafts SKILL.md itself via the "
                   ".pending-llm-draft.json hand-off file.")
@click.option("--cookies-from-browser", "cookies_from_browser", default=None,
              help="Use cookies from a logged-in browser (chrome/firefox/safari/...). "
                   "Bypasses YT anti-bot on blocked IPs.")
@click.option("--cookies", "cookies_file", type=click.Path(), default=None,
              help="Path to a Netscape-format cookies.txt. Alternative to "
                   "--cookies-from-browser.")
@click.option("--yes", is_flag=True, help="Skip all confirmation prompts.")
def bootstrap(**kw):
    """Create a new advisor skill from a YouTube channel."""
    out = bs_mod.run(**kw)
    click.echo(out["guide"])


@main.command()
@click.option("--advisor", "advisor_dir", type=click.Path(exists=True), required=True)
@click.option("--since", type=click.DateTime(["%Y-%m-%d"]), default=None)
@click.option("--max", "max_n", type=int, default=None)
@click.option("--ids", type=click.Path(exists=True), default=None)
@click.option("--reindex-only", is_flag=True)
@click.option("--regen-skill-md", is_flag=True)
@click.option("--run-evals", "run_evals_", is_flag=True)
@click.option("--prune", is_flag=True)
@click.option("--no-llm", is_flag=True, default=None,
              help="Skip LLM steps (SKILL.md re-draft + evals). "
                   "Auto-detected when ANTHROPIC_API_KEY is unset.")
@click.option("--cookies-from-browser", "cookies_from_browser", default=None,
              help="Use cookies from a logged-in browser (chrome/firefox/safari/...). "
                   "Bypasses YT anti-bot on blocked IPs.")
@click.option("--cookies", "cookies_file", type=click.Path(), default=None,
              help="Path to a Netscape-format cookies.txt. Alternative to "
                   "--cookies-from-browser.")
@click.option("--yes", is_flag=True)
def update(**kw):
    """Update an existing advisor with new videos from the saved filter."""
    from . import update as up_mod
    result = up_mod.run(**kw)
    click.echo(result["guide"])


@main.command()
@click.argument("advisor_dir", type=click.Path(exists=True))
def status(advisor_dir):
    """Print a one-line live status snapshot for a running advisor bootstrap.

    Reads <advisor>/.progress.json. Silent (exit 0) if no progress file exists.
    """
    import json
    import pathlib

    p = pathlib.Path(advisor_dir) / ".progress.json"
    if not p.exists():
        return
    try:
        state = json.loads(p.read_text())
    except Exception:
        return
    stage = state.get("stage", "?")
    done = state.get("videos_done", 0)
    total = state.get("videos_total", 0)
    elapsed = state.get("elapsed_sec", 0) or 0
    eta = state.get("eta_sec")
    msg = state.get("message") or ""
    mm = f"{elapsed//60:02d}:{elapsed%60:02d}"
    parts = [f"\U0001F3AC {stage}"]
    if total:
        parts.append(f"{done}/{total}")
    parts.append(mm)
    if eta is not None and eta > 0:
        parts.append(f"ETA {eta//60:02d}:{eta%60:02d}")
    if msg and stage in {"init", "resolving", "filtering", "drafting", "evals", "done"}:
        parts.append("· " + msg[:60])
    click.echo(" · ".join(parts))


if __name__ == "__main__":
    main()
