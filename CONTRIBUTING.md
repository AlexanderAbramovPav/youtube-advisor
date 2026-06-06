# Contributing to youtube-advisor

Issues and pull requests welcome. A few orientation notes:

## Project layout

- `src/youtube_advisor/` — Python package (filters / fetch_channel / ingest tiers / build_index / build_embeddings / search / gen_skill_md / gen_benchmark / eval_runner / postrun_guide / bootstrap / update / cli). Uses the [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).
- `templates/` — Jinja2 scaffolds rendered into each emitted advisor.
- `tests/` — unit + integration + E2E. Tier-1/Tier-2 ingest tests use mock subprocess + fixture VTT files (no live network). E2E tests mock at module boundaries and exercise the real index/search/eval pipeline.

## Running tests

```bash
cd youtube-advisor && source .venv/bin/activate
pytest -q                    # default: all unit + E2E (mocked-network only)
pytest -m slow               # requires RUN_SLOW=1 — actual fastembed model + faster-whisper
pytest -m e2e                # E2E only
```

## Adding a fixture channel

`tests/fixtures/yc_flat.json` is a `yt-dlp --flat-playlist -J` snapshot of `@ycombinator/videos`. To add another channel for testing, run:

```bash
yt-dlp --flat-playlist -J --playlist-end 5 https://www.youtube.com/@<handle>/videos > tests/fixtures/<handle>_flat.json
```

then patch `subprocess.run` in your test to return this JSON (see `tests/test_fetch_channel.py`).

## Swapping the embedding model

The default is `BAAI/bge-small-en-v1.5` (384-dim, 24 MB). To experiment with another `fastembed`-supported model, pass `--multilingual` to use `BAAI/bge-m3` (1 GB) or fork `youtube_advisor/build_embeddings.py` to accept a custom model name and re-run `build_embeddings` with `incremental=False` to re-encode the corpus.

## Adding language tests

Captions language is auto-detected from the YT API; advisor answer language is configurable. To add a language-specific assertion, ingest a Russian or Spanish video (e.g., a Lex Fridman interview with non-English content) and add a test that verifies `transcript_language` in the frontmatter and `corpus_language` in `corpus_meta.json`.

## Style

- Follow the project's CLAUDE.md conventions: no defensive code for impossible cases; only validate at system boundaries; minimal comments (only when the WHY is non-obvious).
- TDD: write the failing test first, then minimal implementation, then commit.
- One concern per file. Files growing past ~150 lines are a smell.

## Releasing

We use [Semantic Versioning](https://semver.org/) and maintain a [Keep a Changelog](https://keepachangelog.com/) at `CHANGELOG.md`.

To cut a release:

1. Move the `## [Unreleased]` block's entries under a new `## [X.Y.Z] — YYYY-MM-DD` heading.
2. Bump `version` in `pyproject.toml` to match.
3. Commit: `release: vX.Y.Z`.
4. Tag: `git tag -s vX.Y.Z -m "vX.Y.Z"`.
5. Push: `git push && git push --tags`.

The release workflow (`.github/workflows/release.yml`) creates a GitHub Release with the CHANGELOG entry as the body.

Tags that contain a hyphen (e.g. `v0.2.0-beta1`) are auto-marked as prereleases.
