# Changelog

All notable changes to youtube-advisor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Embedding builds now spot changed transcripts by a hash of their content. The hashes are stored in `embeddings.npz`, so a `git checkout` or `touch` no longer re-embeds the corpus. An `embeddings.npz` written by an older version gets its hashes on the next run, and its vectors are kept rather than recomputed. Only transcripts newer than that file's last build are re-embedded.

### Fixed
- A full build now embeds a transcript that has no rows in `embeddings.npz` even when its mtime is old, for example a file restored with `cp -p`.
- `update` no longer stops on a damaged `embeddings.npz`: it logs a warning and rebuilds all embeddings.
- `embeddings.npz` is written to a temp file and then moved into place, so a crash during the write keeps the old file intact.
- Incremental updates re-embed a transcript that changed since the last build, and drop the rows of deleted transcripts so they no longer show up in search.
- A full rebuild after transcripts were removed no longer reuses the old file with the removed videos in it. The eval rollback in `update` now uses the incremental build, so it drops those rows without re-embedding the rest.
- A transcript edited while a build is running is re-embedded on the next update: the npz now stores when the build started, and changes are checked against that time.

## [0.1.0] — 2026-06-06

First public beta. Build a Claude Code advisor skill from any YouTube channel.

### Added
- Two-action user UX: paste a channel + intent in chat → AI builds the advisor → invoke `/<channel-slug>-advisor` to ask questions.
- Three-tier transcript cascade: `youtube-transcript-api` (free YT captions) → `yt-dlp` VTT fallback → local Whisper via `faster-whisper` or scriba.
- Hybrid retrieval: BM25 + dense embeddings (`fastembed` with `bge-small-en-v1.5` default, `bge-m3` opt-in for multilingual) fused via Reciprocal Rank Fusion.
- Citation-first answer contract: every answer carries verbatim quotes with `youtube.com/watch?v=ID&t=SECONDS` deep-links.
- LLM-drafted SKILL.md and benchmark per advisor (when `ANTHROPIC_API_KEY` is set) — OR — handoff to the AI agent in chat via `.pending-llm-draft.json` when no API key is set (`--no-llm` mode, auto-detected).
- Incremental updates: saved selection filter is replayed on `update`, only new videos are fetched and indexed.
- Eval scaffolding: generated benchmark + runner with regression detection (`--run-evals`).
- Selection DSL: `--channel`, `--playlist`, `--since`, `--until`, `--max`, `--title-include`, `--title-exclude`, `--ids`.
- Anti-bot defaults: 1–5s randomized sleep between yt-dlp requests, exponential-backoff retries on HTTP 429.
- Live progress: bootstrap writes `<advisor>/.progress.json` on every stage change and per-video ingest; `scripts/status.sh <advisor>` (and `youtube-advisor status <advisor>`) print a one-line snapshot for a peek from a side terminal.
- YouTube JS-challenge solver: yt-dlp invocations now pass `--remote-components ejs:github` (auto-detected via `yt-dlp --help` probe — silently omitted on older yt-dlp builds without the flag) and `--extractor-args youtube:formats=missing_pot` to avoid HTTP 429s and PO-token errors. Requires `deno` on PATH at runtime; `install.sh` now warns if missing.
- Cookie support: `--cookies-from-browser {chrome,firefox,safari,brave,edge,…}`, `--cookies <file>`, env vars `YOUTUBE_ADVISOR_COOKIES_BROWSER` / `YOUTUBE_ADVISOR_COOKIES_FILE`, and auto-discovery of `~/.config/youtube-advisor/cookies.txt`.
- OSS scaffolding: MIT license, scriba-style README, CONTRIBUTING, THIRD_PARTY_LICENSES, `bash scripts/install.sh` helper.
- Python package uses `src/` layout (`src/youtube_advisor/`).
- 183 tests (unit + integration + E2E).

### Known limitations
- Single channel per advisor (multi-channel rollups planned for v0.2).
- Length-based filtering (e.g. "interviews > 60 minutes") not yet a selection flag.
- LangDetect on update doesn't re-fire — corpus language persists from bootstrap.

[Unreleased]: https://github.com/AlexanderAbramovPav/youtube-advisor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AlexanderAbramovPav/youtube-advisor/releases/tag/v0.1.0
