# Changelog

All notable changes to youtube-advisor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Video length selection on `bootstrap`: `--min-duration` and `--max-duration` (inclusive bounds; accept plain seconds like `3600`, units like `90m`, `2h`, `1h30m`, `MM:SS`, and `H:MM:SS`). Runs before `--max`, keeps manual `--ids` unconditionally, and keeps videos with unknown length (missing, `0`, or non-numeric) with a single warning showing the count. Saved in `references/selection_filter.yaml` and reused automatically by `update` (older filter files without the keys remain unbounded and untouched). Readable bounds (`≥ 1h`, `≤ 20m`) are shown in the bootstrap confirmation prompt and post-run guide.

### Changed
- Requires `youtube-transcript-api>=1.0.0`, the first version with `RequestBlocked` / `IpBlocked` and the `YouTubeTranscriptApi().list()` API already in use.
- Embedding builds now spot changed transcripts by a hash of their content. The hashes are stored in `embeddings.npz`, so a `git checkout` or `touch` no longer re-embeds the corpus. An `embeddings.npz` written by an older version gets its hashes on the next run, and its vectors are kept rather than recomputed. Only transcripts newer than that file's last build are re-embedded. To force a full re-embed, delete `references/embeddings.npz`.

### Fixed
- After a bootstrap that YouTube blocked before any transcript landed, `update` now detects the corpus language from the transcripts it fetches and resolves an `auto` quote style again (a fixed `--quote-style` is kept). While the AI hand-off is still pending, the stub SKILL.md and `.pending-llm-draft.json` are rewritten with the new values.
- The first `update` that finishes without a block after a blocked bootstrap generates and runs the evals the bootstrap skipped. A `--no-llm` update leaves them pending for a later run. A failure there is logged and retried by the next update; it does not fail the update. An existing `evals/benchmark.json` (for example one the AI wrote) is kept and only run, and re-running `bootstrap --no-evals` drops the pending evals.
- `--whisper-skip-if-no-captions` is saved in `selection_filter.yaml`, and `update` honours it; before, `update` always ran Whisper.
- With `--cookies-from-browser`, the captions API (which cannot use browser cookies) no longer runs at 8 concurrent calls with no pause: it runs at most 2 at once with a 0.5–2s jitter, as without cookies. yt-dlp, which does use the browser cookies, keeps the 8-worker pool.
- A YouTube block (HTTP 429, `IpBlocked` / `RequestBlocked`, "Sign in to confirm you're not a bot", "rate-limited by YouTube") is no longer treated as "no captions". The run sends no more requests to YouTube and never falls back to Whisper for a blocked video. The blocked video is logged as `failed:blocked`, videos not tried yet are left alone, and the next `update` retries both. `bootstrap` and `update` print the retry command and exit with code 1. With `--cookies-from-browser`, a block on the cookie-less captions API falls back to yt-dlp with your cookies; a block there stops the run as well.
- A yt-dlp captions run that exits non-zero for another reason (private or unavailable video, age gate, network error) is recorded as `failed:YtDlpFailed` and no longer sent to Whisper.
- `update` counts only the videos that actually landed in its "+N new videos" line.
- A blocked `update` skips `--regen-skill-md` and `--run-evals` rather than spending LLM calls on a partial corpus. Failed videos are no longer added to `transcript_source_counts`.
- `update --regen-skill-md` drafts for the intent given to `bootstrap` (now stored in `corpus_meta.json`), not "(regenerated)". It also removes a leftover `.pending-llm-draft.json` once the LLM draft is written.
- Re-running `bootstrap` into an existing advisor passes `--cookies-from-browser` / `--cookies` and `--no-llm` on to update mode.
- `corpus_meta.json` `n_videos` from `bootstrap` counts only the videos that were ingested.
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
- LangDetect on update doesn't re-fire — corpus language persists from bootstrap.

[Unreleased]: https://github.com/AlexanderAbramovPav/youtube-advisor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AlexanderAbramovPav/youtube-advisor/releases/tag/v0.1.0
