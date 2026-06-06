<p align="center">
  <img src="docs/logo.png" alt="youtube-advisor" width="180" />
</p>

<h1 align="center">youtube-advisor</h1>

<p align="center">
  <strong>Turn any YouTube channel into an AI assistant skill that answers questions with verbatim quotes and timestamped video links.</strong>
</p>

<p align="center">
  <a href="https://github.com/AlexanderAbramovPav/youtube-advisor/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/AlexanderAbramovPav/youtube-advisor?style=flat-square&label=stars&logo=github&color=f7b500" /></a>
  <a href="https://github.com/AlexanderAbramovPav/youtube-advisor/actions/workflows/test.yml"><img alt="tests" src="https://img.shields.io/github/actions/workflow/status/AlexanderAbramovPav/youtube-advisor/test.yml?style=flat-square&label=tests&color=2188ff" /></a>
  <a href="https://github.com/AlexanderAbramovPav/youtube-advisor/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/AlexanderAbramovPav/youtube-advisor?style=flat-square&label=release&color=2188ff" /></a>
  <a href="https://pypi.org/project/youtube-advisor/"><img alt="PyPI" src="https://img.shields.io/pypi/v/youtube-advisor?style=flat-square&label=pypi&color=3775a9&logo=pypi&logoColor=white" /></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-2ea043?style=flat-square" /></a>
  <img alt="100% Local" src="https://img.shields.io/badge/privacy-100%25_local-e36209?style=flat-square" />
  <img alt="No API key" src="https://img.shields.io/badge/zero-api_keys-8957e5?style=flat-square" />
  <img alt="Works with any AI" src="https://img.shields.io/badge/works_with-any_AI_agent-a371f7?style=flat-square" />
</p>

---

## What it feels like

```
You: build an advisor on @ycombinator,
     last 200 videos, hiring + fundraising

AI:  ✓ Resolving @ycombinator (200 videos)
     ✓ Fetching transcripts (3 min)
     ✓ Indexing + embedding
     ✓ Advisor ready: /yc-advisor

You: /yc-advisor how should I think about hiring my first engineer?
AI:  → answer with quoted videos and timestamps
```

## Install

Paste this in your AI chat (Claude Code, Cursor, Codex CLI, etc.):

> Install youtube-advisor for me. PyPI: `pip install youtube-advisor`. For Claude Code skill discovery, also clone https://github.com/AlexanderAbramovPav/youtube-advisor into `~/.claude/skills/youtube-advisor/`.

The AI installs the CLI from PyPI, makes the skill discoverable in `~/.claude/skills/`, checks `ffmpeg` and `uv`. You walk away.

**Or just install the CLI for scripting** (no AI integration):

```bash
pip install youtube-advisor
```

You get the `youtube-advisor` command. See [Advanced](#advanced--for-ci-and-scripting) for CLI usage.

## Use it

**Create an advisor.** Drop a channel URL or handle in chat with a one-sentence intent:

```
build an advisor on @ycombinator from the last 200 videos,
focused on hiring and fundraising
```

```
make an advisor on @lexfridman, interviews from 2024 onwards
```

The assistant resolves the channel, fetches transcripts, builds indices, drafts the advisor's voice, and runs a self-check. No flags, no paths, no shell. You get a new slash command like `/yc-advisor`.

**Ask the advisor.**

```
/yc-advisor how should I think about hiring my first engineer?
```

Every answer comes back with verbatim quotes and clickable timestamps linking to the exact second of the source.

**Refresh later.** One sentence:

```
refresh @ycombinator — check for new videos this week
```

The assistant replays your original selection filter, fetches only what's new, re-indexes the delta, and tells you what changed.

## Why

- **Local-first.** Transcripts and indices live on your disk. No SaaS account, no per-query fees, no vendor that can take your corpus offline.
- **Verbatim quotes with timestamps.** Every answer is grounded in real transcript text and links to `youtube.com/watch?v=ID&t=SECONDS` — click and the video opens at that exact second.
- **Uses your AI's brain.** Skill drafting and answering run through whatever assistant you're already using — nothing extra to configure.
- **Free captions when available; local Whisper when not.** A three-tier cascade keeps cost near zero and falls back to on-device transcription only when YouTube has none.
- **Incremental updates that respect your filter.** Your original selection (channel + date range + title filters + max count) is saved and replayed on every refresh.

## What an answer looks like

> *"The scariest fundraising stories are the ones where the founder spends six months on an 'almost-yes' from an investor who eventually just ghosts."*
>
> — [Dalton & Michael: Fundraising Mistakes (2024-03-12), 14:22](https://youtube.com/watch?v=abc123&t=862)

Each citation is a real line from a real video, anchored to the second. Click through and verify.

## Limitations

- **One channel per advisor.** Multi-channel rollups ("YC + Naval + a16z combined") are planned for v0.2.
- **Whisper fallback is CPU-bound.** Videos without YT captions fall back to local Whisper — roughly 1 hour of compute per 1 hour of audio on Apple Silicon. The assistant warns you and asks before going that route.
- **YouTube anti-bot.** Defaults (1–5s randomized sleep, exponential-backoff retries on 429) handle most cases. On persistent blocks the assistant will offer to use cookies from your logged-in browser.

## Works with other AI tools

`AGENTS.md` is a tool-agnostic mirror of `SKILL.md`. Wire it into whichever tool you use:

| Your tool          | Where to point it                                                              | How to invoke                            |
| ------------------ | ------------------------------------------------------------------------------ | ---------------------------------------- |
| Claude Code        | clone into `~/.claude/skills/youtube-advisor/`                                 | `/youtube-advisor <request>`             |
| OpenAI Codex CLI   | clone anywhere, place `AGENTS.md` at `~/.codex/AGENTS.md` (or project root)    | "build me an advisor on `<channel>`"     |
| Cursor             | clone anywhere, copy `AGENTS.md` to `.cursor/rules/youtube-advisor.md`         | `@youtube-advisor` or natural language   |
| Continue.dev       | clone anywhere, register `youtube-advisor` as a custom slash command           | `/youtube-advisor <request>`             |
| Aider              | clone anywhere, `aider --read <install-dir>/AGENTS.md`                         | natural language in chat                 |
| Goose (Block)      | clone anywhere, add as a shell-command extension                               | mention youtube-advisor in chat          |
| No AI at all       | clone anywhere                                                                 | `youtube-advisor bootstrap --channel @… --out …` |

<details>
<summary><strong>Advanced — for CI and scripting</strong></summary>

Everything below is for power users running outside an AI chat (CI, cron, scripts). **You almost never need this.**

### Manual install

```bash
git clone https://github.com/AlexanderAbramovPav/youtube-advisor ~/.claude/skills/youtube-advisor
cd ~/.claude/skills/youtube-advisor && bash scripts/install.sh
```

### CLI

```bash
# Bootstrap
youtube-advisor bootstrap \
  --channel @ycombinator \
  --since 2022-01-01 \
  --max 200 \
  --title-include "Founder Stories|How to" \
  --title-exclude "shorts|#shorts" \
  --out ~/.claude/skills/yc-advisor \
  --yes

# Update (replays the saved selection filter)
youtube-advisor update --advisor ~/.claude/skills/yc-advisor --yes

# Update — override date, reindex only, or run regression evals
youtube-advisor update --advisor ~/.claude/skills/yc-advisor --since 2026-01-01 --yes
youtube-advisor update --advisor ~/.claude/skills/yc-advisor --reindex-only
youtube-advisor update --advisor ~/.claude/skills/yc-advisor --run-evals --yes
```

`youtube-advisor bootstrap --help` and `youtube-advisor update --help` print the full flag list.

### Bypassing YT anti-bot with browser cookies

When unauthenticated channel-resolution gets 429-blocked, pass cookies from a logged-in browser. yt-dlp reads them straight from the browser profile — no copy-paste:

```bash
youtube-advisor bootstrap --channel @ycombinator --out ~/.claude/skills/yc-advisor \
  --cookies-from-browser chrome --yes

# Or a Netscape-format cookies.txt:
youtube-advisor bootstrap --channel @ycombinator --out ~/.claude/skills/yc-advisor \
  --cookies ~/secrets/yt-cookies.txt --yes
```

Resolution order: explicit flag → `YOUTUBE_ADVISOR_COOKIES_BROWSER` / `YOUTUBE_ADVISOR_COOKIES_FILE` env → `~/.config/youtube-advisor/cookies.txt` if present → none. With cookies set the captions pool stays at concurrency 8; without them it drops to 2 + per-call jitter so YT is less likely to block you.

### Vendoring `_lib`

`youtube-advisor bootstrap --vendor` copies the `youtube_advisor` Python package into the emitted advisor's `scripts/_lib/`. Useful if you want the advisor directory to be fully portable (pushed to a separate Git repo and cloned elsewhere without a Python install of `youtube-advisor`).

### Eval-driven regression detection

`youtube-advisor update --advisor <path> --run-evals` reruns the channel's generated benchmark after sync and prompts to roll back if pass rate dropped.

</details>

## License

[MIT](./LICENSE) — © 2026 Alexander Abramov.

Runtime dependencies (yt-dlp, youtube-transcript-api, fastembed, faster-whisper, rank-bm25, ffmpeg, scriba) are installed locally, not vendored. Their licenses are catalogued in [`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md).

## Acknowledgments

This tool is glue around several excellent OSS projects:

- [**yt-dlp**](https://github.com/yt-dlp/yt-dlp) — channel resolution, caption download, audio extraction.
- [**youtube-transcript-api**](https://github.com/jdepoix/youtube-transcript-api) — Jonas Depoix — Tier 1 captions fetcher.
- [**fastembed**](https://github.com/qdrant/fastembed) — Qdrant — local embedding inference with `bge-small-en-v1.5` (BAAI).
- [**rank-bm25**](https://github.com/dorianbrown/rank_bm25) — Dorian Brown — BM25 ranking.
- [**faster-whisper**](https://github.com/SYSTRAN/faster-whisper) — SYSTRAN — local ASR fallback.
- [**scriba**](https://github.com/AlexanderAbramovPav/scriba) — sibling skill — alternative Whisper transcription backend.
- [**uv**](https://github.com/astral-sh/uv) (Astral) for env setup, and [`ffmpeg`](https://ffmpeg.org/) for audio extraction.
- Conceptual lineage: [yc-avisor](https://github.com/AlexanderAbramovPav/cowork) — the hand-rolled YC-channel-specific predecessor.
