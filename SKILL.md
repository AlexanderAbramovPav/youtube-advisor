---
name: youtube-advisor
description: Build or update a Claude Code advisor skill from any YouTube channel. Use when the user pastes a YouTube channel URL or handle, names a creator, or says "make an advisor on …", "build a skill from …", "обнови … advisor", "refresh that advisor". Triggers on bare URLs like https://youtube.com/@channel, @handles, or creator-name mentions.
---

# youtube-advisor — build a Claude Code advisor from any YouTube channel

## Two-action user contract

The user does exactly two things:

1. **Drops a channel URL/handle and a one-sentence intent in chat.**
2. **Uses the resulting `/<channel-slug>-advisor` skill** to ask questions about that channel.

**You (the AI agent) do everything in between.** Never expose `--since`, `--max`, `--out`, `bash`, or any CLI flag to the user. Never paste a shell command for the user to run. Compose every command silently, run it via Bash, and report results in plain language.

## When this skill triggers

- A YouTube channel URL, `@handle`, or playlist URL appears in the message.
- Creator name appears in a "build me an advisor on …" / "make a skill from …" / "make a YC advisor" framing.
- The user says "обнови yc-advisor", "refresh that advisor", "check for new videos".

## Workflow

### Step 1 — parse intent silently

Extract from the user's message:

- **Channel(s)** — URL, `@handle`, playlist URL, or a named creator.
  - If a creator is named without a URL, search via `yt-dlp ytsearch5:"<name>"` (or `yt-dlp --get-id ytsearch3:"<name> channel"`), then ASK the user "I found @navalr — Naval Ravikant's channel — use that one?" before continuing. Don't guess.
- **Selection hints**:
  - "last year" → `--since 2025-MM-DD` (today minus 1 year).
  - "long videos only" (no threshold) → `--title-exclude "shorts|#shorts"` (already default — no extra flag).
  - "only interviews longer than an hour" / "at least 30 minutes" → `--min-duration 1h` / `--min-duration 30m` (also accepts `3600`, `90m`, `1h30m`, `MM:SS`, `H:MM:SS`).
  - "under 20 minutes" / "between 20 and 60 minutes" → `--max-duration 20m` / `--min-duration 20m --max-duration 1h`.
  - "the Founder Stories series" → `--title-include "Founder Stories"`.
  - Pasted video URLs → write IDs to `/tmp/ids.txt`, pass `--ids /tmp/ids.txt`.
- **Intent sentence** — for the LLM SKILL.md drafter (`--intent "..."`).
- **Language preference** — match the chat language by default; pass `--answer-language` only if the user explicitly says.

If the message is ambiguous, ask AT MOST 3 short questions in plain language. NEVER list flag names. NEVER show CLI syntax.

### Step 2 — confirm in plain language

Print a one-line summary:

> "I'll fetch ~187 videos from @ycombinator (2022 → today), output to `~/.claude/skills/yc-advisor`. Roughly 20 minutes. Go?"

Only ask about Whisper time if the estimated duration is > 15 minutes for the Whisper fallback.

### Step 3 — execute via Bash (HIDDEN from the user)

```
youtube-advisor bootstrap \
  --channel @ycombinator \
  --since 2022-01-01 \
  --out ~/.claude/skills/yc-advisor \
  --from-natural-language --intent "YC startup advice with focus on hiring" \
  --answer-language auto \
  --quote-style auto \
  --yes
```

Run it via the Bash tool. The user does NOT see this command.

### Step 4 — stream progress sparsely

Reuse the scriba progress-streaming pattern:
- Emit a short status message on each stage transition (resolving channel, ingesting, indexing, embedding, drafting SKILL.md, running evals).
- Emit on ~25% audio-processed boundaries for Whisper transcription (NOT every 10%, NOT on a timer).
- NEVER use `/loop`, `ScheduleWakeup`, or any periodic wake-up — they cost prompt-cache misses ($).

#### Live status from a side terminal

While bootstrap is running, you can peek at progress without tailing the verbose stdout:

```
bash <repo>/scripts/status.sh <advisor-dir>
# or
youtube-advisor status <advisor-dir>
```

Both read `<advisor>/.progress.json` and print a one-line snapshot, e.g.
`🎬 ingesting · 4/187 · 02:14 · ETA 18:45`.

Use this when the user asks "how far along is it?" — read the progress JSON, don't tail bash output. The JSON updates atomically on every stage change and per-video ingest.

### Step 4 — polish the draft (YOU are the LLM)

Whenever `bootstrap` ran without `ANTHROPIC_API_KEY` (the default Claude Code case) — or when you passed `--no-llm` — the CLI writes a **stub** `SKILL.md` plus a hand-off file at `<advisor>/.pending-llm-draft.json`. **Do not** echo the postrun guide yet. Instead:

1. Read `<advisor>/.pending-llm-draft.json` (it contains the schema, ~8 transcript excerpts, the corpus metadata, the user intent, and the output paths).
2. Draft a polished `SKILL.md` directly from those samples — anti-patterns must be specific to THIS channel's actual failure modes (not generic clichés), and example_queries must be real questions this corpus can answer well.
3. Re-render `SKILL.md` and `AGENTS.md` at the paths listed in `output_paths`. Use the same Jinja templates the stub came from (`templates/advisor.SKILL.md.tmpl`, `advisor.AGENTS.md.tmpl`) — only the description/purpose/anti_patterns/example_queries change.
4. (Optional) Draft `evals/benchmark.json` (10–20 channel-grounded Q/A items with `expected_quote_video_ids` and `must_terms`) so the user can later run a quality check.
5. **Delete `.pending-llm-draft.json`** — that marker file's presence means "AI hasn't polished yet". Removing it signals the advisor is ready.

Only then echo the postrun guide.

### Step 5 — show the postrun guide

When `bootstrap` exits, it prints the "what's next" guide. Echo it to the user AS-IS (verbatim from the CLI output). Do NOT add commentary about what flags you used. The user should see only the guide and a brief "Done!" framing.

### Update flow

When the user says "обнови yc-advisor", "refresh the advisor", "check for new videos", "что нового на канале":

```
youtube-advisor update --advisor ~/.claude/skills/yc-advisor --yes
```

Same UX: never show the user the command. Echo the postrun summary. If the user provides a hint ("обнови, но только за эту неделю"), translate it to `--since <date>`.

Videos that failed or were blocked by YouTube are retried by the next update; there is no `--reingest` flag. After a blocked bootstrap, the first update that finishes without a block detects the corpus language from the transcripts it now has and generates the evals the bootstrap skipped (not with `--no-llm`).

## Worked examples (for YOU the agent — never shown to the user)

Each example shows: user message → flags YOU compose → what the user sees.

### Example 1 — bare URL + intent

User: `https://www.youtube.com/@ycombinator. I want a YC advisor focused on startup hiring and fundraising.`

You parse:
- channel = `@ycombinator`
- intent = `"YC startup advice with focus on hiring and fundraising"`
- language = match chat language (RU/EN auto)

You run:
```
youtube-advisor bootstrap \
  --channel @ycombinator \
  --out ~/.claude/skills/yc-advisor \
  --from-natural-language \
  --intent "YC startup advice with focus on hiring and fundraising" \
  --answer-language auto \
  --yes
```

User sees: confirmation summary → progress updates → postrun guide. NEVER sees the flags.

### Example 2 — date hint + duration bound

User: `pick interviews longer than 1 hour from 2024 onwards on @lexfridman`

You parse:
- channel = `@lexfridman`
- since = `2024-01-01`
- min_duration = `1h` (inclusive; combine with `--max-duration` for an upper bound)
- title_exclude default already drops shorts.

You run:
```
youtube-advisor bootstrap --channel @lexfridman --since 2024-01-01 --min-duration 1h \
  --out ~/.claude/skills/lex-advisor \
  --from-natural-language --intent "Long-form Lex Fridman interviews" --yes
```

User sees: confirmation summary ("videos ≥ 1h from 2024 onwards") → progress updates → postrun guide. NEVER sees the flags.

### Example 3 — explicit video IDs

User: `Use these specific videos: https://youtu.be/abc, https://youtu.be/def, https://youtu.be/ghi`

You:
1. Extract IDs: `abc`, `def`, `ghi`.
2. `echo -e "abc\ndef\nghi" > /tmp/youtube-advisor-ids.txt` (via Bash).
3. Run:
```
youtube-advisor bootstrap --channel "<infer from one of the videos via yt-dlp>" \
  --ids /tmp/youtube-advisor-ids.txt \
  --max 3 \
  --out ~/.claude/skills/<slug>-advisor \
  --from-natural-language --intent "<ask user>" --yes
```

Ask the user briefly: "These three videos come from <channel>. Use that as the channel handle? And: what should this advisor be good at? (One sentence.)"

### Example 4 — named creator, no URL

User: `Build me an advisor on Naval`

You:
1. Run `yt-dlp --flat-playlist -J "ytsearch3:Naval Ravikant channel" | jq -r '.entries[] | .channel'` to find candidates.
2. Ask: "I found `@navalr` — Naval Ravikant's channel. Use that one?"
3. On `y`, run bootstrap with `--channel @navalr`.

### Example 5 — update with hint

User: `обнови yc-advisor — также проверь нет ли новых видео за эту неделю`

You parse:
- advisor = `~/.claude/skills/yc-advisor`
- since override = 7 days ago.

You run:
```
youtube-advisor update --advisor ~/.claude/skills/yc-advisor --since 2026-05-29 --yes
```

Tell user the result via the postrun summary.

## Rules

- **Never show flags to the user.** Always build the command silently. Run it via Bash. Echo the result.
- **Never paste a `bash` command for the user to run.** You run everything; you echo only natural-language summaries and the post-run guide.
- **Match the user's language** in conversation (RU / EN / etc.) — but the CLI itself is English-only.
- **If Whisper time is estimated > 15 min**, ask before continuing ("Some videos lack captions and would need ~3h of local transcription. Skip them, or wait?").
- **Don't poll status on a timer.** Reuse the scriba pattern: emit on stage change + 25% audio progress.
- **End with the bootstrap_guide output unmodified.** No additional commentary.
- **Never use `/loop` or `ScheduleWakeup`** for progress monitoring — they cost prompt-cache misses.
- **Don't fabricate channel handles.** If the user names a creator without a URL, always disambiguate via search first.

## Installation prerequisites

If `youtube-advisor` is not installed (`which youtube-advisor` fails), do the install yourself — the user shouldn't have to leave chat. Clone into `~/.claude/skills/youtube-advisor` (or `~/tools/youtube-advisor` outside Claude Code), then run `bash scripts/install.sh` via the Bash tool. If `ffmpeg` is missing, `install.sh` will surface that — run `brew install ffmpeg` (macOS) or the platform equivalent and retry. Only ping the user if a step needs a sudo password or a non-trivial decision.

`ANTHROPIC_API_KEY` is **not required** when this skill is invoked from inside Claude Code — the agent in chat (you) drafts `SKILL.md` directly via the `.pending-llm-draft.json` hand-off (see Step 4). If the env var IS set, the CLI auto-uses the Anthropic SDK to pre-draft `SKILL.md` and the eval benchmark — that is purely an optimization, never a prerequisite. Pass `--no-llm` to force the hand-off path explicitly.

## When YouTube blocks the fetch

If you see HTTP 429 / "Sign in to confirm you're not a bot" / `_extract_player_response failed` errors, YT's anti-bot is rate-limiting this IP. yt-dlp invocations already pass `--remote-components ejs:github` (the EJS JS-challenge solver) and `--extractor-args youtube:formats=missing_pot` (PO-token tolerance) by default — but the EJS solver needs **`deno`** on PATH at runtime. If `which deno` fails, install via `brew install deno` (macOS) or `curl -fsSL https://deno.land/install.sh | sh` (Linux) and retry. If you're still blocked after that, two more fixes:

**Exit code 1 with "YouTube blocked requests":** `bootstrap` / `update` stopped at the first block: no further YouTube requests and no Whisper fallback for the blocked video. The advisor is partial but resumable; the blocked video (`failed:blocked` in `ingest.errors.log`) and the videos not tried yet are retried by the next update. Show the user the printed message, then run the printed `youtube-advisor update …` command after a wait or with cookies (below). After a blocked bootstrap, do not draft SKILL.md from `.pending-llm-draft.json` until an update finishes without a block.

**Option 1 — Use your logged-in browser cookies (recommended):**
Pass `--cookies-from-browser chrome` (or `firefox`, `safari`, `brave`, `edge`). yt-dlp reads cookies straight from the browser profile — no copy-paste. With cookies set, the captions pool stays at concurrency 8 (the captions API itself cannot read browser cookies, so it stays at 2 calls at once with a 0.5–2s jitter; yt-dlp uses the cookies).

**Option 2 — Wait it out:**
The defaults already slow requests (1–5s random sleep) and retry on 429 with exponential backoff. Without cookies the captions pool also drops to concurrency 2 + a 0.5–2s jitter per call. A retry after a few minutes usually clears.

You can also set `YOUTUBE_ADVISOR_COOKIES_BROWSER=chrome` once in your shell rc, or drop a Netscape `cookies.txt` at `~/.config/youtube-advisor/cookies.txt`, to default that browser/file for all advisors.
