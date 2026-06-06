# youtube-advisor — build a Claude Code advisor from any YouTube channel

> AGENTS.md is a tool-agnostic pointer for non-Claude-Code AI agents (Codex CLI, Cursor, Continue.dev, Aider, Goose, …). Setup is the only thing that differs per tool; the workflow, rules, and worked examples are canonical in [SKILL.md](./SKILL.md).

## Your tool → how to wire

| Tool                          | Setup                                                                                  | Invoke                                              |
| ----------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Claude Code**               | `git clone … ~/.claude/skills/youtube-advisor`                                         | `/youtube-advisor` then paste channel + intent      |
| **OpenAI Codex CLI**          | clone anywhere, place `AGENTS.md` at `~/.codex/AGENTS.md` (or project root)            | "build me an advisor on @ycombinator"               |
| **Cursor**                    | clone anywhere, copy `AGENTS.md` → `.cursor/rules/youtube-advisor.md`                  | mention `@youtube-advisor` or natural language       |
| **Continue.dev**              | clone anywhere, register as custom slash command in `~/.continue/config.yaml`          | `/youtube-advisor <channel>`                        |
| **Aider**                     | clone anywhere, `aider --read <install-dir>/AGENTS.md`                                        | natural language in chat                             |
| **Goose** (Block)             | clone anywhere, add as a shell-command extension                                       | mention youtube-advisor                              |
| **No AI at all**              | clone anywhere                                                                          | `youtube-advisor bootstrap --channel @yc --out ...` (see CLI in README) |

The bash CLI works in any environment with or without an AI driving it.

> See [SKILL.md](./SKILL.md) for the full workflow, two-action user contract, worked examples, anti-patterns, and installation prerequisites.
