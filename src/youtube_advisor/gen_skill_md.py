from __future__ import annotations
import json
import pathlib
from jinja2 import Environment, FileSystemLoader

from ._llm import call_tool
from ._md import sample_transcripts
from ._naming import short_slug
from ._paths import TEMPLATES_DIR


def _sample_transcripts(advisor_dir: pathlib.Path, n: int = 8) -> list[str]:
    """Back-compat wrapper around _md.sample_transcripts."""
    return sample_transcripts(advisor_dir, n, seed_salt=0, body_chars=3500,
                              with_video_id=False)


def _build_prompt(channel_display_name: str, user_intent: str, meta: dict,
                  answer_language: str, quote_style: str, samples: list[str]) -> str:
    sample_block = "\n".join(f"### Sample {i+1}\n{s}" for i, s in enumerate(samples))
    return f"""You are designing a Claude Code 'advisor' skill scaffolded over YouTube transcripts from {channel_display_name}.

User intent: {user_intent}
Corpus: {meta['n_videos']} videos. Answer language: {answer_language}. Quote style: {quote_style}.

Sample transcript excerpts (each ~3500 chars):
---
{sample_block}
---

Produce SKILL.md metadata via the `emit_skill_metadata` tool. Anti-patterns must be specific to THIS channel's typical failure modes (not generic advice). Example queries should be real questions a user would ask AND that this corpus can actually answer well.
"""


def generate(advisor_dir: pathlib.Path, *, user_intent: str, answer_language: str,
             quote_style: str, channel_display_name: str) -> None:
    meta = json.loads((advisor_dir / "references" / "corpus_meta.json").read_text())
    samples = _sample_transcripts(advisor_dir)

    schema = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "One-paragraph SKILL.md frontmatter description that triggers the skill on relevant user queries.",
            },
            "purpose": {
                "type": "string",
                "description": "2-3 sentence purpose statement.",
            },
            "anti_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 6,
                "description": "Channel-specific failure modes to avoid.",
            },
            "example_queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 5,
                "maxItems": 7,
                "description": "Real example questions this advisor handles well.",
            },
        },
        "required": ["description", "purpose", "anti_patterns", "example_queries"],
    }

    prompt = _build_prompt(channel_display_name, user_intent, meta,
                           answer_language, quote_style, samples)
    drafted = call_tool(prompt, tool_name="emit_skill_metadata", schema=schema)

    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), keep_trailing_newline=True)
    ctx = {
        "channel_slug": short_slug(advisor_dir.name),
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
        "generated_description": drafted["description"],
        "generated_purpose": drafted["purpose"],
        "generated_anti_patterns": "\n".join(f"- {a}" for a in drafted["anti_patterns"]),
        "generated_example_queries": "\n".join(f"- {q}" for q in drafted["example_queries"]),
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
