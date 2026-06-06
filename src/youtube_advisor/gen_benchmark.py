from __future__ import annotations
import json
import pathlib
from jinja2 import Environment, FileSystemLoader

from .search import search
from ._llm import call_tool
from ._md import sample_transcripts
from ._paths import TEMPLATES_DIR


def _validate(advisor_dir: pathlib.Path, questions: list[dict]) -> list[dict]:
    """Keep only questions whose expected_video_ids intersect search()'s top-5."""
    valid = []
    for q in questions:
        try:
            hits = search(advisor_dir, q["q"], top_k=5, mode="keyword")
        except Exception:
            continue
        hit_ids = {h["video_id"] for h in hits}
        if hit_ids & set(q.get("expected_video_ids", [])):
            valid.append(q)
    return valid


def _sample_excerpts(advisor_dir: pathlib.Path, n: int = 12) -> list[tuple[str, str]]:
    """Back-compat wrapper: returns list[(video_id, body)]."""
    return sample_transcripts(advisor_dir, n, seed_salt=1, body_chars=2000,
                              with_video_id=True)


def generate(advisor_dir: pathlib.Path, target: int = 8) -> None:
    meta = json.loads((advisor_dir / "references" / "corpus_meta.json").read_text())
    excerpts = _sample_excerpts(advisor_dir)
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

    if not excerpts:
        # Empty corpus → write an empty benchmark
        (advisor_dir / "evals").mkdir(exist_ok=True)
        (advisor_dir / "evals" / "benchmark.json").write_text(
            env.get_template("benchmark.json.tmpl").render(
                generator_version="0.1.0",
                corpus_version=meta.get("corpus_version", ""),
                questions=[]))
        return

    schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "The question to ask the advisor."},
                        "lang": {"type": "string", "description": "ISO language code, e.g. 'en' or 'ru'."},
                        "expected_video_ids": {
                            "type": "array", "items": {"type": "string"},
                            "description": "Video IDs (file stems) that should contain the answer.",
                        },
                        "expected_keywords": {
                            "type": "array", "items": {"type": "string"},
                            "description": "Keywords expected to appear in retrieved snippets.",
                        },
                        "min_quotes": {"type": "integer", "minimum": 1, "default": 1},
                        "expected_quote_substrings": {
                            "type": "array", "items": {"type": "string"},
                            "description": "Verbatim substrings expected in any answer quote.",
                        },
                    },
                    "required": ["q", "lang", "expected_video_ids", "expected_keywords", "min_quotes"],
                },
            },
        },
        "required": ["questions"],
    }

    excerpt_block = "\n".join(f"### {vid}\n{body}" for vid, body in excerpts)
    prompt = f"""Generate {target + 4} candidate benchmark questions for a YouTube-channel advisor based on these excerpts.

Each question:
- Must be answerable from the excerpts (not generic).
- Must include `expected_video_ids` listing the source video IDs (the stem before each excerpt — e.g. `abc123`, not the full filename).
- Must include 2-4 `expected_keywords` likely to appear in retrieved snippets.
- `lang` should match the excerpt language ('en' for English, 'ru' for Russian, etc.).
- `min_quotes` defaults to 1.
- `expected_quote_substrings` is optional verbatim text expected in an ideal answer's quotes.

Excerpts:
{excerpt_block}
"""

    out = call_tool(prompt, tool_name="emit_questions", schema=schema, max_tokens=3000)
    valid = _validate(advisor_dir, out.get("questions", []))[:target]

    (advisor_dir / "evals").mkdir(exist_ok=True)
    (advisor_dir / "evals" / "benchmark.json").write_text(
        env.get_template("benchmark.json.tmpl").render(
            generator_version="0.1.0",
            corpus_version=meta.get("corpus_version", ""),
            questions=valid))

    # When the LLM produced candidates but none validated, write a sibling
    # `.benchmark_failed` marker so eval_runner can short-circuit
    # `delta_passed = None` instead of treating an empty benchmark as a
    # passing run (which would silently void the rollback contract).
    marker = advisor_dir / "evals" / ".benchmark_failed"
    if not valid and out.get("questions"):
        marker.write_text(
            "LLM produced candidates but none survived validation. "
            "Re-run with a larger corpus or inspect the LLM output."
        )
    elif marker.exists():
        marker.unlink()
