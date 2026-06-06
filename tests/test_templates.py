import json

from jinja2 import Environment, FileSystemLoader

from youtube_advisor._paths import TEMPLATES_DIR as _TEMPLATES_DIR
TEMPLATES_DIR = str(_TEMPLATES_DIR)


def _env():
    return Environment(loader=FileSystemLoader(TEMPLATES_DIR), keep_trailing_newline=True)


def _full_context():
    return {
        "channel_slug": "yc",
        "generated_description": "Answers YC startup questions with verbatim quotes.",
        "advisor_title": "@ycombinator advisor",
        "generated_purpose": "Local skill answering questions about Y Combinator videos.",
        "channels_list": "@ycombinator",
        "n_videos": 187,
        "date_range": "2022-01 → 2026-06",
        "corpus_version": "2026-06-05/187",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
        "quote_style": "translation-first",
        "answer_language_rule": "Answer in Russian; quote verbatim in English",
        "channel_display_name": "@ycombinator",
        "generated_anti_patterns": "- Don't paraphrase generic startup advice.\n- Don't attribute quotes without a source.",
        "generated_example_queries": "- How to hire your first engineer?\n- When to pivot?",
        "generator_version": "0.1.0",
        "repo_url": "https://github.com/AlexanderAbramovPav/youtube-advisor",
        "created_at": "2026-06-05",
        "corpus_language": "en",
        "answer_language": "ru",
    }


def test_skill_md_renders_without_placeholders():
    ctx = _full_context()
    out = _env().get_template("advisor.SKILL.md.tmpl").render(**ctx)
    assert "{{" not in out and "}}" not in out
    assert "name: yc-advisor" in out
    assert "@ycombinator" in out
    assert "Answer in Russian" in out


def test_agents_md_renders_without_frontmatter():
    out = _env().get_template("advisor.AGENTS.md.tmpl").render(**_full_context())
    assert not out.startswith("---")
    assert "{{" not in out


def test_readme_md_renders():
    out = _env().get_template("advisor.README.md.tmpl").render(**_full_context())
    assert "{{" not in out
    assert "/yc-advisor" in out
    assert "youtube-advisor" in out


def test_workflow_md_renders():
    out = _env().get_template("workflow.md.tmpl").render(**_full_context())
    assert "{{" not in out
    assert "scripts/search.py" in out


def test_answer_template_translation_first_branch():
    ctx = {
        **_full_context(),
        "translated_quote": "Найми инженеров",
        "verbatim_quote": "Hire engineers",
        "video_title": "Hiring",
        "published_date": "2024-01-01",
        "timestamp_hms": "00:01:23",
        "timestamp_link": "https://youtu.be/abc&t=83",
    }
    out = _env().get_template("answer_template.md.tmpl").render(**ctx)
    assert "Найми инженеров" in out
    assert '*"Hire engineers"*' in out


def test_answer_template_quote_only_branch():
    ctx = {
        **_full_context(),
        "quote_style": "quote-only",
        "verbatim_quote": "Hire engineers",
        "video_title": "Hiring",
        "published_date": "2024-01-01",
        "timestamp_hms": "00:01:23",
        "timestamp_link": "https://youtu.be/abc&t=83",
    }
    out = _env().get_template("answer_template.md.tmpl").render(**ctx)
    assert "Найми инженеров" not in out
    assert '"Hire engineers"' in out
    assert "*" not in out.split("\n")[1]  # no italics in quote-only mode


def test_benchmark_json_renders_valid_json():
    ctx = {
        **_full_context(),
        "questions": [{"q": "How to hire?", "expected_video_ids": ["a", "b"]}],
    }
    out = _env().get_template("benchmark.json.tmpl").render(**ctx)
    data = json.loads(out)
    assert data["version"] == 1
    assert data["questions"][0]["q"] == "How to hire?"


def test_search_py_renders():
    out = _env().get_template("search.py.tmpl").render(**_full_context())
    assert "from youtube_advisor.search import cli" in out
    assert "_lib" in out
