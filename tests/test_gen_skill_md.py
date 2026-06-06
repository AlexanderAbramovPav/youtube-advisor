import json
import pathlib
import pytest
from unittest.mock import patch
from youtube_advisor.gen_skill_md import generate, _sample_transcripts, _build_prompt
from youtube_advisor.ingest.normalize import write_transcript


def _meta(vid, pub="2024-06-01"):
    return {"video_id": vid, "title": f"T{vid}", "published_date": pub, "channel": "@yc"}


def _seg(t, text="hello world"):
    return {"start": t, "duration": 5, "text": text}


def _setup_advisor(tmp_path, n_videos=8):
    advisor = tmp_path / "yc-advisor"
    advisor.mkdir()
    (advisor / "transcripts").mkdir()
    for i in range(n_videos):
        write_transcript(advisor / "transcripts", _meta(f"vid{i}"),
                         [_seg(j*5, f"sample text from video {i} chunk {j}") for j in range(5)],
                         "captions", "en")
    (advisor / "references").mkdir()
    (advisor / "references" / "corpus_meta.json").write_text(json.dumps({
        "n_videos": n_videos,
        "corpus_version": "2026-06-05/" + str(n_videos),
        "channels": ["@yc"],
        "date_range": "2024-06-01",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
        "corpus_language": "en",
        "last_bootstrap": "2026-06-05T15:00:00+00:00",
    }))
    return advisor


# ----- _sample_transcripts -----

def test_sample_transcripts_returns_8_when_available(tmp_path):
    advisor = _setup_advisor(tmp_path, n_videos=10)
    samples = _sample_transcripts(advisor)
    assert len(samples) == 8


def test_sample_transcripts_returns_all_when_corpus_small(tmp_path):
    advisor = _setup_advisor(tmp_path, n_videos=3)
    samples = _sample_transcripts(advisor)
    assert len(samples) == 3


def test_sample_transcripts_deterministic_with_same_seed(tmp_path):
    advisor = _setup_advisor(tmp_path, n_videos=10)
    s1 = _sample_transcripts(advisor)
    s2 = _sample_transcripts(advisor)
    assert s1 == s2  # same seed → same sample


def test_sample_transcripts_strips_frontmatter(tmp_path):
    advisor = _setup_advisor(tmp_path, n_videos=3)
    samples = _sample_transcripts(advisor)
    for s in samples:
        assert "video_id:" not in s  # frontmatter stripped
        assert "[00:00:00]" in s  # body preserved


def test_sample_transcripts_empty_when_no_transcripts(tmp_path):
    advisor = tmp_path / "empty-advisor"
    (advisor / "transcripts").mkdir(parents=True)
    samples = _sample_transcripts(advisor)
    assert samples == []


# ----- _build_prompt -----

def test_build_prompt_includes_intent_and_samples():
    prompt = _build_prompt("@yc", "advise on hiring", {"n_videos": 10},
                           "ru", "translation-first", ["sample1 text", "sample2 text"])
    assert "@yc" in prompt
    assert "advise on hiring" in prompt
    assert "10 videos" in prompt
    assert "Sample 1" in prompt and "sample1 text" in prompt


# ----- generate (LLM mocked) -----

def test_generate_writes_skill_md(tmp_path):
    advisor = _setup_advisor(tmp_path)
    fake_draft = {
        "description": "Advisor on YC startup advice.",
        "purpose": "Answers YC questions.",
        "anti_patterns": ["AP1", "AP2", "AP3"],
        "example_queries": ["EQ1", "EQ2", "EQ3", "EQ4", "EQ5"],
    }
    with patch("youtube_advisor.gen_skill_md.call_tool", return_value=fake_draft):
        generate(advisor, user_intent="YC advice", answer_language="ru",
                 quote_style="translation-first", channel_display_name="@yc")
    skill_md = (advisor / "SKILL.md").read_text()
    assert "name: yc-advisor" in skill_md
    assert "Advisor on YC startup advice." in skill_md
    assert "- AP1" in skill_md
    assert "- EQ1" in skill_md


def test_generate_writes_all_artifacts(tmp_path):
    advisor = _setup_advisor(tmp_path)
    fake_draft = {
        "description": "D", "purpose": "P",
        "anti_patterns": ["a", "b", "c"],
        "example_queries": ["q1", "q2", "q3", "q4", "q5"],
    }
    with patch("youtube_advisor.gen_skill_md.call_tool", return_value=fake_draft):
        generate(advisor, user_intent="x", answer_language="en",
                 quote_style="quote-only", channel_display_name="@yc")
    assert (advisor / "SKILL.md").exists()
    assert (advisor / "AGENTS.md").exists()
    assert (advisor / "README.md").exists()
    assert (advisor / "references" / "workflow.md").exists()
    assert (advisor / "assets" / "answer_template.md").exists()
    # AGENTS.md should NOT have frontmatter
    agents = (advisor / "AGENTS.md").read_text()
    assert not agents.startswith("---")


def test_generate_passes_correct_context_to_templates(tmp_path):
    advisor = _setup_advisor(tmp_path)
    fake_draft = {
        "description": "TestDesc", "purpose": "TestPurpose",
        "anti_patterns": ["antia", "antib", "antic"],
        "example_queries": ["q1", "q2", "q3", "q4", "q5"],
    }
    with patch("youtube_advisor.gen_skill_md.call_tool", return_value=fake_draft):
        generate(advisor, user_intent="i", answer_language="ru",
                 quote_style="translation-first", channel_display_name="@yc")
    skill = (advisor / "SKILL.md").read_text()
    assert "@yc advisor" in skill  # advisor_title
    assert "BAAI/bge-small-en-v1.5" in skill  # embedding_model
    assert "Answer in Russian" in skill  # answer_language_rule
    answer_tmpl = (advisor / "assets" / "answer_template.md").read_text()
    # translation-first branch produces verbatim italic + translated quote
    assert "translated_quote" not in answer_tmpl  # placeholder rendered/empty in context
