import json
import pathlib
import pytest
from unittest.mock import patch
from youtube_advisor.gen_benchmark import generate, _validate, _sample_excerpts
from youtube_advisor.ingest.normalize import write_transcript
from youtube_advisor.build_index import build_index


def _setup_corpus(tmp_path, model_patch=None):
    """Create a tiny advisor with build_index + embeddings ready."""
    advisor = tmp_path / "yc-advisor"
    advisor.mkdir()
    (advisor / "transcripts").mkdir()
    write_transcript(advisor / "transcripts",
        {"video_id": "abc123", "title": "How to hire", "published_date": "2024-01-01", "channel": "@yc"},
        [{"start": i*5, "duration": 5, "text": f"how to hire your first engineer carefully line {i}"} for i in range(5)],
        "captions", "en")
    write_transcript(advisor / "transcripts",
        {"video_id": "def456", "title": "Pivots", "published_date": "2024-02-01", "channel": "@yc"},
        [{"start": i*5, "duration": 5, "text": f"when to pivot your startup line {i}"} for i in range(5)],
        "captions", "en")
    write_transcript(advisor / "transcripts",
        {"video_id": "ghi789", "title": "Painting", "published_date": "2024-03-01", "channel": "@yc"},
        [{"start": i*5, "duration": 5, "text": f"painting walls red line {i}"} for i in range(5)],
        "captions", "en")
    (advisor / "references").mkdir()
    (advisor / "references" / "corpus_meta.json").write_text(json.dumps({
        "n_videos": 3, "corpus_version": "2026-06-05/3",
        "embedding_model": "BAAI/bge-small-en-v1.5", "embedding_dim": 384,
    }))
    build_index(advisor / "transcripts", advisor / "references")
    return advisor


# ----- _sample_excerpts -----

def test_sample_excerpts_returns_video_id_and_body(tmp_path):
    advisor = _setup_corpus(tmp_path)
    excerpts = _sample_excerpts(advisor, n=2)
    assert len(excerpts) == 2
    for vid, body in excerpts:
        assert vid in {"abc123", "def456", "ghi789"}
        assert "line" in body  # body content present

def test_sample_excerpts_handles_empty_corpus(tmp_path):
    advisor = tmp_path / "empty"
    (advisor / "transcripts").mkdir(parents=True)
    assert _sample_excerpts(advisor) == []

def test_sample_excerpts_deterministic(tmp_path):
    advisor = _setup_corpus(tmp_path)
    s1 = _sample_excerpts(advisor)
    s2 = _sample_excerpts(advisor)
    assert s1 == s2

# ----- _validate -----

def test_validate_keeps_questions_with_matching_video_ids(tmp_path):
    advisor = _setup_corpus(tmp_path)
    questions = [
        {"q": "how to hire engineer", "lang": "en",
         "expected_video_ids": ["abc123"], "expected_keywords": ["hire"], "min_quotes": 1},
        {"q": "completely unrelated xyzzy", "lang": "en",
         "expected_video_ids": ["abc123"], "expected_keywords": ["xyzzy"], "min_quotes": 1},
    ]
    valid = _validate(advisor, questions)
    assert len(valid) == 1
    assert valid[0]["q"] == "how to hire engineer"

# ----- generate (LLM mocked) -----

def test_generate_writes_benchmark_json(tmp_path):
    advisor = _setup_corpus(tmp_path)
    fake = {"questions": [
        {"q": "how to hire?", "lang": "en", "expected_video_ids": ["abc123"],
         "expected_keywords": ["hire"], "min_quotes": 1, "expected_quote_substrings": []},
        {"q": "when to pivot?", "lang": "en", "expected_video_ids": ["def456"],
         "expected_keywords": ["pivot"], "min_quotes": 1, "expected_quote_substrings": []},
    ]}
    with patch("youtube_advisor.gen_benchmark.call_tool", return_value=fake):
        generate(advisor)
    data = json.loads((advisor / "evals" / "benchmark.json").read_text())
    assert data["version"] == 1
    assert data["corpus_version"] == "2026-06-05/3"
    assert len(data["questions"]) == 2

def test_generate_drops_invalid_questions(tmp_path):
    advisor = _setup_corpus(tmp_path)
    fake = {"questions": [
        {"q": "how to hire?", "lang": "en", "expected_video_ids": ["abc123"],
         "expected_keywords": ["hire"], "min_quotes": 1, "expected_quote_substrings": []},
        {"q": "xyzzy nonexistent", "lang": "en", "expected_video_ids": ["abc123"],
         "expected_keywords": ["xyzzy"], "min_quotes": 1, "expected_quote_substrings": []},
    ]}
    with patch("youtube_advisor.gen_benchmark.call_tool", return_value=fake):
        generate(advisor)
    data = json.loads((advisor / "evals" / "benchmark.json").read_text())
    assert len(data["questions"]) == 1
    assert "hire" in data["questions"][0]["q"]

def test_generate_truncates_to_target(tmp_path):
    advisor = _setup_corpus(tmp_path)
    # All questions valid, but more than target
    fake = {"questions": [
        {"q": f"hire engineers q{i}", "lang": "en",
         "expected_video_ids": ["abc123"], "expected_keywords": ["hire"],
         "min_quotes": 1, "expected_quote_substrings": []}
        for i in range(15)
    ]}
    with patch("youtube_advisor.gen_benchmark.call_tool", return_value=fake):
        generate(advisor, target=3)
    data = json.loads((advisor / "evals" / "benchmark.json").read_text())
    assert len(data["questions"]) <= 3

def test_generate_writes_failed_marker_when_all_dropped(tmp_path):
    """Bug #10: when the LLM produced candidates but _validate dropped them
    all, write a `.benchmark_failed` marker so eval_runner reports
    delta_passed=None instead of treating empty benchmark as "passed"."""
    advisor = _setup_corpus(tmp_path)
    fake = {"questions": [
        {"q": "xyzzy nonexistent", "lang": "en", "expected_video_ids": ["abc123"],
         "expected_keywords": ["xyzzy"], "min_quotes": 1, "expected_quote_substrings": []},
    ]}
    with patch("youtube_advisor.gen_benchmark.call_tool", return_value=fake):
        generate(advisor)
    marker = advisor / "evals" / ".benchmark_failed"
    assert marker.exists()


def test_generate_does_not_create_marker_when_at_least_one_valid(tmp_path):
    advisor = _setup_corpus(tmp_path)
    fake = {"questions": [
        {"q": "how to hire?", "lang": "en", "expected_video_ids": ["abc123"],
         "expected_keywords": ["hire"], "min_quotes": 1, "expected_quote_substrings": []},
    ]}
    with patch("youtube_advisor.gen_benchmark.call_tool", return_value=fake):
        generate(advisor)
    assert not (advisor / "evals" / ".benchmark_failed").exists()


def test_eval_runner_short_circuits_on_failed_marker(tmp_path):
    """When .benchmark_failed is present, eval_runner returns delta_passed=None."""
    from youtube_advisor.eval_runner import run as run_evals
    advisor = _setup_corpus(tmp_path)
    (advisor / "evals").mkdir(exist_ok=True)
    (advisor / "evals" / "benchmark.json").write_text(
        '{"version": 1, "corpus_version": "v0", "questions": []}'
    )
    (advisor / "evals" / ".benchmark_failed").write_text("none survived")
    summary = run_evals(advisor)
    assert summary["delta_passed"] is None
    assert summary.get("benchmark_failed") is True


def test_generate_empty_corpus_writes_empty_benchmark(tmp_path):
    advisor = tmp_path / "empty-advisor"
    (advisor / "transcripts").mkdir(parents=True)
    (advisor / "references").mkdir()
    (advisor / "references" / "corpus_meta.json").write_text(
        '{"corpus_version": "v0", "n_videos": 0}')
    # No need to mock LLM — generate short-circuits when no excerpts
    generate(advisor)
    data = json.loads((advisor / "evals" / "benchmark.json").read_text())
    assert data["questions"] == []
