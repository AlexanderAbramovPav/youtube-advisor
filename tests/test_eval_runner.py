import json
import pathlib
from datetime import date, timedelta

import pytest

from youtube_advisor.eval_runner import run
from youtube_advisor.ingest.normalize import write_transcript
from youtube_advisor.build_index import build_index


def _setup_corpus(tmp_path):
    advisor = tmp_path / "yc-advisor"
    advisor.mkdir()
    (advisor / "transcripts").mkdir()
    write_transcript(
        advisor / "transcripts",
        {"video_id": "hire1", "title": "Hiring",
         "published_date": "2024-01-01", "channel": "@yc"},
        [{"start": i * 5, "duration": 5,
          "text": f"how to hire your first engineer carefully line {i}"}
         for i in range(5)],
        "captions", "en",
    )
    write_transcript(
        advisor / "transcripts",
        {"video_id": "pivot1", "title": "Pivots",
         "published_date": "2024-02-01", "channel": "@yc"},
        [{"start": i * 5, "duration": 5,
          "text": f"when to pivot your startup line {i}"}
         for i in range(5)],
        "captions", "en",
    )
    write_transcript(
        advisor / "transcripts",
        {"video_id": "paint1", "title": "Painting",
         "published_date": "2024-03-01", "channel": "@yc"},
        [{"start": i * 5, "duration": 5,
          "text": f"painting walls red line {i}"}
         for i in range(5)],
        "captions", "en",
    )
    (advisor / "references").mkdir()
    build_index(advisor / "transcripts", advisor / "references")
    (advisor / "evals").mkdir()
    return advisor


def _write_benchmark(advisor, questions):
    (advisor / "evals" / "benchmark.json").write_text(json.dumps({
        "version": 1,
        "corpus_version": "v1",
        "questions": questions,
    }))


def test_run_passes_when_video_and_keyword_match(tmp_path):
    advisor = _setup_corpus(tmp_path)
    _write_benchmark(advisor, [
        {"q": "how to hire engineer", "lang": "en",
         "expected_video_ids": ["hire1"],
         "expected_keywords": ["hire"], "min_quotes": 1},
    ])
    summary = run(advisor)
    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert summary["results"][0]["id_ok"] is True
    assert summary["results"][0]["kw_ok"] is True


def test_run_fails_when_video_missing(tmp_path):
    advisor = _setup_corpus(tmp_path)
    _write_benchmark(advisor, [
        {"q": "completely unrelated xyzzy", "lang": "en",
         "expected_video_ids": ["hire1"],
         "expected_keywords": ["xyzzy"], "min_quotes": 1},
    ])
    summary = run(advisor)
    assert summary["passed"] == 0
    assert summary["results"][0]["id_ok"] is False


def test_run_fails_when_keyword_missing(tmp_path):
    advisor = _setup_corpus(tmp_path)
    _write_benchmark(advisor, [
        {"q": "engineer", "lang": "en",
         "expected_video_ids": ["hire1"],
         "expected_keywords": ["unrelated"], "min_quotes": 1},
    ])
    summary = run(advisor)
    assert summary["results"][0]["id_ok"] is True
    assert summary["results"][0]["kw_ok"] is False
    assert summary["passed"] == 0


def test_run_passes_when_no_keywords_required(tmp_path):
    advisor = _setup_corpus(tmp_path)
    _write_benchmark(advisor, [
        {"q": "engineer", "lang": "en",
         "expected_video_ids": ["hire1"],
         "expected_keywords": [], "min_quotes": 1},
    ])
    summary = run(advisor)
    assert summary["passed"] == 1
    assert summary["results"][0]["kw_ok"] is True


def test_run_writes_results_json(tmp_path):
    advisor = _setup_corpus(tmp_path)
    _write_benchmark(advisor, [
        {"q": "hire", "lang": "en",
         "expected_video_ids": ["hire1"],
         "expected_keywords": ["hire"], "min_quotes": 1},
    ])
    run(advisor)
    today = date.today().isoformat()
    out = advisor / "evals" / "results" / f"{today}.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["total"] == 1
    assert "results" in data


def test_run_computes_delta_passed(tmp_path):
    advisor = _setup_corpus(tmp_path)
    _write_benchmark(advisor, [
        {"q": "hire", "lang": "en",
         "expected_video_ids": ["hire1"],
         "expected_keywords": ["hire"], "min_quotes": 1},
        {"q": "pivot", "lang": "en",
         "expected_video_ids": ["pivot1"],
         "expected_keywords": ["pivot"], "min_quotes": 1},
    ])
    out_dir = advisor / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    (out_dir / f"{yesterday}.json").write_text(
        json.dumps({"total": 2, "passed": 1, "results": []}))

    summary = run(advisor)
    assert summary["passed"] == 2
    assert summary["delta_passed"] == 1


def test_run_delta_passed_is_none_for_first_run(tmp_path):
    advisor = _setup_corpus(tmp_path)
    _write_benchmark(advisor, [
        {"q": "hire", "lang": "en",
         "expected_video_ids": ["hire1"],
         "expected_keywords": ["hire"], "min_quotes": 1},
    ])
    summary = run(advisor)
    assert summary["delta_passed"] is None


def test_run_handles_empty_benchmark(tmp_path):
    advisor = _setup_corpus(tmp_path)
    _write_benchmark(advisor, [])
    summary = run(advisor)
    assert summary["total"] == 0
    assert summary["passed"] == 0
    assert summary["results"] == []
