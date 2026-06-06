import json
import pathlib

from youtube_advisor.postrun_guide import bootstrap_guide, update_guide, _source_counts_text, _short_slug

def _write_meta(advisor: pathlib.Path, **overrides):
    (advisor / "references").mkdir(parents=True, exist_ok=True)
    base = {
        "n_videos": 187,
        "channels": ["@ycombinator"],
        "date_range": "2024-01 → 2026-06",
        "corpus_language": "en",
        "answer_language": "ru",
        "quote_style": "translation-first",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
        "transcript_source_counts": {"captions": 184, "whisper-v3": 3},
        "latest_video_date": "2026-06-04",
    }
    base.update(overrides)
    (advisor / "references" / "corpus_meta.json").write_text(json.dumps(base))

def _write_eval(advisor: pathlib.Path, passed=8, total=8, delta_passed=None, date_iso="2026-06-05"):
    (advisor / "evals" / "results").mkdir(parents=True, exist_ok=True)
    data = {"passed": passed, "total": total, "results": []}
    if delta_passed is not None:
        data["delta_passed"] = delta_passed
    (advisor / "evals" / "results" / f"{date_iso}.json").write_text(json.dumps(data))

# ----- _short_slug -----

def test_short_slug_strips_advisor_suffix():
    assert _short_slug("yc-advisor") == "yc"
    assert _short_slug("naval-advisor") == "naval"

def test_short_slug_passthrough_when_no_suffix():
    assert _short_slug("yc") == "yc"
    assert _short_slug("my-skill") == "my-skill"

# ----- _source_counts_text -----

def test_source_counts_captions_and_whisper():
    out = _source_counts_text({"captions": 184, "whisper-v3": 3})
    assert "184 via captions" in out
    assert "3 via Whisper" in out

def test_source_counts_collapses_multiple_whisper_keys():
    out = _source_counts_text({"captions": 10, "whisper-v3": 3, "whisper-fast": 2})
    assert "5 via Whisper" in out

def test_source_counts_includes_failed():
    out = _source_counts_text({"captions": 5, "failed:no-captions": 2})
    assert "2 failed" in out

def test_source_counts_empty_returns_fallback():
    assert "transcripts" in _source_counts_text({})

# ----- bootstrap_guide -----

def test_bootstrap_guide_includes_key_sections(tmp_path):
    advisor = tmp_path / "yc-advisor"
    advisor.mkdir()
    _write_meta(advisor)
    _write_eval(advisor, passed=8, total=8)

    out = bootstrap_guide(advisor)
    assert "✓ Advisor ready:" in out
    assert "How to use" in out
    assert "Keeping it fresh" in out
    assert "Editing the advisor" in out
    assert "187 videos" in out
    assert "184 via captions" in out
    assert "3 via Whisper" in out
    assert "translation-first" in out
    assert "8/8 passing" in out
    # Slug derived from dir name
    assert "/yc-advisor" in out

def test_bootstrap_guide_strips_advisor_suffix_in_slash_command(tmp_path):
    advisor = tmp_path / "yc-advisor"
    advisor.mkdir()
    _write_meta(advisor)
    out = bootstrap_guide(advisor)
    # Should be /yc-advisor, NOT /yc-advisor-advisor
    assert "/yc-advisor" in out
    assert "/yc-advisor-advisor" not in out

def test_bootstrap_guide_handles_missing_evals(tmp_path):
    advisor = tmp_path / "yc-advisor"
    advisor.mkdir()
    _write_meta(advisor)
    out = bootstrap_guide(advisor)
    assert "Evals: not run" in out

# ----- update_guide -----

def test_update_guide_no_change_message(tmp_path):
    advisor = tmp_path / "yc-advisor"
    advisor.mkdir()
    out = update_guide(advisor, n_new=0, n_removed=0)
    assert "0 new, 0 removed" in out
    assert "unchanged" in out

def test_update_guide_with_new_videos(tmp_path):
    advisor = tmp_path / "yc-advisor"
    advisor.mkdir()
    _write_meta(advisor, n_videos=199)
    out = update_guide(advisor, n_new=12, n_removed=0)
    assert "+12 new" in out
    assert "0 removed" in out
    assert "199 videos (was 187)" in out
    assert "Latest: 2026-06-04" in out

def test_update_guide_with_eval_delta_positive(tmp_path):
    advisor = tmp_path / "yc-advisor"
    advisor.mkdir()
    _write_meta(advisor, n_videos=199)
    out = update_guide(advisor, n_new=12, n_removed=0,
                       eval_summary={"passed": 8, "total": 8, "delta_passed": 1})
    assert "8/8 passing (+1)" in out

def test_update_guide_with_eval_delta_negative(tmp_path):
    advisor = tmp_path / "yc-advisor"
    advisor.mkdir()
    _write_meta(advisor, n_videos=199)
    out = update_guide(advisor, n_new=12, n_removed=0,
                       eval_summary={"passed": 7, "total": 8, "delta_passed": -1})
    assert "regression!" in out
    assert "7/8 passing" in out

def test_update_guide_does_not_double_advisor_suffix(tmp_path):
    advisor = tmp_path / "yc-advisor"
    advisor.mkdir()
    _write_meta(advisor, n_videos=199)
    out = update_guide(advisor, n_new=1, n_removed=0)
    assert "/yc-advisor-advisor" not in out
    assert "/yc-advisor" in out

def test_update_guide_eval_delta_zero(tmp_path):
    advisor = tmp_path / "yc-advisor"
    advisor.mkdir()
    _write_meta(advisor, n_videos=199)
    out = update_guide(advisor, n_new=12, n_removed=0,
                       eval_summary={"passed": 8, "total": 8, "delta_passed": 0})
    assert "(no change)" in out
