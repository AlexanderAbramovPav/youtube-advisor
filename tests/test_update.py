import json
import pathlib
import shutil
from unittest.mock import patch
import pytest
import yaml

from youtube_advisor.update import run, _video_id_from_filename, _load_state, _save_state, _filter_from_yaml
from youtube_advisor.ingest.normalize import write_transcript
from youtube_advisor.build_index import build_index


def _stub_video(vid, pub="2024-06-01", title=None):
    return {
        "video_id": vid, "title": title or f"v{vid}", "published_date": pub,
        "length_seconds": 600, "views": 100, "description": "",
        "channel": "@yc", "channel_id": "UCx", "playlist_ids": set(),
    }


def _setup_advisor(tmp_path, existing_ids=("a", "b")):
    """Pre-built advisor with a tiny corpus, filter, and meta."""
    out = tmp_path / "yc-advisor"
    out.mkdir()
    (out / "transcripts").mkdir()
    for vid in existing_ids:
        write_transcript(out / "transcripts",
                         {"video_id": vid, "title": f"t{vid}", "published_date": "2024-06-01", "channel": "@yc"},
                         [{"start": 0, "duration": 1, "text": f"hello {vid}"}], "captions", "en")
    (out / "references").mkdir()
    (out / "references" / "selection_filter.yaml").write_text(yaml.safe_dump({
        "version": 1, "channels": ["@yc"], "since": None, "until": None,
        "max": 500, "title_include": None, "title_exclude": r"#?shorts\b",
        "playlists": [], "extra_ids": [], "created_at": "2026-06-01T00:00:00+00:00",
    }))
    (out / "references" / "corpus_meta.json").write_text(json.dumps({
        "n_videos": len(existing_ids),
        "corpus_version": f"2024-06-01/{len(existing_ids)}",
        "channels": ["@yc"],
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
        "transcript_source_counts": {"captions": len(existing_ids)},
        "answer_language": "en", "quote_style": "quote-only",
    }))
    return out


# ----- helpers -----

def test_video_id_from_filename():
    assert _video_id_from_filename(pathlib.Path("2024-06-01-abc.md")) == "abc"
    assert _video_id_from_filename(pathlib.Path("2024-06-01-with-dashes.md")) == "with-dashes"

def test_load_and_save_state(tmp_path):
    out = _setup_advisor(tmp_path)
    filt, meta = _load_state(out)
    assert filt["channels"] == ["@yc"]
    assert meta["n_videos"] == 2
    meta["n_videos"] = 5
    _save_state(out, filt, meta)
    _, m2 = _load_state(out)
    assert m2["n_videos"] == 5

def test_filter_from_yaml_parses_dates():
    from datetime import date as _date
    cfg = _filter_from_yaml({"since": "2022-01-01", "until": "2025-12-31",
                              "max": 50, "title_include": "a", "extra_ids": ["x"]})
    assert cfg.since == _date(2022, 1, 1)
    assert cfg.until == _date(2025, 12, 31)
    assert cfg.max == 50
    assert cfg.manual_ids == {"x"}


# ----- run() -----

def test_run_no_op_when_nothing_changed(tmp_path):
    out = _setup_advisor(tmp_path, existing_ids=("a", "b"))
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a"), _stub_video("b")]):
        result = run(advisor_dir=str(out), yes=True)
    assert "0 new" in result["guide"]
    assert "unchanged" in result["guide"]
    assert result["evals"] is None

def test_run_ingests_new_videos(tmp_path):
    out = _setup_advisor(tmp_path, existing_ids=("a", "b"))
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video(v) for v in ("a", "b", "c", "d")]), \
         patch("youtube_advisor.update.run_ingest",
               return_value={"ingested": 2, "skipped": 0, "failed": 0, "by_source": {"captions": 2}}):
        # Simulate run_ingest writing the new transcripts (necessary because we mock it)
        def write_after(*a, **kw):
            for vid in ("c", "d"):
                write_transcript(out / "transcripts",
                                 {"video_id": vid, "title": "t", "published_date": "2024-06-01", "channel": "@yc"},
                                 [{"start": 0, "duration": 1, "text": "hi"}], "captions", "en")
            return {"ingested": 2, "skipped": 0, "failed": 0, "by_source": {"captions": 2}}
        with patch("youtube_advisor.update.run_ingest", side_effect=write_after):
            result = run(advisor_dir=str(out), yes=True)
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["n_videos"] == 4
    assert "+2" in result["guide"]

def test_run_reindex_only_skips_resolve(tmp_path):
    out = _setup_advisor(tmp_path)
    # Pre-build initial indices
    build_index(out / "transcripts", out / "references")
    with patch("youtube_advisor.update.resolve_channel",
               side_effect=AssertionError("should not be called")), \
         patch("youtube_advisor.update.build_embeddings") as mock_embed:
        result = run(advisor_dir=str(out), reindex_only=True, yes=True)
    mock_embed.assert_called_once()
    assert "unchanged" in result["guide"]

def test_run_cli_override_since(tmp_path):
    from datetime import date as _date
    out = _setup_advisor(tmp_path)
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a"), _stub_video("b")]):
        run(advisor_dir=str(out), since=_date(2026, 1, 1), yes=True)
    filt = yaml.safe_load((out / "references" / "selection_filter.yaml").read_text())
    assert filt["since"] == "2026-01-01"

def test_run_cli_override_max(tmp_path):
    out = _setup_advisor(tmp_path)
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a"), _stub_video("b")]):
        run(advisor_dir=str(out), max_n=50, yes=True)
    filt = yaml.safe_load((out / "references" / "selection_filter.yaml").read_text())
    assert filt["max"] == 50

def test_run_prune_removes_vanished_videos(tmp_path):
    out = _setup_advisor(tmp_path, existing_ids=("a", "b", "c"))
    # Channel now only has 'a'; b and c should be pruned
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a")]):
        result = run(advisor_dir=str(out), prune=True, yes=True)
    remaining = sorted(p.stem for p in (out / "transcripts").glob("*.md"))
    assert "2024-06-01-a" in remaining
    assert not any("-b" in p for p in remaining)
    assert not any("-c" in p for p in remaining)
    assert "2 removed" in result["guide"]

def test_run_does_not_prune_when_flag_off(tmp_path):
    out = _setup_advisor(tmp_path, existing_ids=("a", "b"))
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a")]):
        run(advisor_dir=str(out), prune=False, yes=True)
    # b is still on disk even though not on channel
    assert (out / "transcripts" / "2024-06-01-b.md").exists()

def test_run_refreshes_latest_video_date(tmp_path):
    out = _setup_advisor(tmp_path, existing_ids=("a", "b"))
    # Pre-existing meta has no latest_video_date; existing transcripts are 2024-06-01.
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a"), _stub_video("b"), _stub_video("c", pub="2026-05-30")]):
        def write_after(*a, **kw):
            write_transcript(out / "transcripts",
                             {"video_id": "c", "title": "tc", "published_date": "2026-05-30", "channel": "@yc"},
                             [{"start": 0, "duration": 1, "text": "hi c"}], "captions", "en")
            return {"ingested": 1, "skipped": 0, "failed": 0, "by_source": {"captions": 1}}
        with patch("youtube_advisor.update.run_ingest", side_effect=write_after):
            run(advisor_dir=str(out), yes=True)
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["latest_video_date"] == "2026-05-30"


def test_run_evals_returns_summary(tmp_path):
    out = _setup_advisor(tmp_path)
    fake_eval = {"total": 2, "passed": 2, "delta_passed": 0, "results": []}
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a"), _stub_video("b")]), \
         patch("youtube_advisor.update.run_evals", return_value=fake_eval):
        # Explicit no_llm=False so the eval branch isn't auto-skipped when
        # ANTHROPIC_API_KEY is unset in the test env.
        result = run(advisor_dir=str(out), run_evals_=True, yes=True, no_llm=False)
    assert result["evals"] == fake_eval


def test_run_no_op_skips_evals_when_no_llm(tmp_path):
    """Bug #5: in the early-return path (no new videos, no missing) we must
    NOT call run_evals when LLM is disabled — evals call the LLM."""
    out = _setup_advisor(tmp_path, existing_ids=("a", "b"))
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a"), _stub_video("b")]), \
         patch("youtube_advisor.update.run_evals") as mock_evals:
        result = run(advisor_dir=str(out), run_evals_=True, no_llm=True, yes=True)
    mock_evals.assert_not_called()
    assert result["evals"] is None


def test_update_calls_install_runtime(tmp_path):
    """Bug #12: update.run must refresh the vendored _lib + scripts wrappers
    so `pip install -U youtube-advisor` propagates to existing advisors."""
    out = _setup_advisor(tmp_path)
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a"), _stub_video("b")]), \
         patch("youtube_advisor.bootstrap._install_runtime") as mock_install:
        run(advisor_dir=str(out), yes=True)
    mock_install.assert_called_once()


def test_run_normalises_datetime_since_to_date_isoformat(tmp_path):
    """Bug #2: click.DateTime hands a datetime; we must persist a YYYY-MM-DD
    (not 'YYYY-MM-DDT00:00:00') so _filter_from_yaml's date.fromisoformat
    works on every supported Python."""
    from datetime import datetime as _dt
    out = _setup_advisor(tmp_path)
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a"), _stub_video("b")]):
        run(advisor_dir=str(out), since=_dt(2026, 1, 1, 12, 30), yes=True)
    filt = yaml.safe_load((out / "references" / "selection_filter.yaml").read_text())
    assert filt["since"] == "2026-01-01"
