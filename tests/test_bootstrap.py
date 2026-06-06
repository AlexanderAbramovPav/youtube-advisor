import json
import pathlib
from unittest.mock import patch, MagicMock
import pytest

from youtube_advisor.bootstrap import run, _resolve_quote_style, _detect_corpus_language


def _stub_video(vid, pub="2024-06-01"):
    return {
        "video_id": vid, "title": f"how to hire {vid}", "published_date": pub,
        "length_seconds": 600, "views": 100, "description": "",
        "channel": "@yc", "channel_id": "UCx", "playlist_ids": set(),
    }


@pytest.fixture
def mock_pipeline():
    """Patch every downstream module so bootstrap can run without network/LLM."""
    with patch("youtube_advisor.bootstrap.resolve_channel",
               return_value=[_stub_video("a"), _stub_video("b"), _stub_video("c")]), \
         patch("youtube_advisor.bootstrap.run_ingest",
               return_value={"ingested": 3, "skipped": 0, "failed": 0, "by_source": {"captions": 3}}), \
         patch("youtube_advisor.bootstrap.build_index", return_value=pathlib.Path("/tmp/idx")), \
         patch("youtube_advisor.bootstrap.build_embeddings", return_value=pathlib.Path("/tmp/emb")), \
         patch("youtube_advisor.bootstrap.gen_skill") as _gen_skill, \
         patch("youtube_advisor.bootstrap.gen_bench") as _gen_bench, \
         patch("youtube_advisor.bootstrap.run_evals") as _run_evals, \
         patch("youtube_advisor.bootstrap._detect_corpus_language", return_value="en"), \
         patch("youtube_advisor.bootstrap.bootstrap_guide", return_value="GUIDE\n"):
        yield {"gen_skill": _gen_skill, "gen_bench": _gen_bench, "run_evals": _run_evals}


def _base_kwargs(out_dir, **overrides):
    base = {
        "channels": ("@yc",),
        "since": None, "until": None, "max_n": 500,
        "playlists": (),
        "title_include": None, "title_exclude": r"#?shorts\b",
        "ids": None, "out_dir": str(out_dir),
        "whisper_skip_if_no_captions": False, "multilingual": False,
        "vendor": False, "no_evals": False,
        "from_natural_language": True,  # so we skip the interactive confirm
        "intent": "test", "answer_language": "en", "quote_style": "auto",
        "yes": True,
        # Existing tests assert the LLM path is exercised; force it explicitly
        # so test outcomes don't depend on ANTHROPIC_API_KEY env state.
        "no_llm": False,
    }
    base.update(overrides)
    return base


# ----- helpers -----

def test_resolve_quote_style_explicit():
    assert _resolve_quote_style("quote-only", "en", "en") == "quote-only"
    assert _resolve_quote_style("translation-first", "en", "ru") == "translation-first"

def test_resolve_quote_style_auto_with_explicit_lang():
    # languages match -> quote-only
    assert _resolve_quote_style("auto", "en", "en") == "quote-only"
    # languages differ -> translation-first
    assert _resolve_quote_style("auto", "ru", "en") == "translation-first"

def test_resolve_quote_style_auto_with_auto_lang():
    assert _resolve_quote_style("auto", "auto", "en") == "translation-first"

def test_detect_corpus_language_default_en(tmp_path):
    (tmp_path).mkdir(exist_ok=True)
    assert _detect_corpus_language(tmp_path) == "en"

def test_detect_corpus_language_from_frontmatter(tmp_path):
    md = tmp_path / "2024-01-01-a.md"
    md.write_text("---\nvideo_id: a\ntranscript_language: ru\n---\n\nbody\n")
    assert _detect_corpus_language(tmp_path) == "ru"


# ----- bootstrap.run -----

def test_run_creates_advisor_dir_and_meta(tmp_path, mock_pipeline):
    out = tmp_path / "yc-advisor"
    result = run(**_base_kwargs(out))
    assert out.exists()
    assert (out / "transcripts").exists()
    assert (out / "references" / "corpus_meta.json").exists()
    assert (out / "references" / "selection_filter.yaml").exists()
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["n_videos"] == 3
    assert meta["channels"] == ["@yc"]
    assert meta["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert "last_bootstrap" in meta

def test_run_calls_gen_skill_and_evals(tmp_path, mock_pipeline):
    out = tmp_path / "yc-advisor"
    run(**_base_kwargs(out))
    mock_pipeline["gen_skill"].assert_called_once()
    mock_pipeline["gen_bench"].assert_called_once()
    mock_pipeline["run_evals"].assert_called_once()

def test_run_skips_evals_when_no_evals_flag(tmp_path, mock_pipeline):
    out = tmp_path / "yc-advisor"
    run(**_base_kwargs(out, no_evals=True))
    mock_pipeline["gen_skill"].assert_called_once()
    mock_pipeline["gen_bench"].assert_not_called()
    mock_pipeline["run_evals"].assert_not_called()

def test_run_uses_multilingual_model_when_flag_set(tmp_path, mock_pipeline):
    out = tmp_path / "yc-advisor"
    run(**_base_kwargs(out, multilingual=True))
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["embedding_model"] == "BAAI/bge-m3"
    assert meta["embedding_dim"] == 1024

def test_run_writes_selection_filter_snapshot(tmp_path, mock_pipeline):
    import yaml
    out = tmp_path / "yc-advisor"
    run(**_base_kwargs(out, title_include="hire"))
    snap = yaml.safe_load((out / "references" / "selection_filter.yaml").read_text())
    assert snap["channels"] == ["@yc"]
    assert snap["title_include"] == "hire"
    assert snap["version"] == 1

def test_run_zero_videos_raises(tmp_path, mock_pipeline):
    with patch("youtube_advisor.bootstrap.resolve_channel", return_value=[]):
        with pytest.raises(Exception):  # click.UsageError or similar
            run(**_base_kwargs(tmp_path / "yc-advisor"))

def test_run_installs_search_wrapper(tmp_path, mock_pipeline):
    out = tmp_path / "yc-advisor"
    run(**_base_kwargs(out))
    wrapper = out / "scripts" / "search.py"
    assert wrapper.exists()
    assert wrapper.stat().st_mode & 0o111  # executable


def test_run_installs_build_index_wrapper(tmp_path, mock_pipeline):
    out = tmp_path / "yc-advisor"
    run(**_base_kwargs(out))
    wrapper = out / "scripts" / "build_index.py"
    assert wrapper.exists()
    assert wrapper.stat().st_mode & 0o111
    body = wrapper.read_text()
    assert "build_index" in body
    assert "build_embeddings" in body

def test_run_returns_guide_meta_summary(tmp_path, mock_pipeline):
    out = tmp_path / "yc-advisor"
    result = run(**_base_kwargs(out))
    assert "guide" in result and "meta" in result and "summary" in result
    assert result["guide"] == "GUIDE\n"
    assert result["summary"]["ingested"] == 3


def test_partial_state_raises_usage_error(tmp_path):
    """Bug #6: an output dir that exists and is non-empty but is missing
    references/selection_filter.yaml must raise a clear UsageError instead of
    delegating to update.run (which would crash opaquely on _load_state)."""
    import click as _click
    out = tmp_path / "partial"
    out.mkdir()
    (out / "transcripts").mkdir()
    (out / "transcripts" / "2024-06-01-a.md").write_text("---\nvideo_id: a\n---\n")
    # No references/selection_filter.yaml
    with pytest.raises(_click.UsageError, match="partially populated"):
        run(**_base_kwargs(out, yes=True))


def test_bootstrap_derives_playlist_ids_for_scoping(tmp_path, mock_pipeline):
    """Bug #13: --playlist URLs must scope the filter to videos that belong
    to those playlists, not just add them as candidates."""
    from youtube_advisor.update import _playlist_ids_from_urls
    ids = _playlist_ids_from_urls(["https://www.youtube.com/playlist?list=PLwanted"])
    assert ids == {"PLwanted"}
    # And the bootstrap derivation should propagate into FilterConfig — assert
    # via apply_filters: candidates not in PLwanted get dropped.
    from youtube_advisor.filters import FilterConfig, apply_filters
    a = _stub_video("a"); a["playlist_ids"] = {"PLwanted"}
    b = _stub_video("b"); b["playlist_ids"] = {"PLother"}
    kept = apply_filters([a, b], FilterConfig(playlist_ids=ids))
    assert [v["video_id"] for v in kept] == ["a"]

def test_run_existing_dir_switches_to_update(tmp_path, mock_pipeline):
    # Non-empty output dir without an advisor state → bootstrap delegates to
    # update, which then fails to load the missing selection_filter.yaml.
    out = tmp_path / "yc-advisor"
    out.mkdir()
    (out / "marker").write_text("x")  # non-empty
    with pytest.raises(Exception):
        run(**_base_kwargs(out))


# ----- --no-llm mode -----

@pytest.fixture
def mock_no_llm_pipeline():
    """Patch only the network/ingest side (NOT gen_skill / gen_bench / run_evals)
    so we can assert those LLM-touching functions are NOT called in --no-llm mode."""
    # Write a stub transcript so sample_transcripts() returns something useful.
    with patch("youtube_advisor.bootstrap.resolve_channel",
               return_value=[_stub_video("a"), _stub_video("b"), _stub_video("c")]), \
         patch("youtube_advisor.bootstrap.run_ingest") as _run_ingest, \
         patch("youtube_advisor.bootstrap.build_index", return_value=pathlib.Path("/tmp/idx")), \
         patch("youtube_advisor.bootstrap.build_embeddings", return_value=pathlib.Path("/tmp/emb")), \
         patch("youtube_advisor.bootstrap.gen_skill") as _gen_skill, \
         patch("youtube_advisor.bootstrap.gen_bench") as _gen_bench, \
         patch("youtube_advisor.bootstrap.run_evals") as _run_evals, \
         patch("youtube_advisor.bootstrap._detect_corpus_language", return_value="en"):

        def _ingest_side_effect(selected, transcripts_dir, **_kw):
            # Write a couple of fake transcript files so sample_transcripts works.
            transcripts_dir.mkdir(parents=True, exist_ok=True)
            for v in selected[:2]:
                p = transcripts_dir / f"2024-06-01-{v['video_id']}.md"
                p.write_text(
                    "---\n"
                    f"video_id: {v['video_id']}\n"
                    f"title: {v['title']}\n"
                    "video_url: https://youtu.be/x\n"
                    "published_date: 2024-06-01\n"
                    "transcript_language: en\n"
                    "---\n\n"
                    "[00:00:10] First, hire someone who can ship.\n"
                    "[00:00:20] Then worry about culture later.\n"
                )
            return {"ingested": 2, "skipped": 0, "failed": 0, "by_source": {"captions": 2}}

        _run_ingest.side_effect = _ingest_side_effect
        yield {"gen_skill": _gen_skill, "gen_bench": _gen_bench, "run_evals": _run_evals}


def test_bootstrap_no_llm_mode_writes_stub_and_marker(tmp_path, mock_no_llm_pipeline):
    out = tmp_path / "yc-advisor"
    kw = _base_kwargs(out, no_llm=True)
    result = run(**kw)

    # No LLM function should have been called.
    mock_no_llm_pipeline["gen_skill"].assert_not_called()
    mock_no_llm_pipeline["gen_bench"].assert_not_called()
    mock_no_llm_pipeline["run_evals"].assert_not_called()

    # Stub SKILL.md exists.
    skill = out / "SKILL.md"
    assert skill.exists()
    body = skill.read_text()
    assert "yc-advisor" in body or "advisor" in body.lower()

    # AGENTS.md, README.md also exist (rendered from templates).
    assert (out / "AGENTS.md").exists()
    assert (out / "README.md").exists()
    assert (out / "references" / "workflow.md").exists()
    assert (out / "assets" / "answer_template.md").exists()

    # Hand-off file exists with the required keys.
    marker = out / ".pending-llm-draft.json"
    assert marker.exists()
    payload = json.loads(marker.read_text())
    for key in ("task", "samples", "schema", "output_paths",
                "corpus_meta", "intent", "answer_language", "quote_style"):
        assert key in payload, f"missing key: {key}"
    assert isinstance(payload["samples"], list)
    assert len(payload["samples"]) >= 1
    assert all("video_id" in s and "excerpt" in s for s in payload["samples"])

    # Guide text mentions the hand-off.
    assert ".pending-llm-draft.json" in result["guide"]

    # No evals/benchmark.json in --no-llm mode.
    assert not (out / "evals" / "benchmark.json").exists()


def test_bootstrap_auto_detects_missing_api_key(tmp_path, mock_no_llm_pipeline, monkeypatch):
    """When ANTHROPIC_API_KEY is unset and no_llm is None, behave as --no-llm."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "yc-advisor"
    # no_llm=None → auto-detect from env.
    kw = _base_kwargs(out, no_llm=None)
    run(**kw)

    mock_no_llm_pipeline["gen_skill"].assert_not_called()
    mock_no_llm_pipeline["gen_bench"].assert_not_called()
    mock_no_llm_pipeline["run_evals"].assert_not_called()
    assert (out / ".pending-llm-draft.json").exists()


def test_bootstrap_auto_detects_present_api_key(tmp_path, mock_no_llm_pipeline, monkeypatch):
    """When ANTHROPIC_API_KEY IS set and no_llm is None, run the LLM path."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    out = tmp_path / "yc-advisor"
    kw = _base_kwargs(out, no_llm=None)
    run(**kw)

    mock_no_llm_pipeline["gen_skill"].assert_called_once()
    mock_no_llm_pipeline["gen_bench"].assert_called_once()
    mock_no_llm_pipeline["run_evals"].assert_called_once()
    assert not (out / ".pending-llm-draft.json").exists()
