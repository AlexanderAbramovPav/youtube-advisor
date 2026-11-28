import json
import pathlib
from unittest.mock import patch
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
    run(**_base_kwargs(out))
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
    a = _stub_video("a")
    a["playlist_ids"] = {"PLwanted"}
    b = _stub_video("b")
    b["playlist_ids"] = {"PLother"}
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


# ----- YouTube block during ingest -----

_BLOCKED_SUMMARY = {
    "ingested": 0, "skipped": 0, "failed": 1,
    "by_source": {"failed:blocked": 1},
    "blocked": True, "not_attempted": 2, "block_detail": "tier1: a: IpBlocked",
}


@pytest.mark.parametrize("no_llm", [False, True])
def test_block_leaves_resumable_advisor_and_skips_llm(tmp_path, mock_no_llm_pipeline, no_llm):
    """A block on the first video leaves 0 transcripts. Bootstrap still writes
    the state that lets a re-run resume (selection_filter.yaml, corpus_meta),
    spends no LLM calls on the partial corpus, and returns the retry command."""
    from youtube_advisor.bootstrap import run_ingest as patched_ingest
    patched_ingest.side_effect = lambda *a, **kw: dict(_BLOCKED_SUMMARY)
    out = tmp_path / "yc-advisor"
    result = run(**_base_kwargs(out, no_llm=no_llm))
    mock_no_llm_pipeline["gen_skill"].assert_not_called()
    mock_no_llm_pipeline["gen_bench"].assert_not_called()
    mock_no_llm_pipeline["run_evals"].assert_not_called()
    assert result["summary"]["blocked"] is True
    assert (out / "references" / "selection_filter.yaml").exists()
    assert (out / "references" / "corpus_meta.json").exists()
    assert (out / "SKILL.md").exists()
    assert (out / ".pending-llm-draft.json").exists()
    assert f"youtube-advisor update --advisor {out} --yes --regen-skill-md" in result["guide"]
    assert "1 video(s) blocked, 2 not tried yet" in result["guide"]
    progress = json.loads((out / ".progress.json").read_text())
    assert progress["stage"] == "blocked"


def test_cli_bootstrap_exits_nonzero_on_block(tmp_path):
    from click.testing import CliRunner
    from youtube_advisor.cli import main
    with patch("youtube_advisor.cli.bs_mod.run",
               return_value={"guide": "GUIDE-TEXT", "summary": dict(_BLOCKED_SUMMARY)}):
        res = CliRunner().invoke(main, ["bootstrap", "--channel", "@yc",
                                        "--out", str(tmp_path / "x"), "--yes"])
    assert res.exit_code == 1
    assert "GUIDE-TEXT" in res.output
    assert "YouTube blocked requests" in res.output


def test_cli_bootstrap_exits_zero_without_block(tmp_path):
    from click.testing import CliRunner
    from youtube_advisor.cli import main
    with patch("youtube_advisor.cli.bs_mod.run",
               return_value={"guide": "GUIDE-TEXT",
                             "summary": {"ingested": 1, "failed": 0, "by_source": {}}}):
        res = CliRunner().invoke(main, ["bootstrap", "--channel", "@yc",
                                        "--out", str(tmp_path / "x"), "--yes"])
    assert res.exit_code == 0, res.output


def test_resume_into_existing_advisor_passes_cookies_to_update(tmp_path):
    out = tmp_path / "yc-advisor"
    (out / "references").mkdir(parents=True)
    (out / "references" / "selection_filter.yaml").write_text("version: 1\n")
    with patch("youtube_advisor.update.run", return_value={"guide": "G"}) as up:
        run(**_base_kwargs(out, cookies_from_browser="chrome", no_llm=True))
    kw = up.call_args.kwargs
    assert kw["cookies_from_browser"] == "chrome"
    assert kw["no_llm"] is True
    assert kw["advisor_dir"] == str(out)


def test_meta_records_intent_and_counts_only_landed_videos(tmp_path, mock_pipeline):
    out = tmp_path / "yc-advisor"
    with patch("youtube_advisor.bootstrap.run_ingest",
               return_value={"ingested": 2, "skipped": 0, "failed": 1,
                             "by_source": {"captions": 2, "failed:YtDlpFailed": 1}}):
        run(**_base_kwargs(out))
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["n_videos"] == 2
    assert meta["user_intent"] == "test"


# ----- state a later update needs -----

def test_selection_filter_saves_whisper_skip(tmp_path, mock_pipeline):
    import yaml
    out = tmp_path / "yc-advisor"
    run(**_base_kwargs(out, whisper_skip_if_no_captions=True))
    snap = yaml.safe_load((out / "references" / "selection_filter.yaml").read_text())
    assert snap["whisper_skip_if_no_captions"] is True


@pytest.mark.parametrize("no_llm,no_evals,pending", [
    (False, False, True),   # bootstrap would have generated evals
    (True, False, False),   # --no-llm bootstrap never generates evals
    (False, True, False),   # --no-evals
])
def test_blocked_bootstrap_marks_evals_pending(tmp_path, mock_no_llm_pipeline,
                                               no_llm, no_evals, pending):
    from youtube_advisor.bootstrap import run_ingest as patched_ingest
    patched_ingest.side_effect = lambda *a, **kw: dict(_BLOCKED_SUMMARY)
    out = tmp_path / "yc-advisor"
    run(**_base_kwargs(out, no_llm=no_llm, no_evals=no_evals))
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta.get("evals_pending", False) is pending
    # No transcripts: the stored "en" is a placeholder.
    assert meta["corpus_language_detected"] is False
    assert meta["quote_style_requested"] == "auto"


def test_clean_bootstrap_marks_language_detected(tmp_path, mock_no_llm_pipeline):
    out = tmp_path / "yc-advisor"
    run(**_base_kwargs(out, quote_style="quote-only"))
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["corpus_language_detected"] is True
    assert meta["quote_style_requested"] == "quote-only"
    assert "evals_pending" not in meta


@pytest.mark.parametrize("flag,expected", [(False, None), (True, True)])
def test_resume_passes_whisper_skip_only_when_given(tmp_path, flag, expected):
    out = tmp_path / "yc-advisor"
    (out / "references").mkdir(parents=True)
    (out / "references" / "selection_filter.yaml").write_text("version: 1\n")
    with patch("youtube_advisor.update.run", return_value={"guide": "G"}) as up:
        run(**_base_kwargs(out, whisper_skip_if_no_captions=flag, no_evals=flag))
    assert up.call_args.kwargs["whisper_skip_if_no_captions"] is expected
    assert up.call_args.kwargs["no_evals"] is flag


def test_blocked_bootstrap_then_clean_update(tmp_path, mock_no_llm_pipeline):
    """End to end over the resume path: a bootstrap blocked on its first
    video, then an update that lands Russian transcripts without a block."""
    from youtube_advisor.bootstrap import run_ingest as patched_ingest
    from youtube_advisor.ingest.normalize import write_transcript
    from youtube_advisor.update import run as update_run
    patched_ingest.side_effect = lambda *a, **kw: dict(_BLOCKED_SUMMARY)
    out = tmp_path / "yc-advisor"
    run(**_base_kwargs(out, no_llm=False, answer_language="en"))

    def ingest(videos, out_dir, **kw):
        for v in videos:
            write_transcript(out_dir, v, [{"start": 0, "duration": 1, "text": "privet"}],
                             "captions", "ru")
        return {"ingested": len(videos), "skipped": 0, "failed": 0,
                "by_source": {"captions": len(videos)}}

    # The fixture stubs language detection to "en"; use the real one here.
    with patch("youtube_advisor.bootstrap._detect_corpus_language",
               side_effect=_detect_corpus_language), \
         patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a"), _stub_video("b"), _stub_video("c")]), \
         patch("youtube_advisor.update.run_ingest", side_effect=ingest), \
         patch("youtube_advisor.update.build_embeddings"), \
         patch("youtube_advisor.update.gen_bench") as bench, \
         patch("youtube_advisor.update.run_evals",
               return_value={"total": 1, "passed": 1, "delta_passed": None}) as evals:
        result = update_run(advisor_dir=str(out), yes=True, no_llm=False)
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["corpus_language"] == "ru"
    assert meta["quote_style"] == "translation-first"
    bench.assert_called_once_with(out)
    evals.assert_called_once_with(out)
    assert "evals_pending" not in meta
    assert "blocked" not in result["summary"]


# ----- duration selection -----

def test_cli_bootstrap_duration_flags_parsed(tmp_path):
    from click.testing import CliRunner
    from youtube_advisor.cli import main
    with patch("youtube_advisor.cli.bs_mod.run",
               return_value={"guide": "OK", "summary": {}}) as run_mock:
        res = CliRunner().invoke(
            main,
            ["bootstrap", "--channel", "@yc", "--min-duration", "1h30m",
             "--max-duration", "2:00:00", "--out", str(tmp_path / "x"), "--yes"],
        )
    assert res.exit_code == 0, res.output
    assert run_mock.call_args.kwargs["min_duration"] == 5400
    assert run_mock.call_args.kwargs["max_duration"] == 7200


@pytest.mark.parametrize("args", [
    ["--min-duration", "notaduration"],
    ["--max-duration", "1:60"],
    ["--min-duration=-5"],
    ["--min-duration", "-10m"],
    ["--max-duration=-1"],
    ["--min-duration", "2h", "--max-duration", "1h"],
    ["--max-duration", "0"],
    ["--max-duration", "0:00"],
])
def test_cli_bootstrap_duration_errors_exit_2_without_traceback(tmp_path, args):
    from click.testing import CliRunner
    from youtube_advisor.cli import main
    with patch("youtube_advisor.cli.bs_mod.run") as run_mock:
        res = CliRunner().invoke(
            main,
            ["bootstrap", "--channel", "@yc", *args, "--out", str(tmp_path / "x"), "--yes"],
        )
    assert res.exit_code == 2, res.output
    assert "Traceback" not in res.output
    run_mock.assert_not_called()


def test_bootstrap_saves_duration_bounds_and_nulls(tmp_path, mock_pipeline):
    import yaml
    out1 = tmp_path / "bounded"
    run(**_base_kwargs(out1, min_duration=300, max_duration=900))
    snap1 = yaml.safe_load((out1 / "references" / "selection_filter.yaml").read_text())
    assert snap1["min_duration"] == 300
    assert snap1["max_duration"] == 900

    out2 = tmp_path / "unbounded"
    run(**_base_kwargs(out2))
    snap2 = yaml.safe_load((out2 / "references" / "selection_filter.yaml").read_text())
    assert "min_duration" in snap2 and snap2["min_duration"] is None
    assert "max_duration" in snap2 and snap2["max_duration"] is None


def test_bootstrap_disables_listing_limit_when_duration_bound_set(tmp_path, mock_pipeline):
    out = tmp_path / "yc-advisor"
    with patch("youtube_advisor.bootstrap.resolve_channel",
               return_value=[_stub_video("a")]) as res_ch:
        run(**_base_kwargs(out, max_n=10, min_duration=300))
    assert res_ch.call_args.kwargs["limit"] is None


@pytest.mark.parametrize("min_d,max_d,expected", [
    (3600, None, "≥ 1h"),
    (None, 1200, "≤ 20m"),
    (600, 3600, "≥ 10m, ≤ 1h"),
])
def test_bootstrap_confirm_shows_readable_duration_bounds(
    tmp_path, mock_pipeline, capsys, min_d, max_d, expected
):
    out = tmp_path / "yc-advisor"
    vids = [{**_stub_video("a"), "length_seconds": 3600 if min_d == 3600 else 900}]
    with patch("youtube_advisor.bootstrap.resolve_channel", return_value=vids), \
         patch("youtube_advisor.bootstrap.click.confirm", return_value=True):
        run(**_base_kwargs(
            out,
            yes=False,
            from_natural_language=False,
            min_duration=min_d,
            max_duration=max_d,
        ))
    captured = capsys.readouterr().out
    assert expected in captured


def test_bootstrap_confirm_omits_duration_when_unset(tmp_path, mock_pipeline, capsys):
    out = tmp_path / "yc-advisor"
    with patch("youtube_advisor.bootstrap.click.confirm", return_value=True):
        run(**_base_kwargs(out, yes=False, from_natural_language=False))
    captured = capsys.readouterr().out
    assert "duration" not in captured.lower()
