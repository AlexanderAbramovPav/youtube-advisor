import json
import pathlib
from unittest.mock import patch
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


# ----- YouTube block during ingest -----

def test_block_keeps_index_work_reports_retry_and_next_run_retries(tmp_path):
    """c lands, d is blocked: the index is still rebuilt for c, the guide
    counts only what landed and carries the retry command, and the next
    update hands d to run_ingest again (no --reingest flag needed)."""
    out = _setup_advisor(tmp_path, existing_ids=("a", "b"))
    listing = [_stub_video(v) for v in ("a", "b", "c", "d")]

    def blocked_ingest(videos, out_dir, **kw):
        write_transcript(out_dir, {"video_id": "c", "title": "t",
                                   "published_date": "2024-06-01", "channel": "@yc"},
                         [{"start": 0, "duration": 1, "text": "hi"}], "captions", "en")
        return {"ingested": 1, "skipped": 0, "failed": 1,
                "by_source": {"captions": 1, "failed:blocked": 1},
                "blocked": True, "not_attempted": 0, "block_detail": "tier1: d: IpBlocked"}

    with patch("youtube_advisor.update.resolve_channel", return_value=listing), \
         patch("youtube_advisor.update.run_ingest", side_effect=blocked_ingest) as ingest:
        result = run(advisor_dir=str(out), yes=True)
    assert [v["video_id"] for v in ingest.call_args.args[0]] == ["c", "d"]
    assert result["summary"]["blocked"] is True
    assert "+1 new" in result["guide"]
    assert "YouTube blocked requests (tier1: d: IpBlocked)" in result["guide"]
    assert f"youtube-advisor update --advisor {out} --yes" in result["guide"]
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["n_videos"] == 3

    with patch("youtube_advisor.update.resolve_channel", return_value=listing), \
         patch("youtube_advisor.update.run_ingest",
               return_value={"ingested": 0, "skipped": 0, "failed": 0, "by_source": {}}) as again:
        run(advisor_dir=str(out), yes=True)
    assert [v["video_id"] for v in again.call_args.args[0]] == ["d"]


def test_cli_update_exits_nonzero_on_block(tmp_path):
    from click.testing import CliRunner
    from youtube_advisor.cli import main
    with patch("youtube_advisor.update.run",
               return_value={"guide": "G", "evals": None,
                             "summary": {"blocked": True, "by_source": {}}}):
        res = CliRunner().invoke(main, ["update", "--advisor", str(tmp_path), "--yes"])
    assert res.exit_code == 1
    assert "YouTube blocked requests" in res.output


def _blocked_ingest_writing(vid):
    def _ingest(videos, out_dir, **kw):
        write_transcript(out_dir, {"video_id": vid, "title": "t",
                                   "published_date": "2024-06-01", "channel": "@yc"},
                         [{"start": 0, "duration": 1, "text": "hi"}], "captions", "en")
        return {"ingested": 1, "skipped": 0, "failed": 1,
                "by_source": {"captions": 1, "failed:blocked": 1},
                "blocked": True, "not_attempted": 0, "block_detail": "x"}
    return _ingest


def test_blocked_update_skips_skill_regen_and_evals(tmp_path):
    out = _setup_advisor(tmp_path, existing_ids=("a", "b"))
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video(v) for v in ("a", "b", "c", "d")]), \
         patch("youtube_advisor.update.run_ingest", side_effect=_blocked_ingest_writing("c")), \
         patch("youtube_advisor.gen_skill_md.generate") as gen, \
         patch("youtube_advisor.update.run_evals") as evals:
        result = run(advisor_dir=str(out), yes=True, regen_skill_md=True,
                     run_evals_=True, no_llm=False)
    gen.assert_not_called()
    assert (f"youtube-advisor update --advisor {out} --yes --regen-skill-md --run-evals"
            in result["guide"])
    evals.assert_not_called()
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    # Failures are not corpus sources.
    assert meta["transcript_source_counts"] == {"captions": 3}


def test_regen_uses_stored_intent_and_drops_stale_handoff(tmp_path):
    out = _setup_advisor(tmp_path, existing_ids=("a", "b"))
    meta_p = out / "references" / "corpus_meta.json"
    meta = json.loads(meta_p.read_text())
    meta["user_intent"] = "pricing advice"
    meta_p.write_text(json.dumps(meta))
    (out / ".pending-llm-draft.json").write_text("{}")

    def ok_ingest(videos, out_dir, **kw):
        write_transcript(out_dir, {"video_id": "c", "title": "t",
                                   "published_date": "2024-06-01", "channel": "@yc"},
                         [{"start": 0, "duration": 1, "text": "hi"}], "captions", "en")
        return {"ingested": 1, "skipped": 0, "failed": 0, "by_source": {"captions": 1}}

    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video(v) for v in ("a", "b", "c")]), \
         patch("youtube_advisor.update.run_ingest", side_effect=ok_ingest), \
         patch("youtube_advisor.gen_skill_md.generate") as gen:
        run(advisor_dir=str(out), yes=True, regen_skill_md=True, no_llm=False)
    assert gen.call_args.kwargs["user_intent"] == "pricing advice"
    assert not (out / ".pending-llm-draft.json").exists()


# ----- resume after a blocked bootstrap -----

def _ingest_writing(vid, lang="en", blocked=False):
    def _ingest(videos, out_dir, **kw):
        write_transcript(out_dir, {"video_id": vid, "title": "t",
                                   "published_date": "2024-06-01", "channel": "@yc"},
                         [{"start": 0, "duration": 1, "text": "privet"}], "captions", lang)
        s = {"ingested": 1, "skipped": 0, "failed": int(blocked),
             "by_source": {"captions": 1}}
        if blocked:
            s["by_source"]["failed:blocked"] = 1
            s.update(blocked=True, not_attempted=0, block_detail="x")
        return s
    return _ingest


def _empty_bootstrap_advisor(tmp_path, **meta_overrides):
    """State a bootstrap leaves when YouTube blocked the first video."""
    out = _setup_advisor(tmp_path, existing_ids=())
    meta_p = out / "references" / "corpus_meta.json"
    meta = json.loads(meta_p.read_text())
    meta.update({"n_videos": 0, "corpus_language": "en",
                 "corpus_language_detected": False,
                 "answer_language": "en", "quote_style": "quote-only",
                 "quote_style_requested": "auto",
                 "corpus_version": "2024-06-01/3"})
    meta.update(meta_overrides)
    meta_p.write_text(json.dumps(meta))
    return out


def _run_update(out, ingest, **kw):
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video(v) for v in ("a", "b", "c")]), \
         patch("youtube_advisor.update.run_ingest", side_effect=ingest) as ing, \
         patch("youtube_advisor.update.build_embeddings"), \
         patch("youtube_advisor.update.gen_bench") as bench, \
         patch("youtube_advisor.update.run_evals",
               return_value={"total": 0, "passed": 0, "delta_passed": None}) as evals:
        result = run(advisor_dir=str(out), yes=True, **kw)
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    return result, meta, {"ingest": ing, "gen_bench": bench, "run_evals": evals}


def test_language_detected_after_empty_bootstrap_and_auto_quote_style_resolved(tmp_path):
    out = _empty_bootstrap_advisor(tmp_path)
    (out / ".pending-llm-draft.json").write_text("{}")
    _, meta, _ = _run_update(out, _ingest_writing("a", lang="ru"), no_llm=True)
    assert meta["corpus_language"] == "ru"
    assert meta["corpus_language_detected"] is True
    # answer en, corpus ru → "auto" resolves to translation-first now.
    assert meta["quote_style"] == "translation-first"
    # The stub and the AI hand-off are re-rendered with the new values.
    assert "corpus `ru`" in (out / "README.md").read_text()
    handoff = json.loads((out / ".pending-llm-draft.json").read_text())
    assert handoff["quote_style"] == "translation-first"
    assert handoff["corpus_meta"]["corpus_language"] == "ru"


def test_fixed_quote_style_kept_when_language_detected(tmp_path):
    out = _empty_bootstrap_advisor(tmp_path, quote_style_requested="quote-only")
    _, meta, _ = _run_update(out, _ingest_writing("a", lang="ru"), no_llm=True)
    assert meta["corpus_language"] == "ru"
    assert meta["quote_style"] == "quote-only"


def test_language_detected_once_then_left_alone(tmp_path):
    out = _empty_bootstrap_advisor(tmp_path)
    _run_update(out, _ingest_writing("a", lang="ru"), no_llm=True)
    _, meta, _ = _run_update(out, _ingest_writing("b", lang="en"), no_llm=True)
    assert meta["corpus_language"] == "ru"


def test_language_detected_from_transcripts_of_a_blocked_update(tmp_path):
    out = _empty_bootstrap_advisor(tmp_path)
    _, meta, _ = _run_update(out, _ingest_writing("a", lang="ru", blocked=True), no_llm=True)
    assert meta["corpus_language"] == "ru"


def test_legacy_meta_without_marker(tmp_path):
    """No marker: an empty stored corpus means the language was a placeholder.
    Without quote_style_requested, the stored style is kept as a fixed one."""
    out = _empty_bootstrap_advisor(tmp_path)
    meta_p = out / "references" / "corpus_meta.json"
    meta = json.loads(meta_p.read_text())
    del meta["corpus_language_detected"], meta["quote_style_requested"]
    meta_p.write_text(json.dumps(meta))
    _, meta, _ = _run_update(out, _ingest_writing("a", lang="ru"), no_llm=True)
    assert meta["corpus_language"] == "ru"
    assert meta["quote_style"] == "quote-only"


def test_legacy_meta_with_corpus_keeps_language(tmp_path):
    out = _setup_advisor(tmp_path, existing_ids=("a",))
    _, meta, _ = _run_update(out, _ingest_writing("b", lang="ru"), no_llm=True)
    assert "corpus_language" not in meta
    assert meta["quote_style"] == "quote-only"


def test_whisper_skip_saved_at_bootstrap_is_honoured(tmp_path):
    out = _setup_advisor(tmp_path, existing_ids=("a",))
    filt_p = out / "references" / "selection_filter.yaml"
    filt = yaml.safe_load(filt_p.read_text())
    filt["whisper_skip_if_no_captions"] = True
    filt_p.write_text(yaml.safe_dump(filt))
    _, _, m = _run_update(out, _ingest_writing("b"), no_llm=True)
    assert m["ingest"].call_args.kwargs["skip_whisper"] is True


def test_whisper_runs_when_flag_not_saved(tmp_path):
    out = _setup_advisor(tmp_path, existing_ids=("a",))
    _, _, m = _run_update(out, _ingest_writing("b"), no_llm=True)
    assert m["ingest"].call_args.kwargs["skip_whisper"] is False


def test_whisper_skip_override_is_saved(tmp_path):
    out = _setup_advisor(tmp_path, existing_ids=("a",))
    _, _, m = _run_update(out, _ingest_writing("b"), no_llm=True,
                          whisper_skip_if_no_captions=True)
    assert m["ingest"].call_args.kwargs["skip_whisper"] is True
    filt = yaml.safe_load((out / "references" / "selection_filter.yaml").read_text())
    assert filt["whisper_skip_if_no_captions"] is True


def test_first_clean_update_generates_pending_evals(tmp_path):
    out = _empty_bootstrap_advisor(tmp_path, evals_pending=True)
    result, meta, m = _run_update(out, _ingest_writing("a"), no_llm=False,
                                  run_evals_=True)
    m["gen_bench"].assert_called_once_with(out)
    m["run_evals"].assert_called_once_with(out)  # not twice with --run-evals
    assert result["evals"] is not None
    assert "evals_pending" not in meta

    # The next update does not generate them again.
    _, _, m = _run_update(out, _ingest_writing("b"), no_llm=False)
    m["gen_bench"].assert_not_called()


def test_blocked_update_keeps_evals_pending(tmp_path):
    out = _empty_bootstrap_advisor(tmp_path, evals_pending=True)
    result, meta, m = _run_update(out, _ingest_writing("a", blocked=True), no_llm=False)
    m["gen_bench"].assert_not_called()
    m["run_evals"].assert_not_called()
    assert meta["evals_pending"] is True
    assert "youtube-advisor update --advisor" in result["guide"]


def test_no_llm_update_keeps_evals_pending(tmp_path):
    out = _empty_bootstrap_advisor(tmp_path, evals_pending=True)
    _, meta, m = _run_update(out, _ingest_writing("a"), no_llm=True)
    m["gen_bench"].assert_not_called()
    assert meta["evals_pending"] is True


def test_pending_evals_generated_when_nothing_new(tmp_path):
    """The update that landed the transcripts ran with --no-llm; a later one
    with nothing new still generates the pending evals."""
    out = _empty_bootstrap_advisor(tmp_path, evals_pending=True)
    _run_update(out, _ingest_writing("a"), no_llm=True)
    for vid in ("b", "c"):
        write_transcript(out / "transcripts",
                         {"video_id": vid, "title": "t", "published_date": "2024-06-01",
                          "channel": "@yc"},
                         [{"start": 0, "duration": 1, "text": "x"}], "captions", "en")
    _, meta, m = _run_update(out, _ingest_writing("z"), no_llm=False)
    m["ingest"].assert_not_called()
    m["gen_bench"].assert_called_once_with(out)
    assert "evals_pending" not in meta


def test_stub_refreshed_when_placeholder_language_is_confirmed(tmp_path):
    """English channel: "en" is detected again, but the hand-off a blocked
    bootstrap wrote had no samples and 0 videos, so it is still rewritten."""
    out = _empty_bootstrap_advisor(tmp_path)
    (out / ".pending-llm-draft.json").write_text(json.dumps({"samples": []}))
    _, meta, _ = _run_update(out, _ingest_writing("a", lang="en"), no_llm=True)
    assert meta["corpus_language"] == "en"
    assert meta["corpus_language_detected"] is True
    handoff = json.loads((out / ".pending-llm-draft.json").read_text())
    assert [s["video_id"] for s in handoff["samples"]] == ["a"]
    assert handoff["corpus_meta"]["n_videos"] == 1


def test_pending_evals_failure_does_not_fail_update(tmp_path):
    out = _empty_bootstrap_advisor(tmp_path, evals_pending=True)
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a")]), \
         patch("youtube_advisor.update.run_ingest", side_effect=_ingest_writing("a")), \
         patch("youtube_advisor.update.build_embeddings"), \
         patch("youtube_advisor.update.gen_bench", side_effect=RuntimeError("no credit")), \
         patch("youtube_advisor.update.run_evals") as evals:
        result = run(advisor_dir=str(out), yes=True, no_llm=False)
    evals.assert_not_called()
    assert "+1 new" in result["guide"]
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["evals_pending"] is True


def test_pending_evals_kept_when_run_evals_fails(tmp_path):
    out = _empty_bootstrap_advisor(tmp_path, evals_pending=True)
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a")]), \
         patch("youtube_advisor.update.run_ingest", side_effect=_ingest_writing("a")), \
         patch("youtube_advisor.update.build_embeddings"), \
         patch("youtube_advisor.update.gen_bench"), \
         patch("youtube_advisor.update.run_evals", side_effect=ValueError("bad json")):
        run(advisor_dir=str(out), yes=True, no_llm=False)
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["evals_pending"] is True


def test_benchmark_written_by_the_ai_is_kept(tmp_path):
    out = _empty_bootstrap_advisor(tmp_path, evals_pending=True)
    (out / "evals").mkdir()
    (out / "evals" / "benchmark.json").write_text('{"questions": []}')
    _, meta, m = _run_update(out, _ingest_writing("a"), no_llm=False)
    m["gen_bench"].assert_not_called()
    m["run_evals"].assert_called_once_with(out)
    assert "evals_pending" not in meta


def _writing_gen_bench(out):
    (out / "evals").mkdir(exist_ok=True)
    (out / "evals" / "benchmark.json").write_text('{"questions": []}')


def test_failed_eval_run_is_retried_by_next_update(tmp_path):
    """gen_bench wrote the benchmark, then run_evals failed: the next update
    runs the evals (without a new benchmark) before clearing the marker."""
    out = _empty_bootstrap_advisor(tmp_path, evals_pending=True)
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a"), _stub_video("b")]), \
         patch("youtube_advisor.update.run_ingest", side_effect=_ingest_writing("a")), \
         patch("youtube_advisor.update.build_embeddings"), \
         patch("youtube_advisor.update.gen_bench", side_effect=_writing_gen_bench), \
         patch("youtube_advisor.update.run_evals", side_effect=ValueError("bad")):
        run(advisor_dir=str(out), yes=True, no_llm=False)
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["evals_pending"] is True
    _, meta, m = _run_update(out, _ingest_writing("b"), no_llm=False)
    m["gen_bench"].assert_not_called()
    m["run_evals"].assert_called_once_with(out)
    assert "evals_pending" not in meta


def test_failed_pending_evals_do_not_fall_through_to_run_evals(tmp_path):
    """With --run-evals, a failed pending attempt must not be followed by an
    unguarded eval run (no benchmark exists)."""
    out = _empty_bootstrap_advisor(tmp_path, evals_pending=True)
    with patch("youtube_advisor.update.resolve_channel",
               return_value=[_stub_video("a")]), \
         patch("youtube_advisor.update.run_ingest", side_effect=_ingest_writing("a")), \
         patch("youtube_advisor.update.build_embeddings"), \
         patch("youtube_advisor.update.gen_bench", side_effect=RuntimeError("no credit")), \
         patch("youtube_advisor.update.run_evals") as evals:
        result = run(advisor_dir=str(out), yes=True, no_llm=False, run_evals_=True)
    evals.assert_not_called()
    assert "+1 new" in result["guide"]


def test_no_evals_drops_pending_evals(tmp_path):
    out = _empty_bootstrap_advisor(tmp_path, evals_pending=True)
    _, meta, m = _run_update(out, _ingest_writing("a"), no_llm=False, no_evals=True)
    m["gen_bench"].assert_not_called()
    assert "evals_pending" not in meta
