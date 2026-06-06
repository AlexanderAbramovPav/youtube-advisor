"""End-to-end test of bootstrap + update against a fixture corpus.

Strategy: pre-populate transcripts via write_transcript (bypassing fetch + ingest
cascade), mock the LLM calls (gen_skill_md.generate, gen_benchmark.generate),
and let real build_index / build_embeddings / search / eval_runner /
postrun_guide run on a tiny in-tree dataset.

This gives a real cross-module integration check without flakiness — no network
is touched. The fastembed model (bge-small-en-v1.5, ~24MB) is downloaded once
on first run if not pre-cached.
"""
from __future__ import annotations
import json
import pathlib
from unittest.mock import patch

import pytest

from youtube_advisor.bootstrap import run as bootstrap_run
from youtube_advisor.update import run as update_run
from youtube_advisor.search import search
from youtube_advisor.ingest.normalize import write_transcript


pytestmark = pytest.mark.e2e


def _fixture_videos() -> list[dict]:
    """5 synthetic 'YC-like' video metadata records covering diverse topics."""
    return [
        {"video_id": "vidhire1", "title": "How to hire engineers",
         "published_date": "2024-01-15", "length_seconds": 1789, "views": 50000,
         "description": "Jared Friedman on hiring", "channel": "@yc",
         "channel_id": "UCyc", "playlist_ids": set()},
        {"video_id": "vidpivot1", "title": "When to pivot your startup",
         "published_date": "2024-02-20", "length_seconds": 2100, "views": 30000,
         "description": "Michael Seibel on pivots", "channel": "@yc",
         "channel_id": "UCyc", "playlist_ids": set()},
        {"video_id": "vidpmf1", "title": "Product market fit",
         "published_date": "2024-03-10", "length_seconds": 1500, "views": 80000,
         "description": "How to find PMF", "channel": "@yc",
         "channel_id": "UCyc", "playlist_ids": set()},
        {"video_id": "vidvc1", "title": "How to raise from VCs",
         "published_date": "2024-04-05", "length_seconds": 2400, "views": 40000,
         "description": "Fundraising basics", "channel": "@yc",
         "channel_id": "UCyc", "playlist_ids": set()},
        {"video_id": "vidsales1", "title": "B2B sales for founders",
         "published_date": "2024-05-12", "length_seconds": 1800, "views": 25000,
         "description": "Cold outreach + closing", "channel": "@yc",
         "channel_id": "UCyc", "playlist_ids": set()},
    ]


_BODIES = {
    "vidhire1": [
        "how to hire your first engineer carefully and deliberately",
        "look for technical co-founders who have shipped real products before",
        "salary expectations for engineers should be calibrated to your stage",
    ],
    "vidpivot1": [
        "when to pivot your startup based on real user feedback",
        "pivoting is not failure it is learning from the market",
        "talk to twenty users before deciding to pivot the company",
    ],
    "vidpmf1": [
        "product market fit is the only thing that matters for a startup",
        "you will know PMF when usage explodes organically without paid acquisition",
        "until PMF every founder effort should be product-focused not growth",
    ],
    "vidvc1": [
        "how to raise from VCs and pitch effectively to investors",
        "VCs invest in lines not dots — show progress over time consistently",
        "term sheet negotiation comes down to leverage and competing offers",
    ],
    "vidsales1": [
        "B2B sales for founders requires personal outreach and persistence",
        "founders sell better than salespeople in the first year of company",
        "follow up three times before giving up on a sales prospect",
    ],
}


def _fake_run_ingest(videos, out_dir, **kw):
    """Stand-in for run_ingest: write each video's transcript with synthetic body."""
    for v in videos:
        body = _BODIES.get(v["video_id"], ["placeholder content for this video"])
        segs = [{"start": i * 30, "duration": 30, "text": text}
                for i, text in enumerate(body)]
        write_transcript(out_dir, v, segs, "captions", "en")
    return {"ingested": len(videos), "skipped": 0, "failed": 0,
            "by_source": {"captions": len(videos)}}


def _fake_gen_skill(advisor_dir, **kw):
    """Write minimal SKILL.md + scaffold so postrun_guide can read it."""
    (advisor_dir / "SKILL.md").write_text(
        "---\nname: yc-advisor\ndescription: test advisor\n---\n# yc-advisor\n"
    )
    (advisor_dir / "AGENTS.md").write_text("# yc-advisor\nbody\n")
    (advisor_dir / "README.md").write_text("# yc-advisor\nstub\n")
    (advisor_dir / "references").mkdir(exist_ok=True)
    (advisor_dir / "references" / "workflow.md").write_text("# workflow\n")
    (advisor_dir / "assets").mkdir(exist_ok=True)
    (advisor_dir / "assets" / "answer_template.md").write_text("> quote\n")


def _fake_gen_benchmark(advisor_dir, target=8):
    """Write a benchmark with one passing question targeting vidhire1."""
    (advisor_dir / "evals").mkdir(exist_ok=True)
    (advisor_dir / "evals" / "benchmark.json").write_text(json.dumps({
        "version": 1,
        "generated_by": "test",
        "corpus_version": "test",
        "questions": [
            {"q": "how to hire engineer", "lang": "en",
             "expected_video_ids": ["vidhire1"],
             "expected_keywords": ["hire"], "min_quotes": 1,
             "expected_quote_substrings": []},
        ],
    }))


def _base_kwargs(out, **overrides):
    base = {
        "channels": ("@yc",), "since": None, "until": None, "max_n": 500,
        "playlists": (), "title_include": None, "title_exclude": r"#?shorts\b",
        "ids": None, "out_dir": str(out),
        "whisper_skip_if_no_captions": False, "multilingual": False,
        "vendor": False, "no_evals": False,
        "from_natural_language": True, "intent": "YC startup advice",
        "answer_language": "en", "quote_style": "auto", "yes": True,
        # Force the LLM path so the test's mocked gen_skill / gen_bench fire.
        "no_llm": False,
    }
    base.update(overrides)
    return base


def test_bootstrap_then_update_idempotent(tmp_path):
    out = tmp_path / "yc-advisor"
    videos = _fixture_videos()

    with patch("youtube_advisor.bootstrap.resolve_channel", return_value=videos), \
         patch("youtube_advisor.bootstrap.run_ingest", side_effect=_fake_run_ingest), \
         patch("youtube_advisor.bootstrap.gen_skill", side_effect=_fake_gen_skill), \
         patch("youtube_advisor.bootstrap.gen_bench", side_effect=_fake_gen_benchmark):
        result = bootstrap_run(**_base_kwargs(out))

    # ---- Artifacts exist ----
    assert (out / "SKILL.md").exists()
    assert (out / "AGENTS.md").exists()
    assert (out / "references" / "index.json").exists()
    assert (out / "references" / "embeddings.npz").exists()
    assert (out / "references" / "corpus_meta.json").exists()
    assert (out / "references" / "selection_filter.yaml").exists()
    assert (out / "evals" / "benchmark.json").exists()
    assert (out / "scripts" / "search.py").exists()

    # ---- Corpus meta ----
    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["n_videos"] == 5
    assert meta["channels"] == ["@yc"]
    assert meta["embedding_model"] == "BAAI/bge-small-en-v1.5"

    # ---- Eval pass: 1 question targeting "hire" → vidhire1 ----
    results_dir = out / "evals" / "results"
    result_files = list(results_dir.glob("*.json"))
    assert len(result_files) == 1
    eval_summary = json.loads(result_files[0].read_text())
    assert eval_summary["total"] == 1
    assert eval_summary["passed"] == 1

    # ---- Search returns timestamp-linked results from the right video ----
    hits = search(out, "hire engineer", top_k=3, mode="hybrid")
    assert len(hits) > 0
    top = hits[0]
    assert top["video_id"] == "vidhire1"
    assert "&t=" in top["timestamp_link"]
    assert "youtube.com/watch?v=vidhire1" in top["timestamp_link"]

    # ---- Postrun guide present in bootstrap result ----
    assert "yc-advisor" in result["guide"]
    assert "How to use" in result["guide"] or "Corpus:" in result["guide"]

    # ---- Update is no-op ----
    with patch("youtube_advisor.update.resolve_channel", return_value=videos):
        update_result = update_run(advisor_dir=str(out), yes=True)
    assert "0 new" in update_result["guide"] and "unchanged" in update_result["guide"]
    meta_after = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta_after["n_videos"] == 5


def test_bootstrap_then_update_adds_new_video(tmp_path):
    out = tmp_path / "yc-advisor"
    initial = _fixture_videos()
    extra = {
        "video_id": "vidnew1", "title": "New video on growth",
        "published_date": "2026-06-04", "length_seconds": 1200, "views": 1000,
        "description": "Growth tactics", "channel": "@yc",
        "channel_id": "UCyc", "playlist_ids": set(),
    }

    # First bootstrap with 5 videos.
    with patch("youtube_advisor.bootstrap.resolve_channel", return_value=initial), \
         patch("youtube_advisor.bootstrap.run_ingest", side_effect=_fake_run_ingest), \
         patch("youtube_advisor.bootstrap.gen_skill", side_effect=_fake_gen_skill), \
         patch("youtube_advisor.bootstrap.gen_bench", side_effect=_fake_gen_benchmark):
        bootstrap_run(**_base_kwargs(out, intent="YC advice"))

    # Update sees +1 new video; ingest mock writes just the new one.
    def _ingest_extra(videos, out_dir, **kw):
        for v in videos:
            write_transcript(
                out_dir, v,
                [{"start": 0, "duration": 30,
                  "text": f"growth tactics and lessons for {v['video_id']}"}],
                "captions", "en",
            )
        return {"ingested": len(videos), "skipped": 0, "failed": 0,
                "by_source": {"captions": len(videos)}}

    with patch("youtube_advisor.update.resolve_channel", return_value=initial + [extra]), \
         patch("youtube_advisor.update.run_ingest", side_effect=_ingest_extra):
        result = update_run(advisor_dir=str(out), yes=True)

    meta = json.loads((out / "references" / "corpus_meta.json").read_text())
    assert meta["n_videos"] == 6
    assert "+1" in result["guide"]
    assert (out / "transcripts" / "2026-06-04-vidnew1.md").exists()
