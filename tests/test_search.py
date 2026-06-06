import os
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from youtube_advisor.search import (
    search, _load_corpus_chunks, _bm25_rank, _rrf, _tok
)
from youtube_advisor.ingest.normalize import write_transcript
from youtube_advisor.build_index import build_index
from youtube_advisor.build_embeddings import build_embeddings

def _meta(vid, title, pub="2024-06-01"):
    return {"video_id": vid, "title": title, "published_date": pub, "channel": "@x"}

def _segs(text, n=10):
    return [{"start": i*5, "duration": 5, "text": text} for i in range(n)]

def _fake_embed_model(dim=384, vectors=None):
    """Stub fastembed: returns predetermined or zero vectors."""
    def embed(texts):
        for i, _ in enumerate(texts):
            if vectors is not None and i < len(vectors):
                yield vectors[i]
            else:
                v = np.zeros(dim, dtype=np.float32)
                v[hash(str(i)) % dim] = 1.0
                yield v
    m = MagicMock()
    m.embed = embed
    return m

def _setup_corpus(tmp_path, model_patch):
    """Helper: build a tiny corpus with three videos and call build_embeddings under mocked fastembed.

    Three docs are needed so BM25Okapi's IDF is non-zero for terms appearing in 1 doc.
    """
    t = tmp_path / "transcripts"
    write_transcript(t, _meta("a", "Hiring engineers"),
                     _segs("how to hire your first engineer carefully"), "captions", "en")
    write_transcript(t, _meta("b", "Bread baking"),
                     _segs("knead the dough slowly with care"), "captions", "en")
    write_transcript(t, _meta("c", "Painting walls"),
                     _segs("paint the wall with a roller brush"), "captions", "en")
    build_index(t, tmp_path / "references")
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=model_patch):
        build_embeddings(t, tmp_path / "references")

# ----- helpers -----

def test_tok_pattern():
    assert _tok.findall("Hello, world! 2026") == ["Hello", "world", "2026"]

def test_rrf_combines_rank_lists():
    # Doc 0 ranked 1st by both → highest RRF score
    # Doc 1 ranked 2nd by both
    # Doc 2 only in one list
    a = [(0, 10.0), (1, 5.0), (2, 1.0)]
    b = [(0, 0.9),  (1, 0.5), (3, 0.1)]
    merged = _rrf([a, b])
    assert merged[0][0] == 0
    # Combined ranks should boost 0 and 1 above 2 and 3
    top_ids = [m[0] for m in merged[:2]]
    assert 0 in top_ids and 1 in top_ids

def test_load_corpus_chunks_returns_chunks_with_metadata(tmp_path):
    t = tmp_path / "transcripts"
    write_transcript(t, _meta("a", "Title"), _segs("hi there"), "captions", "en")
    chunks = _load_corpus_chunks(tmp_path)
    assert len(chunks) >= 1
    c = chunks[0]
    assert c["video_id"] == "a"
    assert c["title"] == "Title"
    assert c["video_url"] == "https://www.youtube.com/watch?v=a"
    assert "start_sec" in c

# ----- _bm25_rank -----

def test_bm25_rank_returns_relevant_indices():
    chunks = [
        {"text": "how to hire engineers"},
        {"text": "knead dough slowly"},
        {"text": "engineers hire process"},
    ]
    ranked = _bm25_rank(chunks, "hire engineer", top_k=3)
    top_ids = [idx for idx, _ in ranked]
    assert 0 in top_ids[:2]  # has both terms
    assert 1 not in top_ids[:1]  # dough chunk should not be first

def test_bm25_rank_returns_empty_when_no_matches():
    chunks = [{"text": "completely unrelated text"}]
    ranked = _bm25_rank(chunks, "xyzzy nonexistent", top_k=5)
    assert ranked == []

# ----- search() integration -----

def test_search_keyword_finds_relevant(tmp_path):
    fake = _fake_embed_model()
    _setup_corpus(tmp_path, fake)
    res = search(tmp_path, "engineer", mode="keyword", top_k=3)
    assert res[0]["video_id"] == "a"
    assert "&t=" in res[0]["timestamp_link"]
    assert res[0]["snippet"]

def test_search_hybrid_runs_both_channels(tmp_path):
    fake = _fake_embed_model()
    _setup_corpus(tmp_path, fake)
    with patch("youtube_advisor.search.TextEmbedding", return_value=fake):
        res = search(tmp_path, "engineer", mode="hybrid", top_k=3)
    assert len(res) > 0
    # video 'a' should still rank highest because BM25 contributes
    assert res[0]["video_id"] == "a"

def test_search_semantic_uses_fastembed(tmp_path):
    fake = _fake_embed_model()
    _setup_corpus(tmp_path, fake)
    with patch("youtube_advisor.search.TextEmbedding", return_value=fake):
        res = search(tmp_path, "engineer", mode="semantic", top_k=3)
    assert isinstance(res, list)
    # Semantic with fake-stub vectors won't be meaningful, but the call should work
    assert all("timestamp_link" in r for r in res)

def test_search_empty_corpus_returns_empty(tmp_path):
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "references").mkdir()
    res = search(tmp_path, "anything", mode="keyword", top_k=5)
    assert res == []

def test_search_top_k_respected(tmp_path):
    fake = _fake_embed_model()
    _setup_corpus(tmp_path, fake)
    res = search(tmp_path, "engineer", mode="keyword", top_k=2)
    assert len(res) <= 2

def test_search_result_has_required_fields(tmp_path):
    fake = _fake_embed_model()
    _setup_corpus(tmp_path, fake)
    res = search(tmp_path, "engineer", mode="keyword", top_k=1)
    r = res[0]
    for field in ("video_id", "chunk_index", "title", "published_date",
                  "start_sec", "score", "snippet", "timestamp_link"):
        assert field in r

def test_search_timestamp_link_format(tmp_path):
    fake = _fake_embed_model()
    _setup_corpus(tmp_path, fake)
    res = search(tmp_path, "engineer", mode="keyword", top_k=1)
    link = res[0]["timestamp_link"]
    assert "watch?v=" in link


def test_load_corpus_chunks_cached_across_calls(tmp_path):
    """E-1/E-2: repeated searches on the same corpus must hit the cache —
    chunks load happens once until the corpus mtime changes."""
    from youtube_advisor.search import (
        _cached_chunks, _cached_npz, _corpus_mtime, _load_corpus_chunks_uncached,
    )
    _cached_chunks.cache_clear()
    fake = _fake_embed_model()
    _setup_corpus(tmp_path, fake)

    call_count = {"n": 0}
    original = _load_corpus_chunks_uncached
    def counting(corpus_dir):
        call_count["n"] += 1
        return original(corpus_dir)
    with patch("youtube_advisor.search._load_corpus_chunks_uncached",
               side_effect=counting):
        _cached_chunks.cache_clear()
        search(tmp_path, "engineer", mode="keyword", top_k=1)
        search(tmp_path, "hire", mode="keyword", top_k=1)
    # Both queries on same corpus → 1 uncached load.
    assert call_count["n"] == 1

# ----- regression: row alignment after incremental embedding build -----

def test_semantic_rank_aligned_after_incremental_build(tmp_path):
    """Build embeddings for 'm' and 'z' first, then incrementally add 'a'.
    After this the npz row order is m, z, a — but sorted(glob) order is a, m, z.
    A buggy _semantic_rank that treats npz rows as positional into chunks[]
    would return the wrong video for a query that uniquely matches 'a'.

    We assign each video a unique one-hot embedding so that querying with
    that one-hot deterministically picks the matching row.
    """
    t = tmp_path / "transcripts"
    # write only m and z first
    write_transcript(t, _meta("m", "Middle"),
                     _segs("middle text"), "captions", "en")
    write_transcript(t, _meta("z", "Zed"),
                     _segs("zed text"), "captions", "en")

    dim = 8
    # one-hot per video: m → axis 0, z → axis 1, a → axis 2
    m_vec = np.zeros(dim, dtype=np.float32); m_vec[0] = 1.0
    z_vec = np.zeros(dim, dtype=np.float32); z_vec[1] = 1.0
    a_vec = np.zeros(dim, dtype=np.float32); a_vec[2] = 1.0

    # First pass: only m and z; chunk count is 1 per video (short text).
    fake1 = _fake_embed_model(dim=dim, vectors=[m_vec, z_vec])
    build_index(t, tmp_path / "references")
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=fake1):
        build_embeddings(t, tmp_path / "references")

    # Now add 'a' and incrementally extend embeddings — its row gets appended
    # at the end (row index 2), so npz order is [m, z, a].
    write_transcript(t, _meta("a", "Apple"),
                     _segs("apple text"), "captions", "en")
    build_index(t, tmp_path / "references", incremental=True)
    fake2 = _fake_embed_model(dim=dim, vectors=[a_vec])
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=fake2):
        build_embeddings(t, tmp_path / "references", incremental=True)

    # Query with the one-hot for 'a' — must return video 'a', not whatever
    # video happens to sit at sorted-glob position 2 (which is 'z').
    query_fake = _fake_embed_model(dim=dim, vectors=[a_vec])
    with patch("youtube_advisor.search.TextEmbedding", return_value=query_fake):
        res = search(tmp_path, "apple", mode="semantic", top_k=1)
    assert res, "semantic search returned nothing"
    assert res[0]["video_id"] == "a", (
        f"expected video 'a' (matching one-hot), got {res[0]['video_id']}. "
        "This indicates _semantic_rank is using npz row index as a positional "
        "index into chunks[] (the bug)."
    )


# ----- live smoke -----

@pytest.mark.slow
@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="downloads fastembed model")
def test_real_hybrid_search(tmp_path):
    t = tmp_path / "transcripts"
    write_transcript(t, _meta("a", "Hiring"),
                     _segs("how to hire your first engineer"), "captions", "en")
    write_transcript(t, _meta("b", "Cooking"),
                     _segs("the dough should rise overnight"), "captions", "en")
    build_index(t, tmp_path / "references")
    build_embeddings(t, tmp_path / "references")
    res = search(tmp_path, "engineering hire", mode="hybrid", top_k=2)
    assert res[0]["video_id"] == "a"
