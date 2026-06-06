import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from youtube_advisor.ingest.normalize import write_transcript
from youtube_advisor.build_embeddings import (
    build_embeddings,
    _chunks,
    _parse_md,
    CHUNK_WORDS,
    OVERLAP_WORDS,
    OFFSET_DTYPE,
)


def _meta(vid, pub="2024-06-01"):
    return {"video_id": vid, "title": f"T{vid}", "published_date": pub, "channel": "@x"}


def _seg(t, text):
    return {"start": t, "duration": 5, "text": text}


def _fake_embed_model(dim=384):
    """Return a TextEmbedding stub that produces deterministic fake vectors of `dim`."""
    def embed(texts):
        for i, _ in enumerate(texts):
            v = np.zeros(dim, dtype=np.float32)
            v[i % dim] = 1.0
            yield v
    m = MagicMock()
    m.embed = embed
    return m


def test_build_embeddings_model_change_forces_full_rebuild(tmp_path):
    """Bug #4: when embeddings.npz exists but stored model name differs from
    the requested one (user switched --multilingual), the fast-path must NOT
    return the stale npz — search would shape-error against the new
    corpus_meta.embedding_dim."""
    transcripts = tmp_path / "transcripts"
    refs = tmp_path / "references"
    transcripts.mkdir()
    write_transcript(transcripts, _meta("v1"), [_seg(0, "hello world")], "captions", "en")

    # First build with the default model (dim=384).
    with patch("youtube_advisor.build_embeddings.TextEmbedding",
               return_value=_fake_embed_model(dim=384)):
        build_embeddings(transcripts, refs, model_name="BAAI/bge-small-en-v1.5")
    first = np.load(refs / "embeddings.npz", allow_pickle=False)
    assert str(first["model"]) == "BAAI/bge-small-en-v1.5"
    assert first["vectors"].shape[1] == 384

    # Touch transcripts to keep mtimes consistent (not newer than npz).
    # Re-run with a DIFFERENT model. Fast-path must be bypassed despite no
    # transcript changes, so we rebuild and overwrite with bge-m3/1024.
    with patch("youtube_advisor.build_embeddings.TextEmbedding",
               return_value=_fake_embed_model(dim=1024)):
        build_embeddings(transcripts, refs, model_name="BAAI/bge-m3")
    second = np.load(refs / "embeddings.npz", allow_pickle=False)
    assert str(second["model"]) == "BAAI/bge-m3"
    assert second["vectors"].shape[1] == 1024


# ----- _chunks -----

def test_chunks_short_text_single_chunk(tmp_path):
    body = "\n".join(f"[00:00:{i:02d}] hello world" for i in range(5))
    chunks = _chunks(body)
    assert len(chunks) == 1
    assert chunks[0]["start_sec"] == 0
    assert chunks[0]["end_sec"] == 4
    assert "hello world" in chunks[0]["text"]


def test_chunks_long_text_sliding_window():
    body = "\n".join(f"[00:00:{i:02d}] " + ("word " * 50).strip() for i in range(60))
    # 60 lines × 50 words = 3000 words → multiple chunks
    chunks = _chunks(body)
    assert len(chunks) >= 2
    # All non-final chunks have exactly CHUNK_WORDS words
    for c in chunks[:-1]:
        assert len(c["text"].split()) == CHUNK_WORDS


def test_chunks_skip_malformed_lines():
    body = "this line has no timestamp\n[00:00:00] valid line\nnot a timestamp either"
    chunks = _chunks(body)
    assert len(chunks) == 1
    assert "valid line" in chunks[0]["text"]
    assert "no timestamp" not in chunks[0]["text"]


def test_chunks_empty_body_returns_empty():
    assert _chunks("") == []
    assert _chunks("no timestamps at all\nanother line") == []


# ----- build_embeddings -----

def test_build_emits_npz_with_vectors_and_offsets(tmp_path):
    t = tmp_path / "transcripts"
    write_transcript(t, _meta("a"),
        [_seg(i * 5, f"word{i} " * 80) for i in range(10)],
        "captions", "en")
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=_fake_embed_model()):
        out = build_embeddings(t, tmp_path / "references")
    npz = np.load(out, allow_pickle=False)
    assert "vectors" in npz.files
    assert "offsets" in npz.files
    assert "model" in npz.files
    assert npz["vectors"].shape[0] >= 1
    assert npz["offsets"].dtype.names[0] == "video_id"
    assert npz["vectors"].dtype == np.float16


def test_build_offsets_align_with_chunks(tmp_path):
    t = tmp_path / "transcripts"
    write_transcript(t, _meta("a"),
        [_seg(i * 5, "x " * 100) for i in range(15)],
        "captions", "en")
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=_fake_embed_model()):
        out = build_embeddings(t, tmp_path / "references")
    npz = np.load(out, allow_pickle=False)
    offsets = npz["offsets"]
    # Each row has the right video_id
    assert all(str(vid) == "a" for vid in offsets["video_id"])
    # chunk_index is monotonic from 0
    assert list(offsets["chunk_index"]) == list(range(len(offsets)))


def test_build_persists_model_name(tmp_path):
    t = tmp_path / "transcripts"
    write_transcript(t, _meta("a"), [_seg(0, "hi")], "captions", "en")
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=_fake_embed_model()):
        out = build_embeddings(t, tmp_path / "references", model_name="BAAI/bge-small-en-v1.5")
    npz = np.load(out, allow_pickle=False)
    assert str(npz["model"]) == "BAAI/bge-small-en-v1.5"


def test_build_empty_transcripts_returns_empty_npz(tmp_path):
    t = tmp_path / "transcripts"
    t.mkdir()
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=_fake_embed_model()):
        out = build_embeddings(t, tmp_path / "references")
    npz = np.load(out, allow_pickle=False)
    assert npz["vectors"].shape == (0, 384)
    assert len(npz["offsets"]) == 0


def test_incremental_skips_existing_video_ids(tmp_path):
    t = tmp_path / "transcripts"
    r = tmp_path / "references"
    write_transcript(t, _meta("a"), [_seg(0, "first")], "captions", "en")
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=_fake_embed_model()):
        build_embeddings(t, r)
    before = np.load(r / "embeddings.npz", allow_pickle=False)
    n_before = before["vectors"].shape[0]

    # Add a new video
    write_transcript(t, _meta("b"), [_seg(0, "second")], "captions", "en")
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=_fake_embed_model()):
        build_embeddings(t, r, incremental=True)
    after = np.load(r / "embeddings.npz", allow_pickle=False)
    assert after["vectors"].shape[0] > n_before
    ids = set(after["offsets"]["video_id"].tolist())
    assert {"a", "b"} <= ids


def test_rebuild_skipped_when_no_transcript_changed(tmp_path):
    """Second call with no transcript modifications must not rewrite embeddings.npz."""
    import time
    t = tmp_path / "transcripts"
    r = tmp_path / "references"
    write_transcript(t, _meta("a"), [_seg(0, "first")], "captions", "en")
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=_fake_embed_model()):
        build_embeddings(t, r)
    mtime_before = (r / "embeddings.npz").stat().st_mtime
    # Make sure mtime is observable (some filesystems have 1s granularity).
    time.sleep(0.05)
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=_fake_embed_model()):
        build_embeddings(t, r)
    mtime_after = (r / "embeddings.npz").stat().st_mtime
    assert mtime_after == mtime_before


def test_full_rebuild_overwrites(tmp_path):
    t = tmp_path / "transcripts"
    r = tmp_path / "references"
    write_transcript(t, _meta("a"), [_seg(0, "first")], "captions", "en")
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=_fake_embed_model()):
        build_embeddings(t, r)
    # Full rebuild should still work and yield the same result
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=_fake_embed_model()):
        out = build_embeddings(t, r, incremental=False)
    npz = np.load(out, allow_pickle=False)
    ids = set(npz["offsets"]["video_id"].tolist())
    assert "a" in ids


# ----- live smoke -----

@pytest.mark.slow
@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="downloads fastembed model")
def test_real_fastembed_pipeline(tmp_path):
    t = tmp_path / "transcripts"
    write_transcript(t, _meta("a"), [_seg(0, "hello world how are you today")], "captions", "en")
    out = build_embeddings(t, tmp_path / "references")
    npz = np.load(out, allow_pickle=False)
    assert npz["vectors"].shape == (1, 384)
