import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from youtube_advisor.ingest.normalize import write_transcript
from youtube_advisor.build_embeddings import (
    build_embeddings,
    _chunks,
    CHUNK_WORDS,
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


# ----- robustness + incremental rebuild -----

def _recording_model(dim=384):
    """Stub model that records every text it was asked to embed."""
    m = _fake_embed_model(dim)
    inner = m.embed
    m.seen = []

    def embed(texts):
        texts = list(texts)
        m.seen.extend(texts)
        return inner(texts)
    m.embed = embed
    return m


def _build(t, r, incremental, model=None):
    model = model or _recording_model()
    with patch("youtube_advisor.build_embeddings.TextEmbedding", return_value=model):
        out = build_embeddings(t, r, incremental=incremental)
    return out, model


def _set_mtime(path, ts):
    os.utime(path, (ts, ts))


def _transcript(t, vid, text):
    write_transcript(t, _meta(vid), [_seg(0, text)], "captions", "en")
    return next(t.glob(f"*-{vid}.md"))


def _ids(npz_path):
    with np.load(npz_path, allow_pickle=False) as npz:
        return npz["offsets"]["video_id"].tolist()


def _hashes(npz_path):
    with np.load(npz_path, allow_pickle=False) as npz:
        if "hashes" not in npz.files:
            return None
        h = npz["hashes"]
        return dict(zip(h["video_id"].tolist(), h["sha256"].tolist()))


def _rewrite_npz(path, drop=(), keep_rows=None):
    """Rewrite the npz without the `drop` members, keeping only the rows whose
    video_id is in `keep_rows` (all rows when None)."""
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files if k not in drop}
    if keep_rows is not None:
        mask = np.isin(arrays["offsets"]["video_id"], list(keep_rows))
        arrays["vectors"] = arrays["vectors"][mask]
        arrays["offsets"] = arrays["offsets"][mask]
    with open(path, "wb") as f:
        np.savez(f, **arrays)


def _make_legacy(path, drop_built_at=False):
    """Turn a fresh npz into one written before hashes were stored (and,
    with drop_built_at, before built_at was stored too). Returns built_at."""
    with np.load(path, allow_pickle=False) as npz:
        built_at = float(npz["built_at"])
    _rewrite_npz(path, drop=("hashes", "built_at") if drop_built_at else ("hashes",))
    return built_at


def _full_build_without_model(t, r):
    """Full build that must not construct the embedding model."""
    with patch("youtube_advisor.build_embeddings.TextEmbedding") as te:
        out = build_embeddings(t, r, incremental=False)
    te.assert_not_called()
    return out


def _corrupt_garbage(path):
    path.write_bytes(b"this is not a zip file")


def _corrupt_flip_middle(path):
    data = bytearray(path.read_bytes())
    mid = len(data) // 2
    for i in range(mid, min(mid + 64, len(data))):
        data[i] ^= 0xFF
    path.write_bytes(bytes(data))


def _corrupt_missing_offsets(path):
    with np.load(path, allow_pickle=False) as npz:
        vectors, model = npz["vectors"], npz["model"]
    with open(path, "wb") as f:
        np.savez(f, vectors=vectors, model=model)


def _corrupt_offsets_dtype(path):
    with np.load(path, allow_pickle=False) as npz:
        vectors, offsets, model = npz["vectors"], npz["offsets"], npz["model"]
    short = np.zeros(len(offsets), dtype=[("video_id", "U16"), ("chunk_index", "i4")])
    short["video_id"] = offsets["video_id"]
    with open(path, "wb") as f:
        np.savez(f, vectors=vectors, offsets=short, model=model)


@pytest.mark.parametrize("corrupt", [
    _corrupt_garbage, _corrupt_flip_middle, _corrupt_missing_offsets,
    _corrupt_offsets_dtype,
], ids=["garbage", "flipped-bytes", "missing-offsets", "offsets-dtype"])
def test_incremental_corrupt_npz_triggers_full_rebuild(tmp_path, caplog, corrupt):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    _transcript(t, "a", "alpha " * 200)
    _transcript(t, "b", "bravo " * 200)
    out, _ = _build(t, r, incremental=False)
    corrupt(out)

    with caplog.at_level("WARNING", logger="youtube_advisor.build_embeddings"):
        out, model = _build(t, r, incremental=True)

    assert "rebuilding all embeddings" in caplog.text
    # Both transcripts were embedded again, and the new file loads cleanly.
    assert any("alpha" in s for s in model.seen)
    assert any("bravo" in s for s in model.seen)
    assert sorted(set(_ids(out))) == ["a", "b"]


def test_failed_write_keeps_old_npz_and_leaves_no_temp_file(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    _transcript(t, "a", "alpha")
    out, _ = _build(t, r, incremental=False)
    old_bytes = out.read_bytes()
    _transcript(t, "b", "bravo")

    def failing_savez(file, **arrays):
        # Write part of a file, then die, as a crash mid-write would.
        if hasattr(file, "write"):
            file.write(b"PK\x03\x04partial")
        else:
            with open(file, "wb") as f:
                f.write(b"PK\x03\x04partial")
        raise OSError("disk full")

    with patch("youtube_advisor.build_embeddings.np.savez", side_effect=failing_savez):
        with pytest.raises(OSError, match="disk full"):
            _build(t, r, incremental=True)

    assert out.read_bytes() == old_bytes
    assert sorted(p.name for p in r.iterdir()) == ["embeddings.npz"]


def test_save_removes_temp_file_left_by_killed_process(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    _transcript(t, "a", "alpha")
    r.mkdir()
    (r / ".embeddings.99999.tmp.npz").write_bytes(b"partial")

    _build(t, r, incremental=False)

    assert sorted(p.name for p in r.iterdir()) == ["embeddings.npz"]


def test_incremental_reembeds_content_edited_during_previous_build(tmp_path):
    """The stored hash describes the text that was embedded, so an edit that
    lands after the build read the file is picked up next time, whatever the
    file's mtime says."""
    t, r = tmp_path / "transcripts", tmp_path / "references"
    a = _transcript(t, "a", "alpha")
    out, _ = _build(t, r, incremental=False)
    with np.load(out, allow_pickle=False) as npz:
        built_at = float(npz["built_at"])
    a.write_text(a.read_text().replace("alpha", "alpha edited"))
    _set_mtime(a, built_at - 5)

    _, model = _build(t, r, incremental=True)

    assert model.seen == ["alpha edited"]


def test_legacy_incremental_reembeds_transcript_newer_than_built_at(tmp_path):
    """A legacy npz (no hashes) still re-embeds a transcript whose mtime is
    newer than the stored built_at, and trusts the others."""
    t, r = tmp_path / "transcripts", tmp_path / "references"
    a = _transcript(t, "a", "alpha")
    b = _transcript(t, "b", "bravo")
    out, _ = _build(t, r, incremental=False)
    built_at = _make_legacy(out)
    _set_mtime(a, built_at + 5)     # edited mid-build...
    _set_mtime(b, built_at - 5)
    _set_mtime(out, built_at + 10)  # ...before the npz write finished

    out, model = _build(t, r, incremental=True)

    assert model.seen == ["alpha"]
    assert sorted(_ids(out)) == ["a", "b"]
    assert sorted(_hashes(out)) == ["a", "b"]


def test_incremental_embeds_only_new_transcript(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    a = _transcript(t, "a", "alpha")
    out, _ = _build(t, r, incremental=False)
    emb_mtime = out.stat().st_mtime
    _set_mtime(a, emb_mtime - 100)
    b = _transcript(t, "b", "bravo")
    # Even an old mtime must not hide a transcript the npz has never seen.
    _set_mtime(b, emb_mtime - 50)

    out, model = _build(t, r, incremental=True)

    assert model.seen == ["bravo"]
    assert sorted(_ids(out)) == ["a", "b"]


def test_incremental_reembeds_transcript_newer_than_npz(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    a = _transcript(t, "a", "alpha")
    b = _transcript(t, "b", "bravo")
    out, _ = _build(t, r, incremental=False)
    emb_mtime = out.stat().st_mtime
    _set_mtime(a, emb_mtime - 100)
    b.write_text(b.read_text().replace("bravo", "bravo edited"))
    _set_mtime(b, emb_mtime + 100)

    out, model = _build(t, r, incremental=True)

    assert model.seen == ["bravo edited"]
    # The changed transcript's old row was replaced, not duplicated.
    assert sorted(_ids(out)) == ["a", "b"]


def test_incremental_drops_rows_of_deleted_transcript(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    a = _transcript(t, "a", "alpha")
    b = _transcript(t, "b", "bravo")
    out, _ = _build(t, r, incremental=False)
    _set_mtime(a, out.stat().st_mtime - 100)
    b.unlink()

    out, model = _build(t, r, incremental=True)

    assert model.seen == []
    assert _ids(out) == ["a"]
    with np.load(out, allow_pickle=False) as npz:
        assert npz["vectors"].shape == (1, 384)


def test_incremental_all_deleted_keeps_vector_dim(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    a = _transcript(t, "a", "alpha")
    out, _ = _build(t, r, incremental=False)
    a.unlink()

    out, _ = _build(t, r, incremental=True)

    with np.load(out, allow_pickle=False) as npz:
        assert npz["vectors"].shape == (0, 384)
        assert len(npz["offsets"]) == 0


def test_full_build_fast_path_rebuilds_after_transcript_deleted(tmp_path):
    """The non-incremental fast path used to return the old file after a
    transcript was removed (e.g. an update rollback), keeping its rows."""
    t, r = tmp_path / "transcripts", tmp_path / "references"
    a = _transcript(t, "a", "alpha")
    b = _transcript(t, "b", "bravo")
    out, _ = _build(t, r, incremental=False)
    _set_mtime(a, out.stat().st_mtime - 100)
    b.unlink()

    out, _ = _build(t, r, incremental=False)

    assert _ids(out) == ["a"]


# ----- content hashes (T888) -----

def test_build_stores_a_hash_for_every_transcript(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    _transcript(t, "a", "alpha")
    # No timestamped lines: no rows, but the hash is still recorded.
    write_transcript(t, _meta("e"), [], "captions", "en")
    out, _ = _build(t, r, incremental=False)

    hashes = _hashes(out)
    assert sorted(hashes) == ["a", "e"]
    assert all(len(h) == 64 for h in hashes.values())
    assert _ids(out) == ["a"]


@pytest.mark.parametrize("change", ["touch", "rewrite-same-bytes"])
def test_touch_or_checkout_does_not_reembed(tmp_path, change):
    """(b) A touch or a git checkout changes mtimes but not content: neither
    build mode re-embeds anything, and the full build leaves the file alone."""
    t, r = tmp_path / "transcripts", tmp_path / "references"
    a = _transcript(t, "a", "alpha")
    b = _transcript(t, "b", "bravo")
    out, _ = _build(t, r, incremental=False)
    future = out.stat().st_mtime + 100
    for p in (a, b):
        if change == "rewrite-same-bytes":
            p.write_bytes(p.read_bytes())   # what a checkout does
        _set_mtime(p, future)
    before = out.read_bytes()

    _full_build_without_model(t, r)
    assert out.read_bytes() == before

    _, model = _build(t, r, incremental=True)
    assert model.seen == []
    assert sorted(_ids(out)) == ["a", "b"]


@pytest.mark.parametrize("drop_built_at", [False, True],
                         ids=["with-built_at", "pre-built_at"])
def test_legacy_incremental_backfills_hashes_without_reembedding(tmp_path, drop_built_at):
    """(a) An npz written before hashes were stored: the first incremental run
    trusts its vectors and only records the hashes; the next run then uses
    them, so a later touch re-embeds nothing."""
    t, r = tmp_path / "transcripts", tmp_path / "references"
    a = _transcript(t, "a", "alpha")
    b = _transcript(t, "b", "bravo")
    out, _ = _build(t, r, incremental=False)
    with np.load(out, allow_pickle=False) as npz:
        vectors_before = npz["vectors"].copy()
    built_at = _make_legacy(out, drop_built_at=drop_built_at)
    for p in (a, b):
        _set_mtime(p, built_at - 100)
    assert _hashes(out) is None

    out, model = _build(t, r, incremental=True)

    assert model.seen == []
    assert sorted(_hashes(out)) == ["a", "b"]
    with np.load(out, allow_pickle=False) as npz:
        assert np.array_equal(npz["vectors"], vectors_before)

    for p in (a, b):
        _set_mtime(p, out.stat().st_mtime + 100)
    _, model = _build(t, r, incremental=True)
    assert model.seen == []


@pytest.mark.parametrize("drop_built_at", [False, True],
                         ids=["with-built_at", "pre-built_at"])
def test_legacy_full_build_backfills_hashes_without_model(tmp_path, drop_built_at):
    """(a) The full-build fast path also records hashes for a legacy npz that
    is still up to date, without loading the model or touching the vectors."""
    t, r = tmp_path / "transcripts", tmp_path / "references"
    a = _transcript(t, "a", "alpha")
    out, _ = _build(t, r, incremental=False)
    with np.load(out, allow_pickle=False) as npz:
        vectors_before = npz["vectors"].copy()
    built_at = _make_legacy(out, drop_built_at=drop_built_at)
    _set_mtime(a, built_at - 100)

    _full_build_without_model(t, r)

    assert sorted(_hashes(out)) == ["a"]
    with np.load(out, allow_pickle=False) as npz:
        assert np.array_equal(npz["vectors"], vectors_before)
    # Hashes are in place now: a touch no longer matters.
    _set_mtime(a, out.stat().st_mtime + 100)
    _full_build_without_model(t, r)


def test_legacy_full_build_compares_with_built_at_not_npz_mtime(tmp_path):
    """A transcript edited mid-build is newer than built_at but older than
    the npz file: the fast path must still rebuild."""
    t, r = tmp_path / "transcripts", tmp_path / "references"
    a = _transcript(t, "a", "alpha")
    out, _ = _build(t, r, incremental=False)
    built_at = _make_legacy(out)
    _set_mtime(a, built_at + 5)
    _set_mtime(out, built_at + 10)

    _, model = _build(t, r, incremental=False)

    assert model.seen == ["alpha"]


@pytest.mark.parametrize("legacy", [False, True], ids=["hashed", "legacy"])
@pytest.mark.parametrize("incremental", [False, True], ids=["full", "incremental"])
def test_transcript_missing_from_npz_is_embedded(tmp_path, legacy, incremental):
    """(c) A transcript with chunks whose rows are missing from the npz (e.g.
    restored with `cp -p`, old mtime) is embedded, even if its hash is stored."""
    t, r = tmp_path / "transcripts", tmp_path / "references"
    a = _transcript(t, "a", "alpha")
    b = _transcript(t, "b", "bravo")
    out, _ = _build(t, r, incremental=False)
    with np.load(out, allow_pickle=False) as npz:
        built_at = float(npz["built_at"])
    _rewrite_npz(out, drop=("hashes",) if legacy else (), keep_rows={"a"})
    for p in (a, b):
        _set_mtime(p, built_at - 100)

    out, model = _build(t, r, incremental=incremental)

    assert "bravo" in model.seen
    if incremental:
        assert model.seen == ["bravo"]
    assert sorted(_ids(out)) == ["a", "b"]


def test_full_build_embeds_new_transcript_with_old_mtime(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    _transcript(t, "a", "alpha")
    out, _ = _build(t, r, incremental=False)
    b = _transcript(t, "b", "bravo")
    _set_mtime(b, out.stat().st_mtime - 100)

    out, model = _build(t, r, incremental=False)

    assert "bravo" in model.seen
    assert sorted(_ids(out)) == ["a", "b"]


def test_transcript_without_chunks_does_not_force_rebuild(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    _transcript(t, "a", "alpha")
    write_transcript(t, _meta("e"), [], "captions", "en")
    _build(t, r, incremental=False)

    _full_build_without_model(t, r)
    _, model = _build(t, r, incremental=True)
    assert model.seen == []


def test_chunking_change_reembeds(tmp_path):
    """The hash covers the chunking parameters: new chunk sizes re-embed."""
    t, r = tmp_path / "transcripts", tmp_path / "references"
    _transcript(t, "a", "alpha " * 30)
    _build(t, r, incremental=False)

    with patch("youtube_advisor.build_embeddings.CHUNK_WORDS", 10), \
         patch("youtube_advisor.build_embeddings.OVERLAP_WORDS", 2):
        _, model = _build(t, r, incremental=True)

    assert len(model.seen) > 1
