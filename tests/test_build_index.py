import json, time, pathlib
import pytest
from youtube_advisor.ingest.normalize import write_transcript
from youtube_advisor.build_index import build_index, _parse_frontmatter

def _meta(vid, pub="2024-06-01", title=None):
    return {"video_id": vid, "title": title or f"Title {vid}",
            "published_date": pub, "channel": "@x", "length_seconds": 600}

def _seg(t, text="hi"):
    return {"start": t, "duration": 5, "text": text}

# ----- _parse_frontmatter unit -----

def test_parse_frontmatter_returns_dict_and_body():
    text = "---\ntitle: T\nvideo_id: a\n---\n\nbody here\n"
    fm, body = _parse_frontmatter(text)
    assert fm == {"title": "T", "video_id": "a"}
    assert body == "\nbody here\n"

def test_parse_frontmatter_returns_empty_when_no_frontmatter():
    fm, body = _parse_frontmatter("no frontmatter here")
    assert fm == {}
    assert body == "no frontmatter here"

# ----- build_index -----

def test_build_index_creates_index_json(tmp_path):
    t = tmp_path / "transcripts"
    r = tmp_path / "references"
    write_transcript(t, _meta("a"), [_seg(0, "hello world")], "captions", "en")
    out = build_index(t, r)
    assert out == r / "index.json"
    data = json.loads(out.read_text())
    assert data["version"] == 1
    assert data["n_videos"] == 1
    v = data["videos"][0]
    assert v["video_id"] == "a"
    assert v["title"] == "Title a"
    assert v["transcript_source"] == "captions"
    assert v["video_url"] == "https://www.youtube.com/watch?v=a"
    assert v["summary"].startswith("[00:00:00] hello world")

def test_sorted_by_published_date_desc(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    write_transcript(t, _meta("a", "2022-01-01"), [], "captions", "en")
    write_transcript(t, _meta("b", "2026-01-01"), [], "captions", "en")
    write_transcript(t, _meta("c", "2024-06-01"), [], "captions", "en")
    out = build_index(t, r)
    ids = [v["video_id"] for v in json.loads(out.read_text())["videos"]]
    assert ids == ["b", "c", "a"]

def test_skips_mds_without_video_id(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    (t).mkdir()
    (t / "broken.md").write_text("no frontmatter\n")
    write_transcript(t, _meta("a"), [], "captions", "en")
    out = build_index(t, r)
    data = json.loads(out.read_text())
    assert data["n_videos"] == 1
    assert data["videos"][0]["video_id"] == "a"

def test_summary_truncated_to_200_chars(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    long_text = "x" * 1000
    write_transcript(t, _meta("a"), [_seg(0, long_text)], "captions", "en")
    out = build_index(t, r)
    summary = json.loads(out.read_text())["videos"][0]["summary"]
    assert len(summary) == 200

def test_incremental_reuses_existing_entries(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    write_transcript(t, _meta("a"), [_seg(0, "v1")], "captions", "en")
    build_index(t, r)
    # Sleep to ensure the new transcript has a strictly later mtime
    time.sleep(0.05)
    write_transcript(t, _meta("b"), [_seg(0, "v2")], "captions", "en")
    # Touch index.json BEFORE writing the new transcript to simulate ordering — actually
    # we want index.json mtime older than the new transcript: that's natural after sleep.
    # But _meta("a") was written first; rewriting it would update mtime. We don't rewrite it,
    # so its mtime is unchanged, satisfying "mtime <= out_mtime → reuse existing entry".
    out = build_index(t, r, incremental=True)
    data = json.loads(out.read_text())
    ids = {v["video_id"] for v in data["videos"]}
    assert ids == {"a", "b"}
    assert data["n_videos"] == 2

def test_incremental_rewrites_modified_entries(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    write_transcript(t, _meta("a", title="Old title"), [], "captions", "en")
    build_index(t, r)
    time.sleep(0.05)
    # Overwrite with new content
    write_transcript(t, _meta("a", title="New title"), [], "captions", "en")
    out = build_index(t, r, incremental=True)
    data = json.loads(out.read_text())
    assert data["videos"][0]["title"] == "New title"

def test_full_rebuild_ignores_existing_index(tmp_path):
    t, r = tmp_path / "transcripts", tmp_path / "references"
    write_transcript(t, _meta("a", title="V1"), [], "captions", "en")
    build_index(t, r)
    # Modify the transcript silently — full rebuild should re-read everything
    write_transcript(t, _meta("a", title="V2"), [], "captions", "en")
    out = build_index(t, r, incremental=False)
    data = json.loads(out.read_text())
    assert data["videos"][0]["title"] == "V2"

def test_empty_transcripts_dir_returns_empty_index(tmp_path):
    t = tmp_path / "transcripts"
    t.mkdir()
    out = build_index(t, tmp_path / "references")
    data = json.loads(out.read_text())
    assert data == {"version": 1, "n_videos": 0, "videos": []}
