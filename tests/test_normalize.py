import yaml
import pathlib
from youtube_advisor.ingest.normalize import write_transcript, _hms


def test_hms_formatting():
    assert _hms(0) == "00:00:00"
    assert _hms(65) == "00:01:05"
    assert _hms(3661) == "01:01:01"
    assert _hms(3.7) == "00:00:03"  # truncates fractional seconds


def test_writes_atomic_md(tmp_path):
    meta = {
        "video_id": "abc", "title": "T", "published_date": "2024-06-01",
        "channel": "@x", "channel_id": "UCx", "length_seconds": 100, "views": 10,
        "description": "d", "playlist_ids": set(),
    }
    segs = [{"start": 0, "duration": 2, "text": "Hi"}, {"start": 12, "duration": 3, "text": "there"}]
    p = write_transcript(tmp_path, meta, segs, "captions", "en")
    assert p.name == "2024-06-01-abc.md"
    text = p.read_text()
    fm_text, body = text.split("---\n", 2)[1:]
    data = yaml.safe_load(fm_text)
    assert data["video_id"] == "abc"
    assert data["transcript_source"] == "captions"
    assert data["transcript_language"] == "en"
    assert data["video_url"] == "https://www.youtube.com/watch?v=abc"
    assert data["length_minutes"] == 1
    assert "[00:00:00] Hi" in body
    assert "[00:00:12] there" in body


def test_skips_segments_with_empty_text(tmp_path):
    meta = {"video_id": "id1", "title": "T", "published_date": "2024-01-01"}
    segs = [
        {"start": 0, "duration": 1, "text": "first"},
        {"start": 5, "duration": 1, "text": ""},
        {"start": 10, "duration": 1, "text": "third"},
    ]
    p = write_transcript(tmp_path, meta, segs, "captions", "en")
    body = p.read_text().split("---\n", 2)[2]
    assert "[00:00:00] first" in body
    assert "[00:00:10] third" in body
    assert "[00:00:05]" not in body


def test_description_truncated_to_500_chars(tmp_path):
    meta = {"video_id": "id2", "title": "T", "published_date": "2024-01-01",
            "description": "x" * 1000}
    p = write_transcript(tmp_path, meta, [], "captions", "en")
    fm_text = p.read_text().split("---\n", 2)[1]
    data = yaml.safe_load(fm_text)
    assert len(data["description"]) == 500


def test_unicode_in_title_and_body(tmp_path):
    meta = {"video_id": "id3", "title": "Привет мир", "published_date": "2024-01-01"}
    segs = [{"start": 0, "duration": 1, "text": "Здравствуй"}]
    p = write_transcript(tmp_path, meta, segs, "whisper-v3", "ru")
    text = p.read_text()
    assert "Привет мир" in text
    assert "Здравствуй" in text


def test_atomic_write_uses_tmpfile(tmp_path, monkeypatch):
    """Verify the .tmp- prefix files are not left over after success."""
    meta = {"video_id": "id4", "title": "T", "published_date": "2024-01-01"}
    write_transcript(tmp_path, meta, [], "captions", "en")
    tmp_files = list(tmp_path.glob(".tmp-*"))
    assert tmp_files == []  # cleaned up by os.replace


def test_returns_path(tmp_path):
    meta = {"video_id": "abc", "title": "T", "published_date": "2024-06-01"}
    p = write_transcript(tmp_path, meta, [], "captions", "en")
    assert isinstance(p, pathlib.Path)
    assert p.exists()


def test_creates_out_dir_if_missing(tmp_path):
    out = tmp_path / "nested" / "transcripts"
    assert not out.exists()
    meta = {"video_id": "abc", "title": "T", "published_date": "2024-06-01"}
    p = write_transcript(out, meta, [], "captions", "en")
    assert p.parent == out
    assert p.exists()
