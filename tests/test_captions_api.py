import pytest
from unittest.mock import patch, MagicMock
from youtube_advisor.ingest.captions_api import fetch_captions, NoCaptionsAvailable
from youtube_transcript_api._errors import TranscriptsDisabled


def _fake_transcript(lang_code: str, is_generated: bool, segs: list[dict]):
    t = MagicMock()
    t.language_code = lang_code
    t.is_generated = is_generated
    # fetch() returns a FetchedTranscript-like object; to_raw_data() returns list[dict]
    fetched = MagicMock()
    fetched.to_raw_data.return_value = segs
    t.fetch.return_value = fetched
    return t


def _fake_listing(manual: dict, generated: dict):
    listing = MagicMock()
    listing._manually_created_transcripts = manual
    listing._generated_transcripts = generated
    # Make it iterable too, in case impl falls back to iteration
    listing.__iter__ = lambda self: iter(
        list(manual.values()) + list(generated.values())
    )
    return listing


def test_prefers_manual_english_over_generated():
    segs_manual = [{"text": "manual hello", "start": 0.0, "duration": 1.0}]
    segs_auto = [{"text": "auto hello", "start": 0.0, "duration": 1.0}]
    listing = _fake_listing(
        manual={"en": _fake_transcript("en", False, segs_manual)},
        generated={"en": _fake_transcript("en", True, segs_auto)},
    )
    with patch(
        "youtube_advisor.ingest.captions_api.YouTubeTranscriptApi",
    ) as MockApi:
        MockApi.return_value.list.return_value = listing
        out, lang = fetch_captions("Th8JoIan4dg")
    assert lang == "en"
    assert out[0]["text"] == "manual hello"


def test_falls_back_to_generated_when_no_manual():
    segs = [{"text": "auto hi", "start": 0.0, "duration": 1.0}]
    listing = _fake_listing(
        manual={},
        generated={"en": _fake_transcript("en", True, segs)},
    )
    with patch(
        "youtube_advisor.ingest.captions_api.YouTubeTranscriptApi",
    ) as MockApi:
        MockApi.return_value.list.return_value = listing
        out, lang = fetch_captions("vid1")
    assert lang == "en"
    assert out == segs


def test_prefers_english_over_other_languages():
    segs_es = [{"text": "hola", "start": 0.0, "duration": 1.0}]
    segs_en = [{"text": "hi", "start": 0.0, "duration": 1.0}]
    listing = _fake_listing(
        manual={
            "es": _fake_transcript("es", False, segs_es),
            "en": _fake_transcript("en", False, segs_en),
        },
        generated={},
    )
    with patch(
        "youtube_advisor.ingest.captions_api.YouTubeTranscriptApi",
    ) as MockApi:
        MockApi.return_value.list.return_value = listing
        out, lang = fetch_captions("vid2")
    assert lang == "en"


def test_no_captions_raises_when_listing_empty():
    listing = _fake_listing(manual={}, generated={})
    with patch(
        "youtube_advisor.ingest.captions_api.YouTubeTranscriptApi",
    ) as MockApi:
        MockApi.return_value.list.return_value = listing
        with pytest.raises(NoCaptionsAvailable) as exc_info:
            fetch_captions("vid_no_captions")
        assert exc_info.value.video_id == "vid_no_captions"


def test_no_captions_raises_when_transcripts_disabled():
    with patch(
        "youtube_advisor.ingest.captions_api.YouTubeTranscriptApi",
    ) as MockApi:
        MockApi.return_value.list.side_effect = TranscriptsDisabled("vid")
        with pytest.raises(NoCaptionsAvailable):
            fetch_captions("vid")
