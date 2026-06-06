from __future__ import annotations
import http.cookiejar
from typing import Optional

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript


class NoCaptionsAvailable(Exception):
    def __init__(self, video_id: str):
        self.video_id = video_id
        super().__init__(video_id)


def _build_http_client(cookies: Optional[dict]) -> Optional[requests.Session]:
    """Build a requests.Session preloaded with cookies, when supplied.

    youtube-transcript-api accepts a custom `http_client` (requests.Session).
    We can only honor a Netscape-format cookies file here (`{"file": ...}`).
    For `{"browser": ...}` we cannot extract cookies without yt-dlp's
    browser-cookie code, so the caller will still benefit from yt-dlp fallbacks
    in Tier 2. We log nothing and silently fall back to an unauthenticated
    session in that case.
    """
    if not cookies or "file" not in cookies:
        return None
    jar = http.cookiejar.MozillaCookieJar(str(cookies["file"]))
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception:
        return None
    s = requests.Session()
    s.cookies = jar  # type: ignore[assignment]
    return s


def fetch_captions(
    video_id: str, cookies: Optional[dict] = None
) -> tuple[list[dict], str]:
    """Fetch captions for a YouTube video via youtube-transcript-api (Tier 1).

    Prefers manually-created captions over auto-generated; English first when
    language is unspecified.

    Args:
        video_id: YouTube video ID (not the full URL).
        cookies: optional dict from `_cookies.resolve_cookies()`. Only the
            `{"file": <path>}` shape is honored here (yt-tx-api lacks a browser-
            cookie loader); `{"browser": ...}` callers get an unauthenticated
            request and rely on the Tier 2 yt-dlp fallback for auth.

    Returns:
        A tuple of (segments, lang) where segments is a list of
        {start, duration, text} dicts and lang is the BCP-47 language code.

    Raises:
        NoCaptionsAvailable: if no captions exist for the video.
    """
    http_client = _build_http_client(cookies)
    try:
        api = YouTubeTranscriptApi(http_client=http_client) if http_client else YouTubeTranscriptApi()
        listing = api.list(video_id)
    except CouldNotRetrieveTranscript:
        raise NoCaptionsAvailable(video_id)

    # Partition into manual vs generated; fall back to iteration if private
    # attrs are absent (forward-compatibility guard).
    try:
        manual = list(listing._manually_created_transcripts.values())
        generated = list(listing._generated_transcripts.values())
    except AttributeError:
        all_t = list(listing)
        manual = [t for t in all_t if not t.is_generated]
        generated = [t for t in all_t if t.is_generated]

    candidates = manual + generated
    if not candidates:
        raise NoCaptionsAvailable(video_id)

    # English-first, then manual-before-generated within each language group.
    candidates.sort(
        key=lambda t: (
            0 if t.language_code.startswith("en") else 1,
            0 if not t.is_generated else 1,
        )
    )
    t = candidates[0]
    fetched = t.fetch()
    # FetchedTranscript.to_raw_data() returns list[dict] with text/start/duration keys.
    segs = fetched.to_raw_data()
    return segs, t.language_code
