from datetime import date
from youtube_advisor.filters import FilterConfig, apply_filters

def vid(id_, title="x", published="2024-01-01"):
    return {"video_id": id_, "title": title, "published_date": published, "playlist_ids": set()}

def test_date_range_inclusive():
    candidates = [vid("a", published="2022-01-01"), vid("b", published="2024-06-01"), vid("c", published="2026-01-01")]
    cfg = FilterConfig(since=date(2024,1,1), until=date(2025,12,31))
    out = apply_filters(candidates, cfg)
    assert [v["video_id"] for v in out] == ["b"]

def test_max_keeps_newest():
    candidates = [vid(c, published=f"2024-{i+1:02d}-01") for i, c in enumerate(["a","b","c","d"])]
    cfg = FilterConfig(max=2)
    out = apply_filters(candidates, cfg)
    assert [v["video_id"] for v in out] == ["d", "c"]

def test_title_include_excludes_others():
    candidates = [vid("a", title="How to hire"), vid("b", title="random talk")]
    cfg = FilterConfig(title_include=r"hire|hiring")
    out = apply_filters(candidates, cfg)
    assert [v["video_id"] for v in out] == ["a"]

def test_title_exclude_default_shorts():
    candidates = [vid("a", title="Talk #shorts"), vid("b", title="Talk")]
    cfg = FilterConfig()  # default excludes shorts
    out = apply_filters(candidates, cfg)
    assert [v["video_id"] for v in out] == ["b"]

def test_playlist_intersection():
    candidates = [vid("a"), vid("b")]
    candidates[0]["playlist_ids"] = {"PLfoo"}
    candidates[1]["playlist_ids"] = {"PLbar"}
    cfg = FilterConfig(playlist_ids={"PLfoo"})
    out = apply_filters(candidates, cfg)
    assert [v["video_id"] for v in out] == ["a"]

def test_manual_ids_bypass_other_filters():
    candidates = [vid("a", published="2010-01-01"), vid("b", title="x #shorts")]
    cfg = FilterConfig(since=date(2024,1,1), manual_ids={"a", "b"})
    out = apply_filters(candidates, cfg)
    assert sorted([v["video_id"] for v in out]) == ["a", "b"]


def test_invalid_title_include_regex_falls_back_to_literal(caplog):
    """Bug #8: user-supplied invalid regex must not crash apply_filters.
    It falls back to literal substring matching (case-insensitive)."""
    candidates = [vid("a", title="(unbalanced talk about something"),
                  vid("b", title="Plain Python tip")]
    cfg = FilterConfig(title_include="(unbalanced")  # invalid: unmatched (
    out = apply_filters(candidates, cfg)
    assert [v["video_id"] for v in out] == ["a"]


def test_invalid_title_exclude_regex_falls_back_to_literal():
    candidates = [vid("a", title="Plain talk *foo"),
                  vid("b", title="Plain talk")]
    cfg = FilterConfig(title_exclude="*foo")  # invalid regex
    out = apply_filters(candidates, cfg)
    assert [v["video_id"] for v in out] == ["b"]


def test_malformed_published_date_drops_video_silently(caplog):
    """Bug #9: a malformed published_date must not crash the run — drop the
    video with a logged warning instead."""
    candidates = [
        vid("a", published="2024-06-01"),
        {"video_id": "b", "title": "bad", "published_date": "not-a-date",
         "playlist_ids": set()},
        {"video_id": "c", "title": "missing", "playlist_ids": set()},  # no key
    ]
    out = apply_filters(candidates, FilterConfig())
    assert [v["video_id"] for v in out] == ["a"]
