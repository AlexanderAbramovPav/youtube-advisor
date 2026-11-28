from datetime import date
import logging
import pytest
from youtube_advisor.filters import (
    FilterConfig,
    apply_filters,
    duration_from_saved,
    format_duration,
    format_duration_bounds,
    parse_duration,
)

def vid(id_, title="x", published="2024-01-01", length_seconds=600):
    d = {"video_id": id_, "title": title, "published_date": published, "playlist_ids": set()}
    if length_seconds is not None:
        d["length_seconds"] = length_seconds
    return d

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


# ----- duration parser & formatter -----

@pytest.mark.parametrize("raw,expected", [
    ("0", 0),
    ("3600", 3600),
    ("  90m  ", 5400),
    ("90m", 5400),
    ("2h", 7200),
    ("1h30m", 5400),
    ("1H30M", 5400),
    ("45s", 45),
    ("1h30m15s", 5415),
    ("1h5s", 3605),
    ("05:30", 330),
    ("5:30", 330),
    ("00:45", 45),
    ("90:00", 5400),
    ("1:30:00", 5400),
    ("01:02:03", 3723),
])
def test_parse_duration_valid(raw, expected):
    assert parse_duration(raw) == expected


@pytest.mark.parametrize("raw", [
    "",
    "   ",
    "-1",
    "-10m",
    "-1:00",
    "+10",
    "abc",
    "1d",
    "1.5h",
    "1.5",
    "60x",
    "30m1h",
    "1m1m",
    "1:60",
    "1:60:00",
    "1:00:60",
    "1:5",
    "1:5:00",
    ":30",
    "10:",
    "1:02:03:04",
    "h",
    "m",
    "s",
    "1h 30m",
])
def test_parse_duration_invalid(raw):
    with pytest.raises(ValueError):
        parse_duration(raw)


def test_format_duration_and_bounds():
    assert format_duration(0) == "0s"
    assert format_duration(45) == "45s"
    assert format_duration(1200) == "20m"
    assert format_duration(3600) == "1h"
    assert format_duration(5400) == "1h30m"
    assert format_duration(3665) == "1h1m5s"
    assert format_duration_bounds(None, None) is None
    assert format_duration_bounds(3600, None) == "≥ 1h"
    assert format_duration_bounds(None, 1200) == "≤ 20m"
    assert format_duration_bounds(1200, 3600) == "≥ 20m, ≤ 1h"


# ----- duration filtering -----

def test_duration_bounds_inclusive_edges():
    candidates = [
        vid("too_short", length_seconds=599),
        vid("at_min", length_seconds=600),
        vid("middle", length_seconds=1800),
        vid("at_max", length_seconds=3600),
        vid("too_long", length_seconds=3601),
    ]
    out_min = apply_filters(candidates, FilterConfig(min_duration=600))
    assert [v["video_id"] for v in out_min] == ["at_min", "middle", "at_max", "too_long"]

    out_max = apply_filters(candidates, FilterConfig(max_duration=3600))
    assert [v["video_id"] for v in out_max] == ["too_short", "at_min", "middle", "at_max"]

    out_both = apply_filters(candidates, FilterConfig(min_duration=600, max_duration=3600))
    assert [v["video_id"] for v in out_both] == ["at_min", "middle", "at_max"]


def test_unknown_length_kept_and_counted_in_single_warning(caplog):
    candidates = [
        vid("missing", length_seconds=None),
        vid("zero", length_seconds=0),
        vid("not_num", length_seconds="3600"),
        vid("bool_val", length_seconds=True),
        vid("short_drop", length_seconds=120),
        vid("long_keep", length_seconds=4000),
    ]
    with caplog.at_level(logging.WARNING, logger="youtube_advisor.filters"):
        out = apply_filters(candidates, FilterConfig(min_duration=3600))
    assert [v["video_id"] for v in out] == ["missing", "zero", "not_num", "bool_val", "long_keep"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "4" in warnings[0].getMessage()


def test_unknown_length_no_warning_when_no_duration_bound(caplog):
    candidates = [vid("missing", length_seconds=None), vid("zero", length_seconds=0)]
    with caplog.at_level(logging.WARNING, logger="youtube_advisor.filters"):
        out = apply_filters(candidates, FilterConfig())
    assert [v["video_id"] for v in out] == ["missing", "zero"]
    assert not caplog.records


def test_manual_ids_bypass_duration_bounds():
    candidates = [
        vid("short_manual", length_seconds=60),
        vid("long_manual", length_seconds=9999),
        vid("short_drop", length_seconds=60),
        vid("in_range", length_seconds=1800),
    ]
    cfg = FilterConfig(min_duration=600, max_duration=3600, manual_ids={"short_manual", "long_manual"})
    out = apply_filters(candidates, cfg)
    assert [v["video_id"] for v in out] == ["in_range", "short_manual", "long_manual"]


def test_duration_filter_runs_before_max_cut():
    candidates = [
        vid("new_short_1", published="2025-05-01", length_seconds=300),
        vid("new_short_2", published="2025-04-01", length_seconds=400),
        vid("old_long_1", published="2025-03-01", length_seconds=4000),
        vid("old_long_2", published="2025-02-01", length_seconds=5000),
        vid("older_long_3", published="2025-01-01", length_seconds=6000),
    ]
    cfg = FilterConfig(min_duration=3600, max=2)
    out = apply_filters(candidates, cfg)
    assert [v["video_id"] for v in out] == ["old_long_1", "old_long_2"]


def test_unknown_length_warning_counts_only_videos_kept_after_max(caplog):
    candidates = [vid(f"u{i}", published=f"2024-01-{i + 1:02d}", length_seconds=None)
                  for i in range(5)]
    candidates.append(vid("long_new", published="2024-02-01", length_seconds=4000))
    with caplog.at_level(logging.WARNING, logger="youtube_advisor.filters"):
        out = apply_filters(candidates, FilterConfig(min_duration=3600, max=3))
    assert [v["video_id"] for v in out] == ["long_new", "u4", "u3"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Kept 2 video(s)" in warnings[0].getMessage()


@pytest.mark.parametrize("value,expected", [
    (None, None), (0, 0), (3600, 3600), (3600.0, 3600),
    ("1h", 3600), ("90m", 5400), ("1:30:00", 5400), ("600", 600),
])
def test_duration_from_saved_accepts_seconds_and_cli_forms(value, expected):
    assert duration_from_saved(value) == expected


@pytest.mark.parametrize("value", [-1, True, "soon", 1.5, [3600]])
def test_duration_from_saved_rejects_bad_values(value):
    with pytest.raises(ValueError):
        duration_from_saved(value)
