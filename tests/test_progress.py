"""Tests for the atomic progress.json writer."""
from __future__ import annotations
import json
import pathlib

from youtube_advisor._progress import Progress


def test_init_writes_progress_file(tmp_path: pathlib.Path):
    p = Progress(tmp_path)
    assert (tmp_path / ".progress.json").exists()
    state = json.loads((tmp_path / ".progress.json").read_text())
    assert state["stage"] == "init"
    assert state["videos_done"] == 0


def test_stage_updates_message_and_resets_stage_timer(tmp_path: pathlib.Path):
    p = Progress(tmp_path)
    p.stage("ingesting", message="hi")
    state = json.loads((tmp_path / ".progress.json").read_text())
    assert state["stage"] == "ingesting"
    assert state["message"] == "hi"
    assert state["stage_sec"] >= 0


def test_videos_sets_total(tmp_path: pathlib.Path):
    p = Progress(tmp_path)
    p.videos(42)
    state = json.loads((tmp_path / ".progress.json").read_text())
    assert state["videos_total"] == 42


def test_tick_increments_done_and_by_source(tmp_path: pathlib.Path):
    p = Progress(tmp_path)
    p.videos(3)
    p.stage("ingesting")
    p.tick(video_id="v1", title="T1", source="captions")
    p.tick(video_id="v2", title="T2", source="captions")
    p.tick(video_id="v3", title="T3", source="whisper-v3")
    state = json.loads((tmp_path / ".progress.json").read_text())
    assert state["videos_done"] == 3
    assert state["current_video"] == "v3"
    assert state["current_video_title"] == "T3"
    assert state["by_source"] == {"captions": 2, "whisper-v3": 1}


def test_done_sets_stage_done_and_zero_eta(tmp_path: pathlib.Path):
    p = Progress(tmp_path)
    p.done(message="ok")
    state = json.loads((tmp_path / ".progress.json").read_text())
    assert state["stage"] == "done"
    assert state["message"] == "ok"
    assert state["eta_sec"] == 0


def test_atomic_write_no_leftover_tmp(tmp_path: pathlib.Path):
    """After several writes, the directory must contain only .progress.json
    (no stray .progress.*.tmp files left behind by mkstemp/replace)."""
    p = Progress(tmp_path)
    p.stage("filtering")
    p.videos(5)
    p.stage("ingesting")
    p.tick(video_id="a", source="captions")
    p.done()
    leftover = [x.name for x in tmp_path.iterdir()
                if x.name.startswith(".progress.") and x.name != ".progress.json"]
    assert leftover == []
