"""Atomic progress.json writer for live status tracking.

Writes to <advisor>/.progress.json every time a stage changes or counters bump.
Readers (status.sh, the AI agent peeking, the shell pane) tail the file.
"""
from __future__ import annotations
import json
import pathlib
import time
from datetime import datetime, timezone

from ._md import atomic_write_text


class Progress:
    def __init__(self, advisor_dir: pathlib.Path):
        self.path = advisor_dir / ".progress.json"
        self.state = {
            "stage": "init",
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stage_started_at": time.time(),
            "elapsed_sec": 0,
            "stage_sec": 0,
            "videos_total": 0,
            "videos_done": 0,
            "by_source": {},
            "current_video": None,
            "current_video_title": None,
            "eta_sec": None,
            "message": None,
        }
        self._start_ts = time.time()
        self._write()

    def stage(self, name: str, *, message: str | None = None) -> None:
        self.state["stage"] = name
        self.state["stage_started_at"] = time.time()
        self.state["stage_sec"] = 0
        if message is not None:
            self.state["message"] = message
        self._write()

    def videos(self, total: int) -> None:
        self.state["videos_total"] = total
        self._write()

    def tick(self, video_id: str | None = None, title: str | None = None,
             source: str | None = None) -> None:
        if video_id:
            self.state["videos_done"] += 1
            self.state["current_video"] = video_id
            self.state["current_video_title"] = title
        if source:
            self.state["by_source"][source] = self.state["by_source"].get(source, 0) + 1
        self._write()

    def done(self, message: str = "Done.") -> None:
        self.state["stage"] = "done"
        self.state["message"] = message
        self._write()

    def _write(self) -> None:
        now = time.time()
        self.state["elapsed_sec"] = int(now - self._start_ts)
        self.state["stage_sec"] = int(now - self.state["stage_started_at"])
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # ETA: very rough — assume per-video time stays constant
        done = self.state["videos_done"]
        total = self.state["videos_total"]
        if done > 0 and total > done and self.state["stage"] == "ingesting":
            per = (now - self._start_ts) / done
            self.state["eta_sec"] = int(per * (total - done))
        elif self.state["stage"] in {"done", "init"}:
            self.state["eta_sec"] = 0 if self.state["stage"] == "done" else None
        # Atomic write via the shared helper — cleans up the temp file on
        # any write/replace failure (no .progress.*.tmp debris on crashes).
        atomic_write_text(self.path, json.dumps(self.state, indent=2))
