#!/usr/bin/env bash
# Usage: bash scripts/status.sh <advisor-dir>
# Prints a one-line status read from <advisor>/.progress.json. Silent if no progress file.
set -euo pipefail
ADV="${1:-}"
[ -z "$ADV" ] && { echo "Usage: status.sh <advisor-dir>" >&2; exit 1; }
P="$ADV/.progress.json"
[ ! -f "$P" ] && exit 0
python3 - "$P" <<'PY'
import json, sys
try:
    p = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
stage = p.get("stage", "?")
done = p.get("videos_done", 0)
total = p.get("videos_total", 0)
elapsed = p.get("elapsed_sec", 0) or 0
eta = p.get("eta_sec")
msg = p.get("message") or ""
mm = f"{elapsed//60:02d}:{elapsed%60:02d}"
parts = [f"\U0001F3AC {stage}"]
if total:
    parts.append(f"{done}/{total}")
parts.append(mm)
if eta is not None and eta > 0:
    parts.append(f"ETA {eta//60:02d}:{eta%60:02d}")
if msg and stage in {"init", "resolving", "filtering", "drafting", "evals", "done"}:
    parts.append("· " + msg[:60])
print(" · ".join(parts))
PY
