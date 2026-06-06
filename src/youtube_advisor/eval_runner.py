from __future__ import annotations
import json
import pathlib
from datetime import date

from .search import search


def run(advisor_dir: pathlib.Path, with_llm: bool = False) -> dict:
    """Run the benchmark in ``advisor_dir/evals/benchmark.json``.

    For each question, search the corpus (keyword mode, top 5) and check:
      - ``id_ok``: any expected_video_ids intersect the returned video IDs.
      - ``kw_ok``: at least one expected_keywords substring appears in the
        concatenated snippets (case-insensitive). If no keywords are given,
        kw_ok is True.

    Writes ``evals/results/YYYY-MM-DD.json`` and computes ``delta_passed``
    versus the most recent previous result file (if any).

    ``with_llm`` is accepted for forward-compat but unused in v1.
    """
    bench_path = advisor_dir / "evals" / "benchmark.json"
    bench = json.loads(bench_path.read_text())

    # If gen_benchmark wrote a `.benchmark_failed` marker (LLM produced
    # candidates but none validated), short-circuit with delta_passed=None
    # so callers don't confuse "no benchmark" with "all questions passed".
    failed_marker = advisor_dir / "evals" / ".benchmark_failed"
    if failed_marker.exists():
        return {"total": 0, "passed": 0, "results": [], "delta_passed": None,
                "benchmark_failed": True}

    results: list[dict] = []
    for q in bench.get("questions", []):
        hits = search(advisor_dir, q["q"], mode="keyword", top_k=5)
        hit_ids = {h["video_id"] for h in hits}
        snippets = " ".join(h["snippet"] for h in hits).lower()

        expected_ids = set(q.get("expected_video_ids", []))
        id_ok = bool(hit_ids & expected_ids) if expected_ids else True

        expected_kws = q.get("expected_keywords") or []
        if expected_kws:
            kw_ok = any(k.lower() in snippets for k in expected_kws)
        else:
            kw_ok = True

        passed = bool(id_ok and kw_ok)
        results.append({
            "q": q["q"],
            "passed": passed,
            "id_ok": bool(id_ok),
            "kw_ok": bool(kw_ok),
            "top_ids": sorted(hit_ids),
        })

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "results": results,
    }

    out_dir = advisor_dir / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    today_path = out_dir / f"{today}.json"

    prev_files = sorted(
        p for p in out_dir.glob("*.json") if p.stem != today
    )
    if prev_files:
        prev = json.loads(prev_files[-1].read_text())
        summary["delta_passed"] = summary["passed"] - prev.get("passed", 0)
    else:
        summary["delta_passed"] = None

    today_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return summary
