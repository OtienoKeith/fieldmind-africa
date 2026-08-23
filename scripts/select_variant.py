#!/usr/bin/env python3
"""Join official profiler and FieldMind quality reports into a tournament."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def key(path: Path) -> str:
    name = path.stem.casefold()
    for suffix in ("-eval", "_eval", "-quality", "_quality", "-profiler", "_profiler"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def official_accuracy(report: dict) -> float | None:
    values = [float(row["score"]) * 100 for row in report.get("accuracy", []) if isinstance(row.get("score"), (int, float))]
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler-dir", type=Path, default=ROOT / "benchmarks/profiler")
    parser.add_argument("--quality-dir", type=Path, default=ROOT / "benchmarks/quality")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/tournament.csv")
    args = parser.parse_args()
    profiler = {key(p): json.loads(p.read_text(encoding="utf-8")) for p in args.profiler_dir.glob("*.json")}
    quality = {key(p): json.loads(p.read_text(encoding="utf-8")) for p in args.quality_dir.glob("*.json")}
    if not profiler:
        raise SystemExit(f"No profiler JSON files in {args.profiler_dir}")
    rows = []
    for name, report in profiler.items():
        tps = float(report["throughput"]["tokens_per_second_generation"])
        peak_gb = float(report["memory"]["peak_rss_mb"]) / 1024.0
        qscore = float(quality[name]["score_percent"]) if name in quality else None
        acc = official_accuracy(report)
        score_input = acc if acc is not None else qscore
        perf = min(tps / 15.0, 1.0) * 100
        efficiency = max(0.0, (7.0 - peak_gb) / 7.0) * 100
        thermal = 10.0 if report.get("cpu_thermal", {}).get("throttled") else 0.0
        composite = None if score_input is None else 0.50 * score_input + 0.30 * perf + 0.20 * efficiency - thermal
        rows.append({
            "variant": name,
            "quality_proxy": qscore,
            "official_accuracy": acc,
            "accuracy_used": "official" if acc is not None else ("fieldmind_proxy" if qscore is not None else "missing"),
            "tps": round(tps, 3),
            "peak_rss_gb": round(peak_gb, 3),
            "performance_score": round(perf, 3),
            "efficiency_score": round(efficiency, 3),
            "thermal_penalty": thermal,
            "estimated_core_score": None if composite is None else round(composite, 3),
        })
    rows.sort(key=lambda r: (-1 if r["estimated_core_score"] is None else -r["estimated_core_score"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
