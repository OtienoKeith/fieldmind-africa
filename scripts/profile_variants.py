#!/usr/bin/env python3
"""Run the official ADTC profiler consistently over multiple GGUF variants."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link_model(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        try:
            destination.symlink_to(source.resolve())
        except OSError as exc:
            raise RuntimeError(
                "Could not hard-link or symlink the GGUF into a temporary submission. "
                "Run on Linux/Kaggle or place variants on the same filesystem."
            ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks/profiler")
    parser.add_argument("--full-accuracy", action="store_true", help="Do not pass --skip-accuracy")
    parser.add_argument("--profiler", default="adtc-profiler")
    args = parser.parse_args()
    if not (Path(args.profiler).exists() or shutil.which(args.profiler)):
        raise SystemExit("adtc-profiler not found; install the official profiler first")
    base_metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        model = model.resolve()
        if not model.exists():
            raise SystemExit(f"Missing model: {model}")
        with tempfile.TemporaryDirectory(prefix="fieldmind-profiler-") as temp_name:
            temp = Path(temp_name)
            (temp / "model").mkdir()
            linked = temp / "model" / model.name
            link_model(model, linked)
            metadata = json.loads(json.dumps(base_metadata))
            metadata["_runtime"]["model_path"] = f"model/{model.name}"
            metadata["model"]["name"] = model.stem
            (temp / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            output = args.output_dir.resolve() / f"{model.stem}.json"
            cmd = [args.profiler, "run", "--submission", str(temp), "--mode", "participant", "--output", str(output)]
            if not args.full_accuracy:
                cmd.append("--skip-accuracy")
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
    print(f"Profiler reports: {args.output_dir}")


if __name__ == "__main__":
    main()
