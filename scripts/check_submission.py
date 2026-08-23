#!/usr/bin/env python3
"""Fail-fast validator for the final ADTC repository contract."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER_RE = re.compile(r"replace|example\.com|tbd|your-|after_training", re.IGNORECASE)
DOMAINS = {
    "math_scientific_reasoning", "healthcare_medical", "agriculture", "creative_writing",
    "coding_assistants", "corporate_enterprise", "autonomous_ai_agents",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-placeholders", action="store_true")
    parser.add_argument("--allow-missing-model", action="store_true")
    args = parser.parse_args()
    errors = []
    metadata_path = ROOT / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Invalid metadata.json: {exc}") from exc
    if metadata.get("domain") not in DOMAINS:
        errors.append("metadata.domain is invalid")
    if len(metadata.get("test_prompts", [])) != 2:
        errors.append("metadata.test_prompts must contain exactly two items")
    if metadata.get("model", {}).get("runtime") != "llama.cpp":
        errors.append("metadata.model.runtime must be llama.cpp")
    if metadata.get("model", {}).get("packaging") != "binary_bundle":
        errors.append("metadata.model.packaging must be binary_bundle")
    if not metadata.get("language_scope"):
        errors.append("metadata.language_scope cannot be empty")
    if not args.allow_placeholders and PLACEHOLDER_RE.search(json.dumps(metadata)):
        errors.append("metadata.json still contains owner placeholders")
    model_rel = metadata.get("_runtime", {}).get("model_path", "")
    model_path = ROOT / model_rel
    if not model_rel.startswith("model/") or not model_rel.endswith(".gguf"):
        errors.append("_runtime.model_path must be model/*.gguf")
    if not args.allow_missing_model:
        if not model_path.exists():
            errors.append(f"model missing: {model_rel}")
        else:
            with model_path.open("rb") as handle:
                if handle.read(4) != b"GGUF":
                    errors.append(f"model does not start with GGUF magic bytes: {model_rel}")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    if "*.gguf" not in gitignore or "model/" not in gitignore:
        errors.append(".gitignore must exclude *.gguf and model/")
    required = ["README.md", "REPORT.md", "download_model.sh", "LICENSE"]
    for name in required:
        if not (ROOT / name).exists():
            errors.append(f"missing required file: {name}")
    downloader = (ROOT / "download_model.sh").read_text(encoding="utf-8")
    if not args.allow_placeholders and ("REPLACE_WITH_PUBLIC" in downloader or "REPLACE_WITH_FINAL" in downloader):
        errors.append("download_model.sh still contains final URL/checksum placeholders")
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    if not args.allow_placeholders and ("TBD — measure" in report or "REPLACE_AFTER_TRAINING" in report):
        errors.append("REPORT.md still contains unmeasured placeholders")
    if errors:
        print("SUBMISSION CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("SUBMISSION CHECK PASSED")


if __name__ == "__main__":
    main()
