#!/usr/bin/env python3
"""Validate schema, provenance, formatting, and train/eval isolation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_HEADINGS = (
    "BEFORE SPENDING MONEY",
    "CONFIDENCE",
)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AssertionError(f"{path}:{number}: invalid JSON: {exc}") from exc


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


def qhash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def validate_training(path: Path) -> set[str]:
    hashes: set[str] = set()
    ids: set[str] = set()
    for number, row in iter_jsonl(path):
        assert isinstance(row.get("id"), str) and row["id"], f"{path}:{number}: missing id"
        assert row["id"] not in ids, f"{path}:{number}: duplicate id {row['id']}"
        ids.add(row["id"])
        messages = row.get("messages")
        assert isinstance(messages, list) and len(messages) == 3, f"{path}:{number}: messages must contain 3 items"
        assert [m.get("role") for m in messages] == ["system", "user", "assistant"], f"{path}:{number}: wrong roles"
        assert all(isinstance(m.get("content"), str) and m["content"].strip() for m in messages), f"{path}:{number}: empty content"
        answer = messages[2]["content"].upper()
        if row.get("metadata", {}).get("language") == "en":
            for heading in REQUIRED_HEADINGS:
                assert heading in answer, f"{path}:{number}: missing {heading}"
        metadata = row.get("metadata", {})
        for key in ("country", "crop", "language", "source", "license", "question_sha256"):
            assert metadata.get(key), f"{path}:{number}: missing metadata.{key}"
        computed = qhash(messages[1]["content"])
        assert metadata["question_sha256"] == computed, f"{path}:{number}: wrong question hash"
        hashes.add(computed)
    return hashes


def validate_eval(eval_dir: Path) -> set[str]:
    hashes: set[str] = set()
    ids: set[str] = set()
    files = sorted(eval_dir.glob("*.jsonl"))
    assert len(files) >= 3, "Expected at least three evaluation sets"
    for path in files:
        for number, row in iter_jsonl(path):
            assert row.get("id") and row["id"] not in ids, f"{path}:{number}: duplicate/missing id"
            ids.add(row["id"])
            question = row.get("question")
            assert isinstance(question, str) and len(question) >= 20, f"{path}:{number}: bad question"
            groups = row.get("required_groups")
            assert isinstance(groups, list) and groups, f"{path}:{number}: no required groups"
            assert all(isinstance(g, list) and g and all(isinstance(x, str) and x for x in g) for g in groups), f"{path}:{number}: bad required group"
            assert isinstance(row.get("forbidden_terms"), list), f"{path}:{number}: forbidden_terms must be a list"
            hashes.add(qhash(question))
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "data/eval")
    args = parser.parse_args()
    train_path = args.processed_dir / "train.jsonl"
    validation_path = args.processed_dir / "validation.jsonl"
    assert train_path.exists(), f"Missing {train_path}; run prepare_dataset.py first"
    assert validation_path.exists(), f"Missing {validation_path}; run prepare_dataset.py first"
    train = validate_training(train_path)
    validation = validate_training(validation_path)
    eval_hashes = validate_eval(args.eval_dir)
    assert not (train & validation), "Train/validation question leakage detected"
    assert not ((train | validation) & eval_hashes), "Training/evaluation question leakage detected"
    print(json.dumps({
        "train_unique_questions": len(train),
        "validation_unique_questions": len(validation),
        "eval_unique_questions": len(eval_hashes),
        "leakage": 0,
        "status": "ok",
    }, indent=2))


if __name__ == "__main__":
    main()
