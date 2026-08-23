from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_metadata_contract() -> None:
    data = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    assert data["domain"] == "agriculture"
    assert len(data["test_prompts"]) == 2
    assert data["model"]["runtime"] == "llama.cpp"
    assert data["model"]["packaging"] == "binary_bundle"
    assert data["_runtime"]["model_path"].startswith("model/")
    assert data["_runtime"]["model_path"].endswith(".gguf")


def test_eval_suites_are_distinct_and_well_formed() -> None:
    files = sorted((ROOT / "data/eval").glob("*.jsonl"))
    assert len(files) == 3
    rows = [row for path in files for row in load_jsonl(path)]
    assert len(rows) >= 24
    assert len({row["id"] for row in rows}) == len(rows)
    questions = [" ".join(row["question"].casefold().split()) for row in rows]
    assert len(set(questions)) == len(questions)
    assert all(row["required_groups"] for row in rows)
    assert all(isinstance(row["forbidden_terms"], list) for row in rows)


def test_seed_questions_do_not_copy_eval_questions() -> None:
    seeds = load_jsonl(ROOT / "data/seeds/fieldmind_behavior.jsonl")
    eval_rows = [row for path in (ROOT / "data/eval").glob("*.jsonl") for row in load_jsonl(path)]
    def digest(text: str) -> str:
        return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()

    assert {digest(r["question"]) for r in seeds}.isdisjoint({digest(r["question"]) for r in eval_rows})


def test_notebooks_are_valid_v4_json() -> None:
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        assert all(cell["cell_type"] in {"markdown", "code"} for cell in notebook["cells"])


def test_offline_seed_pipeline(tmp_path: Path) -> None:
    output = tmp_path / "processed"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_dataset.py"), "--offline-seed-only", "--output-dir", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "config/dataset.json").read_text(encoding="utf-8"))
    seeds = load_jsonl(ROOT / "data/seeds/fieldmind_behavior.jsonl")
    assert manifest["mode"] == "offline_seed_only"
    assert manifest["counts"]["train"] == len(seeds) * cfg["seed_upsample"]
    assert manifest["counts"]["validation"] == 0
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_data.py"), "--processed-dir", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_submission_draft_mode_passes() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_submission.py"), "--allow-placeholders", "--allow-missing-model"],
        check=True,
        capture_output=True,
        text=True,
    )
