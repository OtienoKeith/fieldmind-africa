#!/usr/bin/env python3
"""Build deterministic FieldMind train/validation JSONL from open data.

The default path downloads CC-BY-4.0 FarmerChat configurations through the
Hugging Face datasets library. --offline-seed-only is a fast no-network smoke
mode; it is not sufficient for the final fine-tune.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = """You are FieldMind Africa, an offline agricultural decision-support assistant for extension officers, cooperatives, and input shops serving smallholder farmers.

Separate observations from possible causes. Do not pretend a text description proves one diagnosis. Ask for the few field checks that best separate the possibilities. Prefer low-cost, reversible actions. Do not recommend a pesticide, fungicide, veterinary medicine, or exact dose when the crop, symptoms, product label, registration, growth stage, or local context is unclear. For urgent animal distress, suspected poisoning, or rapidly spreading serious disease, direct the user to a trained local professional promptly.

Use these headings in the user's language: WHAT MAY BE HAPPENING; OTHER POSSIBILITIES; CHECK BEFORE ACTING; LOWEST-COST ACTION; BEFORE SPENDING MONEY; CONFIDENCE. Keep the answer practical and concise."""

REDACTION_MARKERS = ("[xxx]", "[redacted]", "<redacted>", "phone number", "email address")
HIGH_RISK_QUERY = re.compile(
    r"\b(exact dose|dosage|which pesticide|which fungicide|what chemical|"
    r"how much pesticide|how much herbicide|mixing rate)\b",
    re.IGNORECASE,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


def stable_fraction(text: str) -> float:
    value = int(hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()[:16], 16)
    return value / float(0xFFFFFFFFFFFFFFFF)


def is_acceptable(query: str, response: str, cfg: dict) -> tuple[bool, str]:
    q, a = query.strip(), response.strip()
    if len(q) < cfg["min_query_chars"]:
        return False, "short_query"
    if len(a) < cfg["min_response_chars"]:
        return False, "short_response"
    if len(a) > cfg["max_response_chars"]:
        return False, "long_response"
    joined = f"{q}\n{a}".casefold()
    if any(marker in joined for marker in REDACTION_MARKERS):
        return False, "redaction_marker"
    if HIGH_RISK_QUERY.search(q):
        return False, "high_risk_dose_query"
    if "http://" in joined or "https://" in joined:
        return False, "external_link"
    return True, "accepted"


def wrap_source_answer(answer: str) -> str:
    clean = re.sub(r"\s+", " ", answer).strip()
    return (
        "WHAT MAY BE HAPPENING\n"
        f"{clean}\n\n"
        "OTHER POSSIBILITIES\n"
        "Similar symptoms can have more than one cause; location, crop stage, weather, and the field pattern may change the conclusion.\n\n"
        "CHECK BEFORE ACTING\n"
        "Compare affected and healthy plants, inspect both leaf surfaces and roots where relevant, and note whether the problem follows wet areas, field edges, one planting batch, or the whole field.\n\n"
        "LOWEST-COST ACTION\n"
        "Begin with a reversible field check or cultural action and seek local extension confirmation when the cause remains unclear.\n\n"
        "BEFORE SPENDING MONEY\n"
        "Do not purchase a chemical or fertilizer from symptoms alone. Confirm the likely cause, local registration, crop label, and safe-use directions first.\n\n"
        "CONFIDENCE\n"
        "Medium for general guidance; the specific field diagnosis depends on the missing observations."
    )


def make_record(*, record_id: str, question: str, answer: str, country: str,
                crop: str, language: str, source: str, license_name: str) -> dict:
    return {
        "id": record_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ],
        "metadata": {
            "country": country or "unknown",
            "crop": crop or "generic",
            "language": language or "en",
            "source": source,
            "license": license_name,
            "question_sha256": hashlib.sha256(normalise(question).encode("utf-8")).hexdigest(),
        },
    }


def load_seed_records(cfg: dict) -> list[dict]:
    seeds = list(iter_jsonl(ROOT / "data/seeds/fieldmind_behavior.jsonl"))
    output: list[dict] = []
    for repeat in range(cfg["seed_upsample"]):
        for row in seeds:
            output.append(make_record(
                record_id=f"{row['id']}_r{repeat:02d}",
                question=row["question"],
                answer=row["answer"],
                country=row["country"],
                crop=row["crop"],
                language=row.get("language", "en"),
                source="fieldmind_original_behavior",
                license_name="GPL-3.0-project-content",
            ))
    return output


def load_farmerchat(cfg: dict) -> tuple[list[dict], Counter]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The datasets package is required for the online build. Install "
            "requirements-data.txt or use --offline-seed-only."
        ) from exc

    output: list[dict] = []
    reasons: Counter = Counter()
    seen: set[str] = set()
    rng = random.Random(cfg["random_seed"])
    dataset_name = cfg["source_dataset"]

    for config_name, limit in cfg["configs"].items():
        split = load_dataset(dataset_name, config_name, split="train")
        indices = list(range(len(split)))
        rng.shuffle(indices)
        accepted = 0
        for index in indices:
            row = split[index]
            query = str(row.get("query") or "").strip()
            response = str(row.get("response") or "").strip()
            ok, reason = is_acceptable(query, response, cfg)
            reasons[reason] += 1
            if not ok:
                continue
            key = normalise(query)
            if key in seen:
                reasons["duplicate_query"] += 1
                continue
            seen.add(key)
            output.append(make_record(
                record_id=f"farmerchat_{config_name}_{index:06d}",
                question=query,
                answer=wrap_source_answer(response),
                country=str(row.get("user_country") or config_name.title()),
                crop=str(row.get("crop") or row.get("asset_name") or "generic"),
                language="en",
                source=f"{dataset_name}:{config_name}",
                license_name=cfg["license"],
            ))
            accepted += 1
            if accepted >= limit:
                break
        if accepted < limit:
            raise RuntimeError(
                f"Only {accepted} acceptable rows found for {config_name}; requested {limit}. "
                "Lower the configured limit or inspect source changes."
            )
    return output, reasons


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/dataset.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--offline-seed-only", action="store_true")
    args = parser.parse_args()

    cfg = read_json(args.config)
    seeds = load_seed_records(cfg)
    source_rows: list[dict] = []
    filter_reasons: Counter = Counter()
    if not args.offline_seed_only:
        source_rows, filter_reasons = load_farmerchat(cfg)

    validation: list[dict] = []
    train: list[dict] = list(seeds)
    for row in source_rows:
        question = row["messages"][1]["content"]
        (validation if stable_fraction(question) < cfg["validation_fraction"] else train).append(row)

    rng = random.Random(cfg["random_seed"])
    rng.shuffle(train)
    rng.shuffle(validation)

    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    hashes = {
        "train.jsonl": write_jsonl(train_path, train),
        "validation.jsonl": write_jsonl(validation_path, validation),
    }
    source_counts = Counter(r["metadata"]["source"] for r in train + validation)
    manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "mode": "offline_seed_only" if args.offline_seed_only else "open_farmerchat",
        "random_seed": cfg["random_seed"],
        "config": cfg,
        "counts": {"train": len(train), "validation": len(validation)},
        "source_counts": dict(sorted(source_counts.items())),
        "filter_counts": dict(sorted(filter_reasons.items())),
        "sha256": hashes,
        "warning": "FarmerChat responses are AI-generated advisory text and are not automatically expert-validated.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
