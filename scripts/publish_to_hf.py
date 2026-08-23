#!/usr/bin/env python3
"""Upload the selected GGUF and model card to a public Hugging Face repo.

This is an optional owner action after benchmarking. It requires a free
Hugging Face account and HF_TOKEN with write permission; training does not.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo such as username/fieldmind-africa-1.7b-gguf")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()
    magic = b""
    if args.model.exists():
        with args.model.open("rb") as handle:
            magic = handle.read(4)
    if magic != b"GGUF":
        raise SystemExit(f"Not a valid local GGUF: {args.model}")
    if not args.token:
        raise SystemExit("Set HF_TOKEN or pass --token. Never commit the token.")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install requirements-data.txt first") from exc

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(args.model),
        path_in_repo=args.model.name,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Upload benchmark-selected {args.model.name}",
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "MODEL_CARD.md"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add model card",
    )
    print(f"Public model: https://huggingface.co/{args.repo_id}/blob/main/{args.model.name}")
    print(f"Direct URL: https://huggingface.co/{args.repo_id}/resolve/main/{args.model.name}?download=true")
    print(f"SHA-256: {sha256(args.model)}")


if __name__ == "__main__":
    main()
