#!/usr/bin/env python3
"""Minimal offline FieldMind terminal demo using llama.cpp."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

SYSTEM = """You are FieldMind Africa, an offline agricultural decision-support assistant. Separate observations from possible causes. Ask for discriminating field checks, prefer low-cost reversible actions, avoid premature chemical or dosage advice, and state uncertainty. Use the headings WHAT MAY BE HAPPENING, OTHER POSSIBILITIES, CHECK BEFORE ACTING, LOWEST-COST ACTION, BEFORE SPENDING MONEY, and CONFIDENCE. Reply in the user's language."""


def find_cli(explicit: str | None) -> str:
    for candidate in (explicit, os.environ.get("LLAMA_CLI"), "llama-cli", "llama.cpp-llama-cli"):
        if candidate and (Path(candidate).exists() or shutil.which(candidate)):
            return candidate
    raise SystemExit("llama-cli not found. Build llama.cpp or pass --llama-cli /path/to/llama-cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="FieldMind Africa offline demo")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-cli")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=420)
    parser.add_argument("--question")
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    cli = find_cli(args.llama_cli)
    question = args.question or input("Farmer's question: ").strip()
    if not question:
        raise SystemExit("Question cannot be empty")
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    command = [
        cli, "-m", str(args.model), "-p", prompt, "-n", str(args.tokens),
        "-t", str(args.threads), "-ngl", "0", "--temp", "0", "--no-display-prompt",
        "--reverse-prompt", "<|im_end|>",
    ]
    print("\nFIELDmind AFRICA — offline CPU inference\n")
    completed = subprocess.run(command, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
