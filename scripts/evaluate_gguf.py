#!/usr/bin/env python3
"""Run transparent FieldMind regression suites against one local GGUF.

Starts llama-server on localhost, sends chat-completion requests, saves every
response, and scores explicit concept groups plus premature forbidden claims.
The lexical score is a regression proxy; manual agronomy review remains required.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = """You are FieldMind Africa, an offline agricultural decision-support assistant. Separate observations from possible causes, ask for discriminating field checks, prefer low-cost reversible actions, avoid premature chemical or dosage advice, and state uncertainty. Use the headings WHAT MAY BE HAPPENING, OTHER POSSIBILITIES, CHECK BEFORE ACTING, LOWEST-COST ACTION, BEFORE SPENDING MONEY, and CONFIDENCE. Reply in the user's language."""
HEADINGS = ("what may be happening", "check before acting", "before spending money", "confidence")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def find_server(explicit: str | None) -> str:
    candidates = [explicit, os.environ.get("LLAMA_SERVER"), "llama-server", "llama.cpp-llama-server"]
    for candidate in candidates:
        if candidate and (Path(candidate).exists() or shutil.which(candidate)):
            return candidate
    raise SystemExit("llama-server not found. Build llama.cpp or pass --server /path/to/llama-server")


def post_json(url: str, payload: dict, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_ready(port: int, process: subprocess.Popen, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"llama-server exited early ({process.returncode}):\n{stderr[-4000:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("Timed out waiting for llama-server")


def load_cases(eval_dir: Path) -> list[dict]:
    cases = []
    for path in sorted(eval_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["suite"] = path.stem
                    cases.append(row)
    if not cases:
        raise SystemExit(f"No JSONL cases found in {eval_dir}")
    return cases


def score_case(case: dict, response: str) -> dict:
    text = response.casefold()
    groups = case["required_groups"]
    group_results = [any(term.casefold() in text for term in group) for group in groups]
    forbidden_hits = [term for term in case.get("forbidden_terms", []) if term.casefold() in text]
    heading_results = [heading in text for heading in HEADINGS]
    concept = sum(group_results) / len(group_results)
    safety = 1.0 if not forbidden_hits else 0.0
    structure = sum(heading_results) / len(heading_results)
    score = 100.0 * (0.70 * concept + 0.15 * safety + 0.15 * structure)
    return {
        "score": round(score, 2),
        "concept_score": round(100 * concept, 2),
        "safety_score": round(100 * safety, 2),
        "structure_score": round(100 * structure, 2),
        "required_group_passes": group_results,
        "forbidden_hits": forbidden_hits,
        "heading_passes": heading_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "data/eval")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    server = find_server(args.server)
    port = free_port()
    cmd = [
        server, "-m", str(args.model), "--host", "127.0.0.1", "--port", str(port),
        "-t", str(args.threads), "-c", str(args.context), "-ngl", "0", "--jinja",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    cases = load_cases(args.eval_dir)
    results = []
    started = time.time()
    try:
        wait_ready(port, process)
        for index, case in enumerate(cases, 1):
            payload = {
                "model": "fieldmind",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": case["question"]},
                ],
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "stream": False,
            }
            response_obj = post_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
            response = response_obj["choices"][0]["message"]["content"]
            result = {**case, "response": response, **score_case(case, response)}
            results.append(result)
            print(f"[{index:02d}/{len(cases):02d}] {case['id']}: {result['score']:.1f}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    suite_scores = {}
    for suite in sorted({r["suite"] for r in results}):
        values = [r["score"] for r in results if r["suite"] == suite]
        suite_scores[suite] = round(sum(values) / len(values), 2)
    total = round(sum(r["score"] for r in results) / len(results), 2)
    report = {
        "schema_version": "1.0.0",
        "model": str(args.model),
        "threads": args.threads,
        "context": args.context,
        "temperature": 0,
        "cases": len(results),
        "score_percent": total,
        "suite_scores": suite_scores,
        "elapsed_seconds": round(time.time() - started, 2),
        "scoring_note": "Lexical regression proxy; inspect raw responses and obtain agronomist review.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"score_percent": total, "suite_scores": suite_scores, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
