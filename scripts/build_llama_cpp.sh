#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${LLAMA_CPP_DIR:-$repo_root/artifacts/llama.cpp}"
# Pinned from upstream master on 2026-08-23. Override deliberately if a newer
# Qwen/GGUF fix is required, and record the resulting commit in REPORT.md.
ref="${LLAMA_CPP_REF:-b0539c43ed13b16bf0d8a0840646faea65469702}"

if [[ ! -d "$source_dir/.git" ]]; then
  mkdir -p "$(dirname "$source_dir")"
  git clone https://github.com/ggml-org/llama.cpp.git "$source_dir"
fi
git -C "$source_dir" fetch --depth 1 origin "$ref"
git -C "$source_dir" checkout --detach "$ref"
cmake -S "$source_dir" -B "$source_dir/build" -DGGML_NATIVE=OFF -DGGML_OPENMP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$source_dir/build" --config Release -j 4 --target llama-cli llama-server llama-bench llama-quantize
echo "llama.cpp binaries built under $source_dir/build/bin"
