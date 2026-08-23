#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 MERGED_HF_DIR OUTPUT_DIR" >&2
  exit 2
fi
merged_dir="$(cd "$1" && pwd)"
output_dir="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_dir="${LLAMA_CPP_DIR:-$repo_root/artifacts/llama.cpp}"
converter="$llama_dir/convert_hf_to_gguf.py"
mkdir -p "$output_dir"

if [[ ! -f "$converter" ]]; then
  echo "Missing $converter. Run scripts/build_llama_cpp.sh first." >&2
  exit 1
fi
python "$converter" "$merged_dir" \
  --outfile "$output_dir/FieldMind-Africa-1.7B-F16.gguf" \
  --outtype f16
echo "Wrote $output_dir/FieldMind-Africa-1.7B-F16.gguf"
