#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 F16_GGUF OUTPUT_DIR" >&2
  exit 2
fi
input="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
output_dir="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_dir="${LLAMA_CPP_DIR:-$repo_root/artifacts/llama.cpp}"
quantizer="${LLAMA_QUANTIZE:-$llama_dir/build/bin/llama-quantize}"
mkdir -p "$output_dir"

if [[ ! -x "$quantizer" ]]; then
  echo "Missing executable $quantizer. Run scripts/build_llama_cpp.sh first." >&2
  exit 1
fi
for quant in Q3_K_M Q4_K_M Q5_K_M Q6_K; do
  out="$output_dir/FieldMind-Africa-1.7B-${quant}.gguf"
  if [[ -f "$out" ]]; then
    echo "Keeping existing $out"
  else
    "$quantizer" "$input" "$out" "$quant"
  fi
done
sha256sum "$output_dir"/FieldMind-Africa-1.7B-*.gguf > "$output_dir/SHA256SUMS"
echo "Quantization candidates and checksums are in $output_dir"
