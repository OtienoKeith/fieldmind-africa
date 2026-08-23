#!/usr/bin/env bash
set -euo pipefail

url="https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf?download=true"
sha="d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5"
path="model/Qwen3-1.7B-baseline-Q4_K_M.gguf"
mkdir -p model

verify() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s  %s\n' "$sha" "$1" | sha256sum --check --status
  else
    printf '%s  %s\n' "$sha" "$1" | shasum -a 256 --check --status
  fi
}

if [[ -f "$path" ]] && verify "$path"; then
  echo "Baseline already present and verified: $path"
  exit 0
fi
tmp="${path}.part"
trap 'rm -f "$tmp"' EXIT
curl --fail --location --retry 3 --output "$tmp" "$url"
verify "$tmp" || { echo "Baseline SHA-256 verification failed" >&2; exit 1; }
mv "$tmp" "$path"
trap - EXIT
echo "Downloaded baseline for pipeline testing only: $path"
