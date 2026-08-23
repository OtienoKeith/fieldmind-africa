#!/usr/bin/env bash
set -euo pipefail

# Public, credential-free model release. Environment overrides remain useful for
# testing mirrors, while the defaults are the exact submission artifact.
MODEL_URL="${MODEL_URL:-https://huggingface.co/otieno28/fieldmind-africa-1.7b-gguf/resolve/main/FieldMind-Africa-1.7B-Q5_K_M.gguf?download=true}"
MODEL_SHA256="${MODEL_SHA256:-42e4b60d3df7f0661c65e1f85ffad11a1cf1be125105e09691712270d7c94795}"
MODEL_PATH="model/FieldMind-Africa-1.7B-Q5_K_M.gguf"

mkdir -p "$(dirname "$MODEL_PATH")"

verify() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s  %s\n' "$MODEL_SHA256" "$file" | sha256sum --check --status
  elif command -v shasum >/dev/null 2>&1; then
    printf '%s  %s\n' "$MODEL_SHA256" "$file" | shasum -a 256 --check --status
  else
    echo "sha256sum or shasum is required" >&2
    return 1
  fi
}

if [[ -f "$MODEL_PATH" ]] && verify "$MODEL_PATH"; then
  echo "Model already present and verified: $MODEL_PATH"
  exit 0
fi

tmp="${MODEL_PATH}.part"
trap 'rm -f "$tmp"' EXIT
rm -f "$tmp"
if command -v curl >/dev/null 2>&1; then
  curl --fail --location --retry 3 --output "$tmp" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
  wget --tries=3 --output-document="$tmp" "$MODEL_URL"
else
  echo "curl or wget is required" >&2
  exit 1
fi
verify "$tmp" || { echo "SHA-256 verification failed" >&2; exit 1; }
mv "$tmp" "$MODEL_PATH"
trap - EXIT
echo "Downloaded and verified: $MODEL_PATH"
