#!/usr/bin/env bash
set -euo pipefail

# REQUIRED OWNER ACTION: replace both defaults after the tournament winner is
# uploaded publicly. Environment overrides exist only for pre-submission testing.
MODEL_URL="${MODEL_URL:-https://huggingface.co/REPLACE_WITH_PUBLIC_REPO/resolve/main/FieldMind-Africa-1.7B-Q5_K_M.gguf?download=true}"
MODEL_SHA256="${MODEL_SHA256:-REPLACE_WITH_FINAL_SHA256}"
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

if [[ "$MODEL_URL" == *"REPLACE_WITH_PUBLIC_REPO"* || "$MODEL_SHA256" == "REPLACE_WITH_FINAL_SHA256" ]]; then
  echo "Final model URL/SHA are not configured. See SUBMISSION_CHECKLIST.md." >&2
  exit 2
fi

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
