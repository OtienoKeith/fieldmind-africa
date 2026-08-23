#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python scripts/prepare_dataset.py --offline-seed-only
python scripts/validate_data.py
python scripts/train_qlora.py --smoke-config
python scripts/check_submission.py --allow-placeholders --allow-missing-model
python -m pytest -q
