# Final owner checklist

The build cannot truthfully complete these identity/hosting steps without the submitter and a trained artifact.

- [ ] Replace `fieldmind-africa-replace-with-adtf-team-id` in `metadata.json` and `REPORT.md`.
- [ ] Replace the submitter name if needed, `replace-me@example.com`, and `replace-me` GitHub handle.
- [ ] Train on a free Kaggle/Colab GPU and keep the console log/config.
- [ ] Run held-out evaluation on every candidate quantization.
- [ ] Run the official ADTC profiler on every candidate in the same CPU environment.
- [ ] Select the real winner; do not assume Q5_K_M wins.
- [ ] Rename/copy the winner to the exact `_runtime.model_path` in `metadata.json`.
- [ ] If another quantization wins, update model name, quantization, path, report, and download script together.
- [ ] Upload the one final GGUF to a public, credential-free URL.
- [ ] Set `MODEL_URL` and `MODEL_SHA256` defaults in `download_model.sh`.
- [ ] From a clean clone, run `bash download_model.sh` twice (idempotency check).
- [ ] Run the full official profiler without `--skip-accuracy` and save `submission.json`.
- [ ] Replace every `TBD — measure` and `REPLACE_AFTER_TRAINING` in `REPORT.md`.
- [ ] Run `python scripts/check_submission.py`; it must exit 0 with no allow flags.
- [ ] Confirm the public GitHub repository contains no GGUF weights and no secrets.
- [ ] Confirm inference succeeds with outbound networking disabled.
- [ ] Record dataset/model licenses and attribution in the public model card.
- [ ] Submit the public repository URL and two-minute video on Devpost before the official deadline.
