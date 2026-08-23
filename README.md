# FieldMind Africa

> **Before you buy the chemical, ask the laptop.**

FieldMind Africa is a reproducible ADTC 2026 agriculture submission pipeline for a small, offline language model. It is designed for extension officers, cooperatives, community centres, and agro-input shops that serve smallholder farmers where connectivity and input budgets are limited.

The product's memorable behaviour is **Spend Guard**: when symptoms are ambiguous, it asks for discriminating observations and recommends the lowest-cost safe check before suggesting a purchase. It does not pretend that a text description is a laboratory diagnosis.

## Live free cloud demo

Open **[FieldMind Africa on Hugging Face Spaces](https://huggingface.co/spaces/otieno28/fieldmind-africa)**. The public Space uses the trained FieldMind Africa 1.7B Q5_K_M GGUF with llama.cpp and makes no paid API calls. Purchase-decision cards return immediately; free-form diagnosis uses the local model and is slower on the free shared CPU. Free Spaces may sleep when inactive, so allow time for a cold start.

The evidence-bounded paths for cassava mosaic, flooded/yellow maize, tomato leaf spots, and purple young maize are deterministic and source-linked in both English and Kiswahili. Warm public-API checks on 23 August returned the purple-maize and Kiswahili cassava cards in about 1.9 seconds each. Unmatched diagnosis questions use the trained GGUF and remain explicitly uncertain.

The repository also includes a **[91-second narrated demo video](video/fieldmind-africa-demo.mp4)** built entirely with local/free tooling.

The current demo is **text-only**, not image or audio multimodal. Its local, no-API language detector supports and fully replies in the project's two verified languages: English and Kiswahili. A user can still override detection from the language menu. Users choose a diagnosis, chemical-purchase, or fertilizer-purchase flow and select the country whose product registry applies. Purchase mode is deterministic rather than free-form: matched cases can produce a SHORTLIST or DO NOT BUY verdict and name registry-backed products where the evidence supports them; unmatched cases request the missing evidence. Exact application rates are transcribed only when the user supplies complete label text containing the rate, because rates cannot safely be transferred between products or formulations. The demo exposes its matched sources, but it remains decision support and cannot guarantee a correct diagnosis.

## What is complete in this repository

- Official-template-compatible `metadata.json`, exactly two domain prompts, and required `download_model.sh`.
- Deterministic preparation of open CC-BY-4.0 FarmerChat Kenya, Nigeria, and Ethiopia data.
- Project-authored FieldMind behaviour examples created in-repo without paid or external API calls.
- QLoRA training for Qwen3-1.7B with Unsloth on a free Kaggle/Colab GPU.
- Merge, GGUF conversion, Q3/Q4/Q5/Q6 quantization, and variant-selection scripts.
- Three held-out evaluations: general agriculture, African context, and Spend Guard safety.
- A llama.cpp CLI demo and ADTC profiler instructions.
- Tests that run without a GPU, model download, or paid service.

## Honest project status

The reproducible pipeline, trained artifact, cloud deployment, and CPU-safe checks are complete. The real open-data build contains 11,280 training records and 316 validation records; its validator reports zero train/validation/evaluation question leakage. A 200-step QLoRA run completed on a free Colab Tesla T4, and the selected 1.26 GB Q5_K_M artifact is published with SHA-256 verification. Official-profiler and held-out measurements are reported only when a saved report exists; no illustrative score is presented as measured.

## Why this aligns with ADTC scoring

The official score is `0.50 × accuracy + 0.30 × throughput + 0.20 × efficiency − thermal penalty`. Throughput is capped at 15 tokens/second and memory efficiency uses a 7 GB profiler limit. Therefore the pipeline tests Q3_K_M, Q4_K_M, Q5_K_M, and Q6_K and selects the highest measured composite—not automatically Q4.

FieldMind also claims the African Use Case bonus because the use case, crops, locations, deployment model, English/Kiswahili prompts, and limited-input decision constraint are load-bearing rather than decorative.

## Quick start: CPU-only work available anywhere

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt -r requirements-dev.txt

# Fast offline smoke build from the included seed examples
python scripts/prepare_dataset.py --offline-seed-only
python scripts/validate_data.py
python scripts/check_submission.py --allow-placeholders --allow-missing-model
pytest -q
```

For a real open-data build (internet required only during preparation):

```bash
python scripts/prepare_dataset.py
```

Outputs are deterministic JSONL files under `data/processed/` plus a manifest with counts, source, license, and SHA-256 hashes.

## Free GPU stage: Kaggle first, Colab fallback

Open the notebooks in order:

1. `notebooks/01_prepare_data.ipynb` — optional if the prepared JSONL files are uploaded with the repo.
2. `notebooks/02_train_fieldmind.ipynb` — QLoRA, adapter save, 16-bit merge, and optional GGUF export.

In Kaggle, enable an available free GPU and internet for dependency/model download. In Colab Free, select a GPU runtime if one is allocated. No paid API key, GPU credit, Weights & Biases account, or private dataset is required. Hugging Face login is needed only if you choose to upload the final artifact; it is not needed to train.

Recommended first run:

```bash
python scripts/train_qlora.py --config config/training.json \
  --train data/processed/train.jsonl \
  --validation data/processed/validation.jsonl
```

## GGUF tournament

The training script saves a merged Transformers model. On Kaggle/Colab CPU or another free Linux runner:

```bash
bash scripts/build_llama_cpp.sh
bash scripts/export_gguf.sh artifacts/fieldmind-merged-16bit artifacts/gguf
bash scripts/quantize_tournament.sh artifacts/gguf/FieldMind-Africa-1.7B-F16.gguf artifacts/gguf
```

Benchmark every candidate through the official profiler:

```bash
python -m pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
python scripts/profile_variants.py --models artifacts/gguf/*.gguf
```

Run the FieldMind held-out rubric with a local `llama-server` binary:

```bash
python scripts/evaluate_gguf.py \
  --model artifacts/gguf/FieldMind-Africa-1.7B-Q5_K_M.gguf \
  --eval-dir data/eval \
  --output benchmarks/fieldmind-q5-eval.json
```

Combine profiler and held-out quality results:

```bash
python scripts/select_variant.py \
  --profiler-dir benchmarks/profiler \
  --quality-dir benchmarks/quality \
  --output benchmarks/tournament.csv
```

The submission artifact is Q5_K_M. Its filename, public URL, and checksum are synchronized in `metadata.json`, `download_model.sh`, and `REPORT.md`.

## Baseline demo before training

This downloads the Apache-2.0 Qwen3-1.7B Q4_K_M baseline to a separate filename. It is useful for testing only and must not be submitted as FieldMind.

```bash
bash scripts/download_baseline.sh
python demo/app.py --model model/Qwen3-1.7B-baseline-Q4_K_M.gguf
```

The final model demo is:

```bash
python demo/app.py --model model/FieldMind-Africa-1.7B-Q5_K_M.gguf
```

## Final submission flow

1. Run the real data preparation and free-GPU training.
2. Evaluate all quantizations on the same 4-thread CPU environment.
3. Upload only the selected GGUF to a public Hugging Face repository.
4. Replace the three identity placeholders and final model URL/SHA in the noted files.
5. Run `bash download_model.sh` from a clean clone.
6. Run the full official profiler, without `--skip-accuracy`.
7. Run `python scripts/check_submission.py` with no allow flags.
8. Commit the official profiler's saved `submission.json` and its exact measurements if the rules require it at submission time.

The optional free-hosting helper prints the exact direct URL and SHA-256 needed by `download_model.sh`:

```bash
HF_TOKEN=your_write_token python scripts/publish_to_hf.py \
  --model artifacts/gguf/FieldMind-Africa-1.7B-Q5_K_M.gguf \
  --repo-id YOUR_HANDLE/fieldmind-africa-1.7b-gguf
```

Never commit the token. Make the model repository public before running the clean-clone download test.

## Free public cloud demo

`cloud_demo/` is a Gradio Hugging Face Space. It runs the public trained FieldMind Africa 1.7B Q5_K_M GGUF with a checksum-verified prebuilt llama.cpp runtime on the free Space CPU. No paid inference API is used. Its label-aware purchase flow separates two decisions: registry-backed product shortlisting and exact label-grounded dose extraction.

Create a free Gradio Space and upload the three files from `cloud_demo/`. On first boot the Space downloads about 1.3 GB plus a 16 MB verified llama.cpp runtime. Free Spaces can sleep, so cold starts are expected. The checked-in defaults already point at the public trained FieldMind GGUF.

See `SUBMISSION_CHECKLIST.md` for the exact owner-only steps.

## Safety and scope

FieldMind is decision support, not a substitute for an agronomist, extension service, pesticide label, soil test, veterinary diagnosis, or local regulation. It may name a chemical option only when the crop/target/country match verified registry evidence. It may state an exact dose only from the user's exact product label, never by transferring a rate between active ingredients, products, or formulations. Training data contains AI-generated advisory responses and is not automatically expert-validated; the data card documents this limitation.

## Open resources and licenses

- Base model: Qwen/Qwen3-1.7B, Apache-2.0.
- Main dataset: DigiGreen/farmerchat-queries, CC-BY-4.0.
- Training: Unsloth, Transformers, TRL, PEFT, PyTorch.
- Inference/export: llama.cpp (MIT).
- Competition profiler/template: ADTC official repositories (GPL-3.0).

The repository code inherits GPL-3.0 compatibility from the official submission template. Dataset records retain their source attribution and CC-BY-4.0 obligations; model redistribution must preserve the Qwen Apache-2.0 notices.
