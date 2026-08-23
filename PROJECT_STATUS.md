# Project status

## Can be executed in this environment

- Repository validation and tests.
- Offline seed-only dataset build.
- Open FarmerChat download and deterministic preparation when network/package access is available.
- Static evaluation-set and train/eval leakage checks.
- Baseline download and CPU inference if llama.cpp is installed and roughly 1.3 GB of bandwidth/storage is acceptable.

## Must be executed on a free Kaggle/Colab GPU

- QLoRA training with Unsloth and Qwen3-1.7B.
- Adapter merge (technically possible on CPU, but slow and memory-heavy).
- First full model sanity evaluation.

## Must happen after training

- Export/quantize the actual FieldMind weights.
- Benchmark Q3/Q4/Q5/Q6 on the same CPU target.
- Choose the final quantization from measured quality, tokens/s, and peak RAM.
- Upload the winning GGUF to a public host.
- Fill the owner identity, final URL, SHA-256, and measured report fields.

No benchmark or quality number in this repository is presented as a measured result until the corresponding JSON artifact exists.
