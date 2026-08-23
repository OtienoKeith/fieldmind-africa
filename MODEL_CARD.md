---
language:
- en
- sw
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
datasets:
- DigiGreen/farmerchat-queries
pipeline_tag: text-generation
tags:
- agriculture
- africa
- gguf
- offline
- qwen3
---

# FieldMind Africa 1.7B

FieldMind Africa is a Qwen3-1.7B agriculture adaptation designed to help an extension officer or cooperative reason about ambiguous crop symptoms before a farmer spends a limited budget on an input.

## Intended use

- Offline, text-only decision support on CPU laptops.
- English and limited Kiswahili questions about smallholder crops and field observations.
- Asking useful follow-up checks and explaining uncertainty.

## Not intended for

- Definitive diagnosis from text alone.
- Pesticide, veterinary medicine, or fertilizer dosage that conflicts with a registered product label or local regulation.
- Emergency human/animal poisoning response.
- Replacing an agronomist, extension service, veterinarian, soil test, laboratory, or field inspection.

## Training

The reproducible scripts combine curated CC-BY-4.0 FarmerChat examples with original, project-authored Spend Guard behaviour cases created without external API calls. Source records are AI-generated advisory responses and should not be assumed expert-validated. The QLoRA configuration and exact data manifest are included in the repository.

## Evaluation

The final 200-step free-Colab run completed on a Tesla T4 in 2,494 seconds with training loss 0.7590 at epoch 0.2837. The exact training configuration, dataset manifest, and metrics are published alongside the GGUF. Fifteen application-level regression tests cover English/Kiswahili routing, registry-grounded product shortlisting, cassava and waterlogging no-buy decisions, purple-maize phosphorus reasoning, exact-label dose extraction, isolated case context, and prompt-leak prevention. Official-profiler numbers are published only from saved profiler output; no illustrative performance score is claimed.

Selected artifact: `FieldMind-Africa-1.7B-Q5_K_M.gguf` (1.26 GB), SHA-256 `42e4b60d3df7f0661c65e1f85ffad11a1cf1be125105e09691712270d7c94795`.

## License and attribution

The fine-tuned weights derive from Qwen3-1.7B (Apache-2.0). Dataset-derived training content is attributed to Digital Green under CC-BY-4.0. Preserve both notices when redistributing the model.
