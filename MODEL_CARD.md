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

REPLACE_AFTER_TRAINING with real profiler and held-out results. Do not publish illustrative numbers.

## License and attribution

The fine-tuned weights derive from Qwen3-1.7B (Apache-2.0). Dataset-derived training content is attributed to Digital Green under CC-BY-4.0. Preserve both notices when redistributing the model.
