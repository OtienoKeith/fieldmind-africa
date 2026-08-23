# FieldMind Africa — Devpost submission copy

## Tagline

Before you buy the chemical, ask the laptop.

## Project links

- Live free cloud demo: https://huggingface.co/spaces/otieno28/fieldmind-africa
- Public source and submission repository: https://github.com/OtienoKeith/fieldmind-africa
- Public model repository: https://huggingface.co/otieno28/fieldmind-africa-1.7b-gguf

## Inspiration

For a smallholder farmer, the wrong input purchase is not a harmless recommendation—it can consume the season's limited fertilizer or crop-protection budget. Yellowing, curling and leaf spots are ambiguous, yet advice often jumps straight from a symptom to a product. FieldMind Africa was built for the moment before money changes hands.

## What it does

FieldMind Africa is a text-only agricultural purchase-decision assistant for extension offices, cooperatives, community centres and agro-input shops. Its Spend Guard behaviour separates observations from possible causes, asks for the few field checks that change the decision, recommends the lowest-cost reversible action, and ends with a clear purchase verdict.

The live demo supports automatic English/Kiswahili response routing. For verified Kenya tomato cases it can shortlist PCPB-listed options, while cassava mosaic-like symptoms correctly return “do not buy fungicide.” Exact chemical rates are never guessed or transferred between formulations: the app will only transcribe a rate from complete label text supplied by the user and still asks the user to verify the crop, target, formulation, country, PPE, PHI and REI.

## How we built it

The submission starts from Qwen3-1.7B and uses Unsloth QLoRA on a freely allocated Colab T4. The deterministic data pipeline prepares CC-BY-4.0 Digital Green FarmerChat records for Kenya, Nigeria and Ethiopia, preserves per-record provenance, filters unsafe dose-seeking and redacted records, and mixes project-authored Spend Guard examples. The executed build produced 11,280 training and 316 validation records. A separate 24-question evaluation suite covers general agriculture, African context and premature-spending safety; the validator reports zero question leakage.

The trained model is merged, converted to GGUF and quantized with llama.cpp. Q3_K_M, Q4_K_M, Q5_K_M and Q6_K candidates are evaluated on the same CPU settings so the final choice balances ADTC's 50% accuracy, 30% throughput and 20% memory-efficiency weights. Runtime inference is local llama.cpp with zero network dependency after the public GGUF download.

The public cloud demonstration uses free Hugging Face Space CPU infrastructure and no paid inference API. Purchase cards are deterministic and evidence-backed for responsiveness and safety; the local GGUF provides the diagnosis-mode language model preview until the final competition model is installed.

## Challenges

The hardest product problem was not generating more possible causes—it was deciding when the evidence justifies spending money. A 1.7B baseline could drift between languages, repeat dose phrases or expose prompt text on unmatched questions. We responded by narrowing the verified language promise to English and Kiswahili, moving chemical/fertilizer purchase mode to tested evidence cards, and treating the final GGUF itself—not Python prompt injection—as the artifact the ADTC profiler must score.

The engineering constraint was equally important: everything had to remain free, reproducible and runnable on ordinary hardware. The pipeline therefore avoids paid APIs, proprietary datasets, GPU credits and cloud inference endpoints.

## Accomplishments

- A working public free-cloud demo with clear purchase verdicts.
- Reproducible open-data preparation with hashes, licenses and zero train/eval question leakage.
- A public GitHub repository with passing automated CPU-safe checks.
- English and Kiswahili official test prompts focused on real Kenya/Nigeria smallholder decisions.
- Label-grounded dose handling that refuses to invent or transfer an application rate.
- A llama.cpp-only offline packaging, evaluation and quantization pipeline.

## What we learned

Agricultural usefulness is not the same as diagnostic confidence. A model can be valuable by protecting a farmer from an unjustified purchase, identifying the next observation and admitting what it cannot know. We also learned that demo middleware can hide weaknesses that a direct GGUF evaluator will expose, so submission quality must be measured on the model artifact itself.

## What's next

The immediate next step is agronomist review and expansion of country-specific product registries. Future versions should add an offline image encoder only after proving that image quality and crop coverage improve decisions, expand verified African-language datasets, and package the selected GGUF with a small local desktop/mobile interface for extension teams.

