# FieldMind training and evaluation data

## Main source

`DigiGreen/farmerchat-queries` is a curated set of farmer questions and AI-generated FarmerChat responses. FieldMind uses the Kenya, Nigeria, and Ethiopia configurations under CC-BY-4.0. The preparation manifest records exact requested/accepted counts and output hashes.

These source responses are advisory model outputs, not guaranteed agronomist-verified ground truth. They may contain overconfident, outdated, location-specific, or unsafe advice. FieldMind therefore uses filtering, a conservative system policy, original behaviour examples, held-out safety tests, and mandatory manual review. This reduces risk; it does not eliminate it.

## Original behaviour examples

`data/seeds/fieldmind_behavior.jsonl` contains project-authored examples created directly in this repository without paid or external API calls. They teach:

- differential possibilities instead of single-cause certainty;
- discriminating checks;
- low-cost, reversible actions;
- explicit purchase decisions;
- confidence and escalation.

Seed examples are upsampled during preparation because behaviour is the product differentiator. They are not part of evaluation.

## Held-out evaluation

The three files under `data/eval/` are original regression cases and are never loaded by the preparation script. Each includes transparent required concept groups and forbidden phrases. Automated substring scoring is deliberately simple and reproducible; the raw answers must also be reviewed by a competent agronomist before public deployment.

## Privacy

The source dataset states that identifiers are masked. The preparation pipeline additionally rejects records containing common redaction markers so the fine-tune is not trained to reproduce them.

## Known biases

- English dominates the open source data; Kiswahili behaviour relies on a small authored set and base-model capability.
- Kenya, Nigeria, and Ethiopia do not represent all African farming systems.
- Source answers may favour practices or products unavailable to a given farmer.
- Text-only observations are inherently incomplete.
- Simple lexical evaluation can miss correct paraphrases or reward superficial keyword use.

## Citation

Digital Green. *FarmerChat Agricultural Q&A Dataset* (2026). CC-BY-4.0.
https://huggingface.co/datasets/DigiGreen/farmerchat-queries
