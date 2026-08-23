---
title: FieldMind Africa
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
---

# FieldMind Africa — free cloud demo

Live Space: https://huggingface.co/spaces/otieno28/fieldmind-africa

This Space runs a real Qwen3-1.7B GGUF with CPU inference inside the free Space allocation and a verified prebuilt llama.cpp runtime. It injects the FieldMind Spend Guard system behaviour and makes no paid API calls.

The live baseline is text-only. Its built-in, no-API detector supports English and Kiswahili, displays the detected language, and returns the complete answer in that language. Users can override detection manually. They can also choose diagnosis, chemical-purchase, or fertilizer-purchase mode and select the applicable country registry. Purchase mode uses deterministic evidence cards: matched cases can name registry-backed products and return a SHORTLIST or DO NOT BUY verdict, while unmatched cases request the missing evidence instead of guessing. An exact rate is transcribed only from complete user-supplied label text, and output rates not present in that label are filtered. These controls reduce hallucination risk but do not turn the app into a laboratory diagnosis or guarantee correctness.

Until the competition fine-tune is trained, the banner identifies the model as a **baseline-backed preview**. To upgrade it, set `MODEL_REPO` and `MODEL_FILE` to the public repository and filename of the selected FieldMind GGUF.

The application can sleep when the free Space is inactive. Its first start must download roughly 1.3 GB plus a 16 MB runtime and load the model, so cold starts are expected.
