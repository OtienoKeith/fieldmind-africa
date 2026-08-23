#!/usr/bin/env python3
"""QLoRA fine-tune for a free NVIDIA Kaggle/Colab runtime.

This intentionally fails early on CPU so users do not accidentally spend hours
or exhaust RAM. Use --smoke-config to validate files/import-independent logic.
"""
# ruff: noqa: I001 -- Unsloth must precede torch/transformers imports at runtime.
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "base_model", "max_seq_length", "seed", "lora_rank", "lora_alpha",
        "learning_rate", "epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "output_dir", "merged_dir", "target_modules",
    }
    missing = sorted(required - cfg.keys())
    if missing:
        raise ValueError(f"Training config missing keys: {', '.join(missing)}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/training.json")
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/validation.jsonl")
    parser.add_argument("--smoke-config", action="store_true")
    args = parser.parse_args()
    cfg = read_config(args.config)
    for path in (args.train, args.validation):
        if not path.exists():
            raise SystemExit(f"Missing {path}; run scripts/prepare_dataset.py first")
    if args.smoke_config:
        print(json.dumps({"status": "ok", "config": str(args.config), "base_model": cfg["base_model"]}, indent=2))
        return

    # Unsloth must be imported before transformers/trl so it can patch them.
    import unsloth  # noqa: F401
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. Run this stage on a free Kaggle/Colab GPU runtime.")

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.0),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
        use_rslora=False,
        loftq_config=None,
    )

    data = load_dataset("json", data_files={"train": str(args.train), "validation": str(args.validation)})

    def format_batch(batch):
        texts = []
        for messages in batch["messages"]:
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            except TypeError:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    remove_columns = data["train"].column_names
    data = data.map(format_batch, batched=True, remove_columns=remove_columns, desc="Applying Qwen chat template")
    output_dir = ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    args_kwargs = dict(
        dataset_text_field="text",
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["epochs"],
        max_steps=cfg.get("max_steps", -1),
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy="steps" if len(data["validation"]) else "no",
        eval_steps=cfg.get("eval_steps", 100),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=2,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=cfg["seed"],
        report_to="none",
        output_dir=str(output_dir),
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data["validation"] if len(data["validation"]) else None,
        args=SFTConfig(**args_kwargs),
    )

    # Train only on assistant tokens where the current template helper supports Qwen.
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    except Exception as exc:  # compatibility fallback, logged rather than hidden
        print(f"WARNING: response-only masking unavailable; training all chat tokens: {exc}")

    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    packages = ("torch", "unsloth", "unsloth_zoo", "transformers", "trl", "peft", "datasets", "bitsandbytes")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    metrics = dict(result.metrics)
    metrics.update({
        "base_model": cfg["base_model"],
        "seed": cfg["seed"],
        "cuda_device": torch.cuda.get_device_name(0),
        "package_versions": versions,
    })
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output_dir / "training_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    manifest_path = args.train.parent / "manifest.json"
    if manifest_path.exists():
        (output_dir / "dataset_manifest.json").write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

    merged_dir = ROOT / cfg["merged_dir"]
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    print(f"Saved adapter to {output_dir}")
    print(f"Saved merged 16-bit model to {merged_dir}")


if __name__ == "__main__":
    main()
