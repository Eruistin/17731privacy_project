# VERSION FOR COLAB
# COLAB T4 does not support bf16. so we use fp16 here!!! The difference is small in this setting, but it exists.

#!/usr/bin/env python
# train_target_lm.py  (LoRA-ready; AdamW only)
import os, argparse
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
import torch  # <<<

# ---- LoRA ----
from peft import LoraConfig, get_peft_model, PeftModel


def infer_lora_targets(model, fallback_gpt2=True):
    """Infer LoRA target modules by model type; GPT-2 uses c_attn/c_fc/c_proj."""
    mt = getattr(getattr(model, "config", None), "model_type", "") or model.__class__.__name__.lower()
    if "gpt2" in mt and fallback_gpt2:
        return ["c_attn", "c_fc", "c_proj"]
    # fallback for LLaMA/Mistral-like architectures
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--data_dir", default="./data/fixed_wiki2")
    ap.add_argument("--data_dir", default="./wiki_json/train")
    ap.add_argument("--train_file", default="train_shadow.json")

    ap.add_argument("--model_name", "-m", default="gpt2")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--outdir", default="./models/gpt2_wiki2_target")
    ap.add_argument("--seed", type=int, default=42)

    # ---- LoRA config ----
    ap.add_argument("--lora", action="store_true", help="use LoRA")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="", help="comma-separated modules, e.g., c_attn,c_fc,c_proj")

    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--merge_lora", action="store_true", help="merge and save LoRA after training")

    args = ap.parse_args()

    # ========= data =========
    data_dir = Path(args.data_dir)
    train_path = data_dir / args.train_file
    train_items = _read_json(train_path)  # list[{'text': str}] or list[str]
    ds = Dataset.from_list(train_items)

    # filter empty samples
    def non_empty_text(ex):
        t = ex.get("text", "")
        return t is not None and len(t.strip()) > 0

    train_raw = ds.filter(non_empty_text)
    print(train_raw)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def tok_map(ex):
        texts = [(t if t and t.strip() else tok.eos_token) for t in ex["text"]]
        return tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.block_size,
            return_attention_mask=True,
        )

    train_tok = train_raw.map(tok_map, batched=True, remove_columns=train_raw.column_names)

    # collator will create labels by shifting; more robust for CLM
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    print("train size:", len(train_tok))
    print("min/max len:", min(len(x) for x in train_tok["input_ids"]),
          max(len(x) for x in train_tok["input_ids"]))

    # ========= model (no quantization) =========
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tok))  # align vocab size
    model.to("cuda")  # <<< ensure model is on GPU

    # ========= optional LoRA =========
    if args.lora:
        # target modules
        if args.target_modules.strip():
            tgt_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        else:
            tgt_modules = infer_lora_targets(model, fallback_gpt2=True)

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=tgt_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

        if args.gradient_checkpointing:
            # required for some backends when using gradient checkpointing + PEFT
            model.enable_input_require_grads()

        try:
            model.print_trainable_parameters()
        except Exception:
            pass

        model.to("cuda")  # <<< ensure LoRA-wrapped model is on GPU

    # ========= training args (AdamW, no early stopping) =========
    ta = TrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=250,
        save_strategy="no",              # no checkpoints by default
        seed=args.seed,
        fp16=True,                       # <<< ensure cuda-friendly mixed precision on T4
        bf16=False,                      # <<< avoid BF16 on T4
        no_cuda=False,                   # <<< keep CUDA enabled
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",             # explicit AdamW optimizer
    )

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=train_tok,
        tokenizer=tok,
        data_collator=collator,
    )

    trainer.train()

    # ========= save =========
    os.makedirs(args.outdir, exist_ok=True)

    if args.lora and args.merge_lora:
        # Merge LoRA into base weights for a standalone model export
        if isinstance(model, PeftModel):
            base = model.merge_and_unload()
        else:
            base = model
        base.save_pretrained(args.outdir)
        tok.save_pretrained(args.outdir)
        print("[OK] merged LoRA and saved full model to", args.outdir)
    else:
        model.save_pretrained(args.outdir)
        tok.save_pretrained(args.outdir)
        print("[OK] model (with LoRA adapters if enabled) saved to", args.outdir)


if __name__ == "__main__":
    main()
