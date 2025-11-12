import os, math, argparse
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, roc_curve, auc as _auc
import matplotlib.pyplot as plt
import json
import lightgbm as lgb
import concurrent.futures
from pathlib import Path
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict
from datasets import Dataset
import torch.nn.functional as F
import zlib
from tqdm import tqdm
from peft import PeftModel, PeftConfig


def tokenize_dataset(ds, tok, max_len):
    ds = ds.filter(lambda ex: isinstance(ex.get("text", ""), str) and len(ex["text"].strip()) > 0)
    def _map(ex):
        out = tok(ex["text"], truncation=True, padding="max_length", max_length=max_len, return_attention_mask=True)
        out["labels"] = out["input_ids"].copy()
        return out
    ds = ds.map(_map, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



@torch.no_grad()
def inference(
    model,
    path,
    tokenizer,
    block_size: int,
    batch_size: int,
    device: torch.device,
    nll_percentiles=(10, 50, 90),
):
    model.eval()

    dl, _ = build_dataloader(path, tokenizer, block_size, batch_size)
    logits_bases, logits_fts = [], []

    for batch in tqdm(dl, desc="extract_feats"):
        input_ids = batch["input_ids"].to(device)            # (B, L)
        attention_mask = batch["attention_mask"].to(device)  # (B, L)

        logits_ft = model(input_ids, attention_mask=attention_mask).logits
        logits_fts.append(logits_ft.cpu().numpy())

        with model.disable_adapter():
            logits_base = model(input_ids, attention_mask=attention_mask).logits
            logits_bases.append(logits_base.cpu().numpy())

    logits_bases = np.asarray(logits_bases, dtype=np.float32)
    logits_fts = np.asarray(logits_fts, dtype=np.float32)
    return logits_fts, logits_bases


def tokenize_dataset_with_overflow(texts, tok, max_len, stride=None):
    if stride is None:
        stride = max_len // 4  # 例如 128
    enc = tok(
        texts,
        truncation=True,
        max_length=max_len,
        stride=stride,
        padding=False,
        return_overflowing_tokens=True,
        return_attention_mask=True,
        return_length=True,
    )
    # overflow_to_sample_mapping 告诉你每个 chunk 来自哪条原文本
    mapping = enc["overflow_to_sample_mapping"]
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # 构建 labels 并右对齐 pad 到 max_len，便于 batch
    def pad_to_max(x, pad_id):
        return x + [pad_id] * (max_len - len(x))
    pad_id = tok.eos_token_id

    rows = []
    for i in range(len(input_ids)):
        ids = input_ids[i]
        am = attention_mask[i]
        if len(ids) > max_len:
            ids = ids[:max_len]
            am = am[:max_len]
        ids = pad_to_max(ids, pad_id)
        am = pad_to_max(am, 0)
        rows.append({
            "input_ids": ids,
            "attention_mask": am,
            "labels": ids.copy(),
            "orig_idx": int(mapping[i]),
        })
    ds = Dataset.from_list(rows)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels","orig_idx"])
    return ds


def build_dataloader(path, tokenizer, block_size, batch_size):
    data = _read_json(path)
    texts = [item['text'] for item in data]
    ds = tokenize_dataset_with_overflow(texts, tokenizer, block_size, stride=block_size//4)
    dl = DataLoader(ds, batch_size=batch_size)
    return dl, texts



def build_shadow_attack_dataset(
    seeds: List[int],
    shadow_data_template: Tuple[str, str],
    shadow_model_dir_template: str,
    tokenizer_name_or_path: str,
    batch_size: int,
    block_size: int,
    out_dir: Path,
    device: torch.device,
):
    """
    For each seed:
      - read shadow train (member) and shadow test (non-member) files
      - load corresponding shadow model
      - extract features on both splits and label 1/0
    Returns:
      X (N, D), y (N,), meta list of dicts with {'seed','orig_id','text'}
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side="right"


    for seed in seeds:
        print(f"[+] Processing shadow seed {seed}")
        train_path = Path(shadow_data_template[0].format(seed=seed))
        test_path = Path(shadow_data_template[1].format(seed=seed))
        assert train_path.exists(), f"{train_path} not found"
        assert test_path.exists(), f"{test_path} not found"


        model_dir = shadow_model_dir_template.format(seed=seed)
        print("loading model from", model_dir)
        config = PeftConfig.from_pretrained(model_dir)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        # model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
        model = PeftModel.from_pretrained(base_model, model_dir).to(device)
        # extract features
        logits_ft_train, logits_base_train = inference(model, train_path, tokenizer, block_size, batch_size, device=device)
        logits_ft_test, logits_base_test = inference(model, train_path, tokenizer, block_size, batch_size, device=device)

        # labels member=1 nonmember=0
        
        np.save(out_dir / f"logits_ft_train_{seed}.npy", logits_ft_train)
        np.save(out_dir / f"logits_base_train_{seed}.npy", logits_base_train)
        np.save(out_dir / f"logits_ft_test_{seed}.npy", logits_ft_test)
        np.save(out_dir / f"logits_base_test_{seed}.npy", logits_base_test)

    
    return






# --------------------
# Main CLI
# --------------------
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--shadow_seeds", nargs="+", type=int, default=[7,15,33,99,111,123,256,333,512,1024],
    parser.add_argument("--shadow_seeds", nargs="+", type=int, default=[7,15,33],
                        help="list of seeds used to build shadow datasets")
    parser.add_argument("--shadow_model_dir_template", type=str, default="models/shadow_{seed}/gpt2_3_lora32_adamw_b8_lr2",
                        help="format template for shadow model directories")
    parser.add_argument("--shadow_data_template", nargs=2, type=str,
                        default=("wiki_json/train/train_shadow_{seed}.json", "wiki_json/test/test_shadow_{seed}.json"),
                        help="two templates: train and test file path templates, use {seed}")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--target_model_dir", type=str, default="models/final/gpt2_3_lora32_adamw_b8_lr2")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--block_size", type=int, default=512)
    # parser.add_argument("--target-test-json", type=str, default="wiki_json/target_val_texts.json")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("Using device:", device)
    out_dir = Path(args.out_dir)

    build_shadow_attack_dataset(
        seeds=args.shadow_seeds,
        shadow_data_template=(args.shadow_data_template[0], args.shadow_data_template[1]),
        shadow_model_dir_template=args.shadow_model_dir_template,
        tokenizer_name_or_path=args.tokenizer,
        batch_size=args.batch_size,
        block_size=args.block_size,
        out_dir=out_dir,
        device=device,
    )


if __name__ == "__main__":
    main()