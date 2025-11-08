#!/usr/bin/env python
# prepare_data.py
# curate training data for the model

import os, json, random
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

OUTDIR = "wiki_json"
SEED = 33
TRAIN_PER_SRC = 10_000
MIN_TOKENS = 25

def set_seed_all(seed: int):
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_text_column(ds: Dataset, src: str) -> Dataset:
    if src == "wikitext103":
        assert "text" in ds.column_names
        return ds.remove_columns([c for c in ds.column_names if c != "text"])
    raise ValueError(src)

def basic_clean(ds: Dataset) -> Dataset:
    ds = ds.filter(lambda ex: isinstance(ex.get("text", None), str) and len(ex["text"].strip()) > 0)
    def _strip_map(ex): return {"text": " ".join(ex["text"].split())}
    return ds.map(_strip_map, batched=False)

def filter_by_tokens(ds: Dataset, tok, min_tokens: int) -> Dataset:
    def _len_map(batch):
        enc = tok(batch["text"], add_special_tokens=False)
        return {"_tok_len": [len(ids) for ids in enc["input_ids"]]}
    ds = ds.map(_len_map, batched=True)
    ds = ds.filter(lambda ex: ex["_tok_len"] >= min_tokens)
    return ds.remove_columns(["_tok_len"])

def sample_n(ds: Dataset, n: int, seed: int):
    n = min(n, len(ds))
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    take = sorted(idx[:n])
    return ds.select(take), set(take)

def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    set_seed_all(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---------- Load & filter (WikiText-103-raw-v1) ----------
    wiki_raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]
    wiki = ensure_text_column(wiki_raw, "wikitext103")
    wiki = basic_clean(wiki)
    wiki = filter_by_tokens(wiki, tok, MIN_TOKENS)

    # train set
    wiki_train, wiki_train_idx = sample_n(wiki, TRAIN_PER_SRC, SEED + 1)

    out_dir = Path(OUTDIR)
    train_json = [{"text": ex["text"]} for ex in wiki_train]
    dump_json(out_dir / f"train_finetune_{SEED}.json", train_json)

    print("[OK] JSON saved to", OUTDIR)

if __name__ == "__main__":
    main()
