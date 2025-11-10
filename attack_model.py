import os, math, argparse
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, roc_curve, auc as _auc
import matplotlib.pyplot as plt
import json
import concurrent.futures
from pathlib import Path
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict
from datasets import Dataset
import torch.nn.functional as F
import zlib
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

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
def extract_lm_features(
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

    feat_rows = []
    feat_names = [
        "avg_neg_nll",
        "avg_token_max_prob",
        "avg_token_entropy",
        "seq_len",
        "avg_top1_top2_gap",
        *[f"nll_p{p}" for p in nll_percentiles],
        "nll_std",
    ]

    for batch in tqdm(dl, desc="extract_feats"):
        input_ids = batch["input_ids"].to(device)            # (B, L)
        attention_mask = batch["attention_mask"].to(device)  # (B, L)
        # labels for next-token: shift left by 1 (predict token t given <=t-1)
        labels = batch["labels"][:, 1:].to(device)           # (B, L-1)

        # forward
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits                                # (B, L, V)

        # log probs for convenience (but keep logits for entropy)
        log_probs = F.log_softmax(logits, dim=-1)             # (B, L, V)
        lp = log_probs[:, :-1, :]                              # (B, L-1, V)
        am = attention_mask[:, 1:].bool()                      # (B, L-1)

        # token-level logp of true token (masked)
        token_logp = torch.gather(lp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        token_logp = token_logp.masked_fill(~am, 0.0)           # zero out padded positions

        token_counts = am.sum(dim=1)                           # (B,)
        sum_logp = token_logp.sum(dim=1)                       # (B,)
        avg_neg_nll = -(sum_logp / token_counts.clamp(min=1)).cpu().numpy()  # (B,)

        # avg token max prob
        token_max_logp = lp.max(dim=-1).values.masked_fill(~am, 0.0)  # (B, L-1)
        avg_token_max_prob = (token_max_logp.exp().sum(dim=1) / token_counts.clamp(min=1)).cpu().numpy()

        # entropy via categorical entropy (memory friendly)
        ent = torch.distributions.Categorical(logits=logits[:, :-1, :]).entropy().masked_fill(~am, 0.0)  # (B, L-1)
        avg_token_entropy = (ent.sum(dim=1) / token_counts.clamp(min=1)).cpu().numpy()

        # top1-top2 gap (probability gap averaged across valid tokens)
        topk_logp = lp.topk(2, dim=-1).values                    # (B, L-1, 2)
        gap = (topk_logp[..., 0] - topk_logp[..., 1]).exp()     # prob gap per token
        gap = gap.masked_fill(~am, 0.0)
        avg_gap = (gap.sum(dim=1) / token_counts.clamp(min=1)).cpu().numpy()

        seq_len_arr = token_counts.cpu().numpy()

        # per-sample per-token NLL arrays -> compute percentiles & std
        B = input_ids.size(0)
        token_nll_lists = []
        for i in range(B):
            valid_logps = token_logp[i][am[i]].cpu().numpy()   # logp for valid tokens
            if valid_logps.size == 0:
                token_nll_lists.append(np.array([np.nan]))
            else:
                token_nll_lists.append(-valid_logps)  # convert to positive NLLs

        p_arrays = {p: [] for p in nll_percentiles}
        stds = []
        for arr in token_nll_lists:
            if np.isnan(arr).all():
                for p in nll_percentiles:
                    p_arrays[p].append(np.nan)
                stds.append(np.nan)
            else:
                for p in nll_percentiles:
                    p_arrays[p].append(np.percentile(arr, p))
                stds.append(np.std(arr, ddof=1))

        # assemble rows
        for i in range(B):
            row = [
                float(avg_neg_nll[i]),
                float(avg_token_max_prob[i]),
                float(avg_token_entropy[i]),
                float(seq_len_arr[i]),
                float(avg_gap[i]),
                *[float(p_arrays[p][i]) for p in nll_percentiles],
                float(stds[i]),
            ]
            feat_rows.append(row)

    X = np.vstack(feat_rows) if len(feat_rows) > 0 else np.zeros((0, len(feat_names)))
    return X


def build_dataloader(path, tokenizer, block_size, batch_size):
    data = _read_json(path)
    texts = [item['text'] for item in data]
    ds = Dataset.from_dict({"text": texts})
    ds = tokenize_dataset(ds, tokenizer, block_size)
    dl = DataLoader(ds, batch_size=batch_size)
    return dl, texts



def build_shadow_attack_dataset(
    seeds: List[int],
    shadow_data_template: Tuple[str, str],
    shadow_model_dir_template: str,
    tokenizer_name_or_path: str,
    batch_size: int,
    block_size: int,
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
    all_feats = []
    all_labels = []
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
        feats_train = extract_lm_features(model, train_path, tokenizer, block_size, batch_size, device=device)
        feats_test = extract_lm_features(model, test_path, tokenizer, block_size, batch_size, device=device)

        # labels member=1 nonmember=0
        all_feats.append(feats_train)
        all_labels.append(np.ones(len(feats_train), dtype=int))

        all_feats.append(feats_test)
        all_labels.append(np.zeros(len(feats_test), dtype=int))

    X = np.vstack(all_feats)
    y = np.concatenate(all_labels)
    return X, y


# --------------------
# Train attack model & evaluate
# --------------------
def train_attack_model(X, y, out_dir: Path, random_state=42):
    print("Attack dataset shape:", X.shape, y.sum(), "members,", len(y)-y.sum(), "non-members")
    # split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)
    # validation metrics
    prob_val = clf.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, prob_val)
    fpr, tpr, thresholds = roc_curve(y_val, prob_val)
    # find TPR at FPR=0.01 (interpolate)
    target_fpr = 0.01
    if (fpr <= target_fpr).any():
        # compute interpolated tpr
        from numpy import interp
        tpr_at_target = interp(target_fpr, fpr, tpr)
    else:
        tpr_at_target = 0.0
    print(f"Validation AUC = {auc_score:.4f}, TPR@FPR=0.01 = {tpr_at_target:.4f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "attack_X.npy", X)
    np.save(out_dir / "attack_y.npy", y)
    print("Saved attack dataset to", out_dir)
    return clf, (auc_score, tpr_at_target)


# --------------------
# Predict on target model and prepare csv
# --------------------
# def predict_on_target_and_write_csv(
#     clf,
#     tokenizer_name_or_path: str,
#     target_model_dir: str,
#     target_texts: List[str],
#     out_csv_path: Path,
#     device: torch.device,
# ):
#     # load target model (or implement API query logic)
#     model = AutoModelForCausalLM.from_pretrained(target_model_dir).to(device)
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
#     feats, ids = extract_lm_features(model, tokenizer, target_texts, device=device)
#     probs = clf.predict_proba(feats)[:, 1]
#     # write csv with id, score
#     import csv
#     with out_csv_path.open("w", newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(["id", "score"])
#         for i, p in enumerate(probs):
#             writer.writerow([i, float(p)])
#     print("Wrote predictions to", out_csv_path)


# --------------------
# Main CLI
# --------------------
def main():
    parser = argparse.ArgumentParser()
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

    X, y = build_shadow_attack_dataset(
        seeds=args.shadow_seeds,
        shadow_data_template=(args.shadow_data_template[0], args.shadow_data_template[1]),
        shadow_model_dir_template=args.shadow_model_dir_template,
        tokenizer_name_or_path=args.target_model_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=device,
    )

    out_dir = Path(args.out_dir)
    clf, metrics = train_attack_model(X, y, out_dir)
    print("Attack model metrics (AUC, TPR@FPR=0.01):", metrics)

    # # If target test file exists, run on it
    # target_test_path = Path(args.target_test_json)
    # if target_test_path.exists():
    #     target_items = read_json(target_test_path)
    #     target_texts = []
    #     for i, it in enumerate(target_items):
    #         text, _ = extract_text_and_id(it, i)
    #         target_texts.append(text)
    #     predict_on_target_and_write_csv(clf, args.tokenizer, args.target_model_dir, target_texts, out_dir / "target_predictions.csv", device)
    # else:
    #     print("No target test file found at", target_test_path, "; skip target prediction step.")

if __name__ == "__main__":
    main()