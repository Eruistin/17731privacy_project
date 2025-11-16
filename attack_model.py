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

def _token_metrics_from_logits(logits, labels, attention_mask, rank_topk=(1,5,10)):
    log_probs = F.log_softmax(logits, dim=-1)              # (B, L, V)
    lp = log_probs[:, :-1, :]                              # (B, L-1, V)
    am = attention_mask[:, 1:].bool()                      # (B, L-1)
    true_lp = torch.gather(lp, -1, labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    true_lp = true_lp.masked_fill(~am, 0.0)

    token_counts = am.sum(dim=1)                           # (B,)
    sum_logp = true_lp.sum(dim=1)
    avg_neg_nll = -(sum_logp / token_counts.clamp(min=1)) # (B,)

    token_max_logp = lp.max(dim=-1).values.masked_fill(~am, 0.0)
    avg_token_max_prob = (token_max_logp.exp().sum(dim=1) / token_counts.clamp(min=1))

    ent = torch.distributions.Categorical(logits=logits[:, :-1, :]).entropy().masked_fill(~am, 0.0)
    avg_token_entropy = (ent.sum(dim=1) / token_counts.clamp(min=1))

    topk_logp = lp.topk(2, dim=-1).values
    gap = (topk_logp[..., 0] - topk_logp[..., 1]).exp().masked_fill(~am, 0.0)
    avg_gap = (gap.sum(dim=1) / token_counts.clamp(min=1))

    out = {
        "avg_neg_nll": avg_neg_nll,
        "avg_token_max_prob": avg_token_max_prob,
        "avg_token_entropy": avg_token_entropy,
        "avg_gap": avg_gap,
        "true_lp": true_lp,          # (B, L-1)
        "am": am,                    # (B, L-1)
        "token_counts": token_counts # (B,)
    }

    for k in rank_topk:
        tk = lp.topk(k, dim=-1).indices    # (B, L-1, k)
        hit = (tk == labels.unsqueeze(-1)).any(dim=-1) & am
        out[f"hit_top{k}_frac"] = (hit.sum(dim=1) / token_counts.clamp(min=1))  # (B,)

    prob_true = true_lp.exp().masked_fill(~am, 0.0)  # (B, L-1)
    thr = 0.99
    runs = []
    for i in range(prob_true.size(0)):
        mask = (prob_true[i] > thr) & am[i]
        max_run, cur = 0, 0
        for v in mask.tolist():
            if v: cur += 1
            else:
                max_run = max(max_run, cur)
                cur = 0
        max_run = max(max_run, cur)
        runs.append(max_run)
    out["max_run_p099"] = torch.tensor(runs, device=avg_neg_nll.device, dtype=torch.float32)

    return out
def _window_aggregate(arr, am, win=128, stride=128, reduce="max"):
    B, L = arr.shape
    vals = []
    for i in range(B):
        valid = am[i]
        x = arr[i]
        x = x[valid]
        res = -1e9 if reduce=="max" else 1e9
        if len(x) == 0:
            vals.append(float("nan"))
            continue
        for s in range(0, len(x), stride):
            w = x[s:s+win]
            if len(w)==0: break
            if reduce=="max":
                res = max(res, float(np.max(w)))
            elif reduce=="p95":
                res = max(res, float(np.percentile(w,95)))
            else:
                res = max(res, float(np.mean(w)))
        vals.append(res)
    return np.array(vals, dtype=np.float32)


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

    for batch in tqdm(dl, desc="extract_feats"):
        input_ids = batch["input_ids"].to(device)            # (B, L)
        attention_mask = batch["attention_mask"].to(device)  # (B, L)
        labels = batch["labels"][:, 1:].to(device)           # (B, L-1)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits                                # (B, L, V)
        m_ft = _token_metrics_from_logits(logits, labels, attention_mask)

        with model.disable_adapter():
            logits_base = model(input_ids, attention_mask=attention_mask).logits
        m_base = _token_metrics_from_logits(logits_base, labels, attention_mask)

        B = input_ids.size(0)
        am = m_ft["am"]
        true_lp_ft = m_ft["true_lp"]           # (B, L-1)
        true_lp_base = m_base["true_lp"]
        d_true_nll = -(true_lp_ft) - (-(true_lp_base))  # = NLL_ft - NLL_base
        d_true_nll = d_true_nll.masked_fill(~am, 0.0)   # (B, L-1)
        delta_nll = -d_true_nll  # (B, L-1)

        token_counts = m_ft["token_counts"]
        def seq_stat(x, am, reduce="mean"):
            x = x.masked_fill(~am, 0.0)
            if reduce == "mean":
                return (x.sum(dim=1) / token_counts.clamp(min=1))
            elif reduce == "std":
                arr=[]
                for i in range(B):
                    v = x[i][am[i]].detach().cpu().numpy()
                    arr.append(float(np.std(v, ddof=1)) if v.size>1 else 0.0)
                return torch.tensor(arr, device=x.device)
            else:
                raise NotImplementedError

        delta_nll_mean = seq_stat(delta_nll, am, "mean")   # (B,)
        delta_nll_std  = seq_stat(delta_nll, am, "std")    # (B,)
        p_arrays = {p: [] for p in nll_percentiles}
        for i in range(B):
            v = delta_nll[i][am[i]].detach().cpu().numpy()
            if v.size == 0:
                for p in nll_percentiles: p_arrays[p].append(np.nan)
            else:
                for p in nll_percentiles: p_arrays[p].append(np.percentile(v, p))

        improved_frac = []
        for i in range(B):
            vb = true_lp_base[i][am[i]]
            vf = true_lp_ft[i][am[i]]
            if vb.numel()==0:
                improved_frac.append(0.0)
            else:
                improved_frac.append(float((vf > vb).float().mean().item()))
        improved_frac = np.array(improved_frac, dtype=np.float32)

        delta_nll_np = delta_nll.detach().cpu().numpy()
        am_np = am.detach().cpu().numpy()
        win_max = _window_aggregate(delta_nll_np, am_np, win=128, stride=128, reduce="max")
        win_p95 = _window_aggregate(delta_nll_np, am_np, win=128, stride=128, reduce="p95")

        hit1_ft = m_ft["hit_top1_frac"].detach().cpu().numpy()
        hit5_ft = m_ft["hit_top5_frac"].detach().cpu().numpy()
        run99_ft = m_ft["max_run_p099"].detach().cpu().numpy()

        base_avg_nll = m_base["avg_neg_nll"].detach().cpu().numpy()
        ft_avg_nll   = m_ft["avg_neg_nll"].detach().cpu().numpy()
        base_entropy = m_base["avg_token_entropy"].detach().cpu().numpy()
        ft_entropy   = m_ft["avg_token_entropy"].detach().cpu().numpy()
        base_gap     = m_base["avg_gap"].detach().cpu().numpy()
        ft_gap       = m_ft["avg_gap"].detach().cpu().numpy()

        for i in range(B):
            row = [
                float(ft_avg_nll[i]),
                float(base_avg_nll[i]),
                float(base_avg_nll[i]-ft_avg_nll[i]),
                float(ft_entropy[i]),
                float(base_entropy[i]),
                float(base_entropy[i]-ft_entropy[i]),
                float(ft_gap[i]),
                float(base_gap[i]),
                float(ft_gap[i]-base_gap[i]),

                float(hit1_ft[i]),
                float(hit5_ft[i]),
                float(run99_ft[i]),

                float(delta_nll_mean[i].item()),
                float(delta_nll_std[i].item()),
                float(improved_frac[i]),
                float(win_max[i]),
                float(win_p95[i]),
            ]
            for p in nll_percentiles:
                row.append(float(p_arrays[p][i]))
            feat_rows.append(row)

    X = np.asarray(feat_rows, dtype=np.float32)
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
        if not(train_path.exists() and test_path.exists()):
            continue


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
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.001,
        "num_leaves": 128,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 1e-3,
        "lambda_l2": 1e-3,
        "verbose": -1,
        "random_state": random_state,
    }

    clf = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        num_boost_round=4000,
    )
    prob_val = clf.predict(X_val, num_iteration=clf.best_iteration)
    auc_score = roc_auc_score(y_val, prob_val)
    fpr, tpr, _ = roc_curve(y_val, prob_val)

    target_fpr = 0.01
    if (fpr <= target_fpr).any():
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
# Main CLI
# --------------------
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--shadow_seeds", nargs="+", type=int, default=[7,15,33,99,111,123,256,333,512,1024],
    parser.add_argument("--shadow_seeds", nargs="+", type=int, default=[7,15,33],
                        help="list of seeds used to build shadow datasets")
    parser.add_argument("--shadow_model_dir_template", type=str, default="models/shadow_models/shadow_{seed}/gpt2_3_lora32_adamw_b8_lr2",
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
    if os.path.exists(out_dir / "attack_X.npy") and os.path.exists(out_dir / "attack_y.npy"):
        X = np.load(out_dir / "attack_X.npy")
        y = np.load(out_dir / "attack_y.npy")
    else:
        X, y = build_shadow_attack_dataset(
            seeds=args.shadow_seeds,
            shadow_data_template=(args.shadow_data_template[0], args.shadow_data_template[1]),
            shadow_model_dir_template=args.shadow_model_dir_template,
            tokenizer_name_or_path=args.target_model_dir,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=device,
        )

    
    clf, metrics = train_attack_model(X, y, out_dir)
    print("Attack model metrics (AUC, TPR@FPR=0.01):", metrics)

if __name__ == "__main__":
    main()