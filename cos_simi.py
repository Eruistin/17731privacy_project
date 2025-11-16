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
    

class MIA:
    def __init__(self, model_path, target_model_path, device):

        self.model = self.load_model(model_path)
        self.target_model = self.load_model(target_model_path)
        
        self.device = device
        self.model.to(device)
        self.target_model.to(device)
        self.target_model.eval()
        self.model.eval()

    def load_model(self, model_path):
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_path).to(self.device)
        return model
    
    
    @torch.no_grad()
    def get_logits(self, dl):
        distances = []
        for batch in tqdm(dl, desc="Generating logits"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            ft_logits = self.model(input_ids, attention_mask=attention_mask).logits
            target_logits = self.target_model(input_ids, attention_mask=attention_mask).logits

            with self.model.disable_grad():
                base_logits = self.model(input_ids, attention_mask=attention_mask).logits

            cos_sim1 = torch.nn.functional.cosine_similarity(ft_logits, target_logits, dim=-1)
            cos_sim2 = torch.nn.functional.cosine_similarity(base_logits, target_logits, dim=-1)
            dist = torch.exp(cos_sim1) / (torch.exp(cos_sim1) + torch.exp(cos_sim2))
            distances.append(dist.cpu().numpy())

        return np.concatenate(distances)
            
            






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="models/shadow_models/shadow_7/gpt2_3_lora32_adamw_b8_lr2")
    parser.add_argument("--target_model_name_or_path", type=str, default="models/shadow_models/shadow_7/gpt2_3_lora32_adamw_b8_lr2")
    parser.add_argument("--data_dir", type=str, default="wiki_json")
    parser.add_argument("--data_name", type=str, default="train/train_shadow_7.json")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    texts = _read_json(data_dir / args.data_name)
    texts = [item['text'] for item in texts]
    dataset = Dataset.from_dict({"text": texts})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = tokenize_dataset(dataset, tokenizer, args.block_size)
    dl = DataLoader(dataset, batch_size=batch_size)
    mia = MIA(args.model_name_or_path, args.target_model_name_or_path, device)

    pred = mia.get_logits(dl)
    y_true = np.array([1] * len(pred))
    auc_score = roc_auc_score(y_true, pred)
    fpr, tpr, _ = roc_curve(y_true, pred)

    target_fpr = 0.01
    if (fpr <= target_fpr).any():
        from numpy import interp
        tpr_at_target = interp(target_fpr, fpr, tpr)
    else:
        tpr_at_target = 0.0

    print(f"Validation AUC = {auc_score:.4f}, TPR@FPR=0.01 = {tpr_at_target:.4f}")


    