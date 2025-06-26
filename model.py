# model.py
import os
import json
import yaml
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# UTILS

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ──────────────────────────────────────────────────────────────────────────────
# LABEL MAP

LABEL2ID = {
    "anger":    0,
    "disgust":  1,
    "fear":     2,
    "joy":      3,
    "neutral":  4,
    "sadness":  5,
    "surprise": 6,
}

# ──────────────────────────────────────────────────────────────────────────────
# DATASET

class EmotionDataset(Dataset):
    def __init__(self, json_path: str, tokenizer: AutoTokenizer, max_length: int = 128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Expected {json_path} to exist.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = []
        for entry in data:
            text = entry.get("text", None)
            label = entry.get("label", None)
            if text is None or label is None:
                continue
            if label not in LABEL2ID:
                continue
            self.samples.append((text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label_str = self.samples[idx]
        label_id = LABEL2ID[label_str]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label_id, dtype=torch.long),
        }

# ──────────────────────────────────────────────────────────────────────────────
# MODEL

class EmotionClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(p=0.3)
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden = outputs.last_hidden_state
        attn_weights = torch.softmax(self.attention(last_hidden), dim=1)
        pooled = (attn_weights * last_hidden).sum(dim=1)
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits

# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(preds==labels).float().mean():.4f}"})

    return total_loss/total, total_correct/total
