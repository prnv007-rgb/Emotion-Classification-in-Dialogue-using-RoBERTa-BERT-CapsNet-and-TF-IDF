import os
import json
import yaml
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
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
            if text is None or label is None or label not in LABEL2ID:
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

        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": torch.tensor(label_id, dtype=torch.long)}

# ──────────────────────────────────────────────────────────────────────────────
# MODEL — with attention pooling on RoBERTa

class EmotionClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(p=0.3)
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        attn_weights = torch.softmax(self.attention(last_hidden), dim=1)  # (batch, seq_len, 1)
        pooled = (attn_weights * last_hidden).sum(dim=1)  # (batch, hidden)
        dropped = self.dropout(pooled)
        return self.classifier(dropped)

# ──────────────────────────────────────────────────────────────────────────────
# TRAIN & EVAL FUNCTIONS

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(preds==labels).float().mean():.4f}"})

    return total_loss/total, total_correct/total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False)
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

# ──────────────────────────────────────────────────────────────────────────────
# MAIN

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(script_dir, "config.yaml"))

    train_json = config["train_json"]
    dev_json = config["dev_json"]
    tokenizer_name = config["tokenizer_name"]
    max_length = config.get("max_length", 128)
    bs_train = config.get("batch_size_train", 32)
    bs_eval = config.get("batch_size_eval", 64)
    num_workers = config.get("num_workers", 2)
    num_epochs = config.get("num_epochs", 5)
    lr = config.get("learning_rate", 5e-5)
    wd = config.get("weight_decay", 0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_ds = EmotionDataset(train_json, tokenizer, max_length)
    dev_ds = EmotionDataset(dev_json, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=bs_train, shuffle=True, num_workers=num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=bs_eval, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = EmotionClassifier(pretrained_model_name=tokenizer_name, num_labels=len(LABEL2ID))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    save_path = os.path.join(script_dir, "best_emotion_model_roberta.pt")

    for epoch in range(1, num_epochs+1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        print(f" → Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")

        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
        print(f" → Dev   loss: {dev_loss:.4f}, Dev   acc: {dev_acc:.4f}")

        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model.state_dict(), save_path)
            print(f" ⭐ New best model saved (dev_acc = {best_acc:.4f})")

if __name__ == "__main__":
    main()