import os
import json
import yaml
import torch
import tempfile
import streamlit as st
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import EmotionClassifier, EmotionDataset, evaluate, LABEL2ID, load_config  # Assuming you saved original code as model.py

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI

st.set_page_config(page_title="Emotion Classifier", layout="centered")
st.title("🧠 Emotion Classifier - RoBERTa Model")

# File upload
uploaded_file = st.file_uploader("Upload a test JSON file", type=["json"])
run_eval = st.button("Run Evaluation")

# Config and tokenizer setup
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

if os.path.isfile(config_path):
    config = load_config(config_path)
else:
    st.error("Missing `config.yaml` file.")
    st.stop()

tokenizer_name = config["tokenizer_name"]
max_length = config.get("max_length", 128)
batch_size = config.get("batch_size_eval", 32)
num_workers = config.get("num_workers", 0)  # Streamlit threads better with 0 workers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

if uploaded_file and run_eval:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Dataset and loader
    try:
        test_ds = EmotionDataset(tmp_path, tokenizer, max_length)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Load model
        model = EmotionClassifier(pretrained_model_name=tokenizer_name, num_labels=len(LABEL2ID))
        model_path = os.path.join(script_dir, "best_emotion_model_roberta.pt")
        if not os.path.isfile(model_path):
            st.error(f"Model checkpoint not found at {model_path}")
            st.stop()

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # Evaluation
        criterion = nn.CrossEntropyLoss()
        st.info("Evaluating model...")
        loss, acc = evaluate(model, test_loader, criterion, device)
        st.success(f"✅ Test Accuracy: {acc:.4f}")
        st.write(f"📉 Loss: {loss:.4f}")

        # Show sample predictions
        st.subheader("📊 Sample Predictions")
        model.eval()
        preds_to_show = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i > 2: break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=1)

                for j in range(min(5, len(preds))):
                    text = tokenizer.decode(input_ids[j], skip_special_tokens=True)
                    st.markdown(f"**Text:** {text}")
                    true_label = list(LABEL2ID.keys())[list(LABEL2ID.values()).index(labels[j].item())]
                    pred_label = list(LABEL2ID.keys())[list(LABEL2ID.values()).index(preds[j].item())]
                    st.write(f"➡️ Ground Truth: `{true_label}` | Prediction: `{pred_label}`")
                    st.write("---")

    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}")
