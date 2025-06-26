# Emotion-Classification-in-Dialogue-using-RoBERTa-BERT-CapsNet-and-TF-IDF


A hybrid deep learning approach to classify emotions in conversational dialogue using SemEval-2024 utterance-level data. Compared performance across TF-IDF, BERT+CapsNet, and RoBERTa with attention pooling.

---

## 📚 Dataset Overview

The dataset is structured as JSON with conversations containing multiple utterances labeled with emotion tags.

Example:
```json
{
  "utterance_ID": 3,
  "text": "Buzz him in.",
  "speaker": "Monica",
  "emotion": "neutral"
}
