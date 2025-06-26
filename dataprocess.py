import json

def load_split(path_to_json):
    items = []
    with open(path_to_json, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    for convo in conversations:
        convo_id = convo["conversation_ID"]
        for utt in convo["conversation"]:
            utt_id = utt["utterance_ID"]
            # Create a unique utterance_id string (optional, for tracking)
            utterance_id = f"{convo_id}_{utt_id}"
            text       = utt["text"]
            emotion    = utt["emotion"]  # e.g., "joy", "sadness", ...
            items.append({
              "utterance_id": utterance_id,
              "text":         text,
              "label":        emotion
            })
    return items

train_data = load_split(r"C:\Users\prana\OneDrive\Desktop\llm\p5\ECF\SemEval-2024\traindata.json")
dev_data   = load_split(r"C:\Users\prana\OneDrive\Desktop\llm\p5\ECF\SemEval-2024\dev.json")
test_data  = load_split(r"C:\Users\prana\OneDrive\Desktop\llm\p5\ECF\SemEval-2024\testdata.json")

with open(r"C:\Users\prana\OneDrive\Desktop\llm\p5\ECF\SemEval-2024\train_data_flat.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open(r"C:\Users\prana\OneDrive\Desktop\llm\p5\ECF\SemEval-2024\dev_data_flat.json", "w", encoding="utf-8") as f:
    json.dump(dev_data, f, indent=2, ensure_ascii=False)

with open(r"C:\Users\prana\OneDrive\Desktop\llm\p5\ECF\SemEval-2024\test_data_flat.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)