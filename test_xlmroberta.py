import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "xlmroberta_finetuned"
LABELS = {0: "Negative", 1: "Positive"}

sentences = [
    "yaar ye movie bohat achi thi",
    "ye service bilkul bakwas hai",
    "mujhe ye phone pasand nahi aya",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

print(f"{'Sentence':<40} {'Sentiment':<10} {'Confidence':>10}")
print("-" * 63)

with torch.no_grad():
    for text in sentences:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
        pred = int(probs.argmax())
        confidence = probs[pred].item()
        print(f"{text:<40} {LABELS[pred]:<10} {confidence:>9.1%}")
