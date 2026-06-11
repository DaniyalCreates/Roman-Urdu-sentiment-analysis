---
language:
- ur
tags:
- sentiment-analysis
- roman-urdu
- xlm-roberta
- text-classification
license: mit
datasets:
- custom
metrics:
- accuracy
---

# Roman Urdu Sentiment — XLM-RoBERTa

Fine-tuned [`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base) for binary sentiment classification of **Roman Urdu** social media text.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("DaniyalCreates/roman-urdu-sentiment-xlmroberta")
model = AutoModelForSequenceClassification.from_pretrained("DaniyalCreates/roman-urdu-sentiment-xlmroberta")
model.eval()

text = "yaar ye movie bohat achi thi"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=1).squeeze()

labels = ["Negative", "Positive"]
print(labels[probs.argmax()], f"{probs.max():.1%}")
# → Positive 99.7%
```

## Training Details

| Setting | Value |
|---------|-------|
| Base model | xlm-roberta-base |
| Dataset | RUSAD (10,999 Roman Urdu reviews) |
| Train / test split | 80 / 20 |
| Epochs | 5 |
| Learning rate | 1e-5 |
| Batch size | 16 |
| Warmup steps | 50 |

## Results

| Metric | Score |
|--------|-------|
| Test accuracy | 82% |
| Macro F1 | 0.82 |
| Negative F1 | 0.82 |
| Positive F1 | 0.83 |

## About

Part of an MSc dissertation in Artificial Intelligence & Data Science at the University of Hull (2025).  
**Author:** Daniyal Tariq
