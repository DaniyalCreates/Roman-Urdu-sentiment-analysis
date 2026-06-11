import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

os.chdir('/Users/daniyaltariq/Desktop/Daniyal Dissertation Coding')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("All libraries imported successfully")
print("Device:", device)

# ── Load fine-tuned model ──────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained('xlmroberta_finetuned')
model = AutoModelForSequenceClassification.from_pretrained('xlmroberta_finetuned')
model = model.to(device)
model.eval()
print("Model loaded successfully")

# ── Load emotion dataset ───────────────────────────────────────────────────────
df_emotion = pd.read_excel('RU-EN-Emotion Dataset.xlsx')
print(df_emotion.shape)
print(df_emotion.columns.tolist())
print(df_emotion.head())

print(df_emotion['Level 2'].value_counts())

# ── Map emotions → binary sentiment ───────────────────────────────────────────
emotion_map = {'Happy': 1, 'Anger': 0, 'Sad': 0, 'Fear': 0}
df_stress = df_emotion[df_emotion['Level 2'].isin(emotion_map.keys())].copy()
df_stress['sentiment'] = df_stress['Level 2'].map(emotion_map)
df_stress = df_stress.reset_index(drop=True)

print(df_stress.shape)
print(df_stress['sentiment'].value_counts())
print(df_stress['Level 2'].value_counts())

# ── Inference ──────────────────────────────────────────────────────────────────
class StressTestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

stress_dataset = StressTestDataset(df_stress['Tweets'].values, tokenizer)
stress_loader = DataLoader(stress_dataset, batch_size=16, shuffle=False)

all_preds = []
total = len(stress_loader)

with torch.no_grad():
    for i, batch in enumerate(stress_loader, 1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        if i % 50 == 0 or i == total:
            print(f"  Batch {i}/{total}")

df_stress['predicted'] = all_preds
print(classification_report(df_stress['sentiment'], df_stress['predicted'],
                             target_names=['Negative', 'Positive']))

# ── Confusion matrix ───────────────────────────────────────────────────────────
cm = confusion_matrix(df_stress['sentiment'], df_stress['predicted'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('XLM-RoBERTa Stress Test Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('StressTest_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved StressTest_confusion_matrix.png")

# ── Misclassified examples ─────────────────────────────────────────────────────
df_stress['original_label'] = df_stress['Level 2']
df_stress['correct'] = df_stress['sentiment'] == df_stress['predicted']

misclassified = df_stress[~df_stress['correct']][
    ['Tweets', 'original_label', 'sentiment', 'predicted']
].reset_index(drop=True)
misclassified.columns = ['Text', 'Emotion', 'Actual_Sentiment', 'Predicted_Sentiment']
misclassified['Actual_Sentiment']     = misclassified['Actual_Sentiment'].map({0: 'Negative', 1: 'Positive'})
misclassified['Predicted_Sentiment']  = misclassified['Predicted_Sentiment'].map({0: 'Negative', 1: 'Positive'})

misclassified.to_csv('misclassified_examples.csv', index=False)
print("Total misclassified:", len(misclassified))
print(misclassified.head(10))
