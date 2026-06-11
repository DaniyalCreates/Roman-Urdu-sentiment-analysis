import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(__file__), "xlmroberta_finetuned")

app = FastAPI()

print("Loading model…")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
_model.eval()
print("Model ready.")


class TextIn(BaseModel):
    text: str


@app.post("/predict")
def predict(body: TextIn):
    inputs = _tokenizer(
        body.text, return_tensors="pt", truncation=True, max_length=128
    )
    with torch.no_grad():
        probs = torch.softmax(_model(**inputs).logits, dim=1).squeeze()
    label = int(probs.argmax())
    return {
        "sentiment": "Positive" if label == 1 else "Negative",
        "confidence": round(probs[label].item() * 100, 1),
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roman Urdu Sentiment</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f4f4f6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #1a1a2e;
  }

  .card {
    background: #fff;
    border-radius: 16px;
    padding: 40px 44px;
    width: 100%;
    max-width: 540px;
    box-shadow: 0 4px 24px rgba(0,0,0,.08);
  }

  h1 {
    font-size: 1.45rem;
    font-weight: 700;
    margin-bottom: 6px;
  }

  .subtitle {
    font-size: .875rem;
    color: #6b7280;
    margin-bottom: 28px;
  }

  label {
    display: block;
    font-size: .8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: #6b7280;
    margin-bottom: 8px;
  }

  textarea {
    width: 100%;
    height: 110px;
    padding: 12px 14px;
    border: 1.5px solid #e5e7eb;
    border-radius: 10px;
    font-size: 1rem;
    font-family: inherit;
    resize: vertical;
    transition: border-color .15s;
    outline: none;
  }

  textarea:focus { border-color: #6366f1; }

  button {
    margin-top: 16px;
    width: 100%;
    padding: 13px;
    background: #6366f1;
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background .15s, opacity .15s;
  }

  button:hover  { background: #4f46e5; }
  button:active { opacity: .85; }
  button:disabled { opacity: .55; cursor: not-allowed; }

  #result {
    margin-top: 24px;
    padding: 20px 22px;
    border-radius: 12px;
    display: none;
    align-items: center;
    gap: 14px;
  }

  #result.positive { background: #f0fdf4; border: 1.5px solid #86efac; }
  #result.negative { background: #fff1f2; border: 1.5px solid #fca5a5; }

  .emoji { font-size: 2rem; line-height: 1; }

  .label-text {
    font-size: 1.35rem;
    font-weight: 700;
  }

  .positive .label-text { color: #16a34a; }
  .negative .label-text { color: #dc2626; }

  .confidence {
    font-size: .875rem;
    color: #6b7280;
    margin-top: 2px;
  }

  .spinner {
    display: none;
    width: 18px; height: 18px;
    border: 2.5px solid #fff;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin .6s linear infinite;
    margin: 0 auto;
  }

  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="card">
  <h1>Roman Urdu Sentiment</h1>
  <p class="subtitle">Type a sentence in Roman Urdu and classify its sentiment.</p>

  <label for="txt">Text</label>
  <textarea id="txt" placeholder="e.g. yaar ye movie bohat achi thi…"></textarea>

  <button id="btn" onclick="classify()">
    <span id="btn-label">Analyse</span>
    <span class="spinner" id="spinner"></span>
  </button>

  <div id="result">
    <span class="emoji" id="emoji"></span>
    <div>
      <div class="label-text" id="sentiment"></div>
      <div class="confidence" id="confidence"></div>
    </div>
  </div>
</div>

<script>
async function classify() {
  const text = document.getElementById('txt').value.trim();
  if (!text) return;

  const btn     = document.getElementById('btn');
  const label   = document.getElementById('btn-label');
  const spinner = document.getElementById('spinner');
  const result  = document.getElementById('result');

  btn.disabled     = true;
  label.style.display  = 'none';
  spinner.style.display = 'block';
  result.style.display  = 'none';

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();

    const isPos = data.sentiment === 'Positive';
    result.className = isPos ? 'positive' : 'negative';
    document.getElementById('emoji').textContent      = isPos ? '😊' : '😞';
    document.getElementById('sentiment').textContent  = data.sentiment;
    document.getElementById('confidence').textContent = `Confidence: ${data.confidence}%`;
    result.style.display = 'flex';
  } catch (e) {
    alert('Error reaching the server.');
  } finally {
    btn.disabled          = false;
    label.style.display   = 'block';
    spinner.style.display = 'none';
  }
}

document.getElementById('txt').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) classify();
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
