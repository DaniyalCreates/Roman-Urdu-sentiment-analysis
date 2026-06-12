"""
annotate.py — local annotation tool for Roman Urdu comments

Usage:
    python annotate.py

Loads:  low_confidence_comments.csv
Saves:  annotations.csv  (resumes automatically if it already exists)
URL:    http://localhost:8001
"""

import os
import csv
import threading
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Data setup ─────────────────────────────────────────────────────────────────

COMMENTS_FILE    = "low_confidence_comments.csv"
ANNOTATIONS_FILE = "annotations.csv"
ANNOTATION_COLS  = ["text", "model_prediction", "model_confidence", "human_label"]

if not Path(COMMENTS_FILE).exists():
    raise FileNotFoundError(f"{COMMENTS_FILE} not found. Run collect_data.py first.")

_all = pd.read_csv(COMMENTS_FILE)[["text", "sentiment", "confidence"]].copy()
_all.rename(columns={"sentiment": "model_prediction",
                      "confidence": "model_confidence"}, inplace=True)
_all["text"] = _all["text"].fillna("").str.strip()

# Load existing annotations and resume
_lock = threading.Lock()

def _load_annotations() -> list[dict]:
    if Path(ANNOTATIONS_FILE).exists():
        df = pd.read_csv(ANNOTATIONS_FILE)
        df = df.reindex(columns=ANNOTATION_COLS, fill_value="")
        return df.to_dict("records")
    return []


def _save_annotations(records: list[dict]) -> None:
    df = pd.DataFrame(records, columns=ANNOTATION_COLS)
    df.to_csv(ANNOTATIONS_FILE, index=False)


_annotations: list[dict] = _load_annotations()

# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI()
VALID_LABELS = {"Positive", "Negative", "Neutral", "Sarcastic/Mixed"}


class LabelPayload(BaseModel):
    label: str


def _current_index() -> int:
    return len(_annotations)


@app.get("/api/state")
def get_state():
    idx   = _current_index()
    total = len(_all)
    if idx >= total:
        return {"done": True, "labeled": idx, "total": total}
    row = _all.iloc[idx]
    return {
        "done":             False,
        "text":             row["text"],
        "model_prediction": row["model_prediction"],
        "model_confidence": float(row["model_confidence"]),
        "labeled":          idx,
        "total":            total,
    }


@app.post("/api/label")
def post_label(payload: LabelPayload):
    if payload.label not in VALID_LABELS:
        raise HTTPException(400, f"Invalid label: {payload.label}")
    with _lock:
        idx = _current_index()
        if idx >= len(_all):
            raise HTTPException(400, "All comments already labelled.")
        row = _all.iloc[idx]
        _annotations.append({
            "text":             row["text"],
            "model_prediction": row["model_prediction"],
            "model_confidence": row["model_confidence"],
            "human_label":      payload.label,
        })
        _save_annotations(_annotations)
    return get_state()


@app.post("/api/undo")
def post_undo():
    with _lock:
        if not _annotations:
            raise HTTPException(400, "Nothing to undo.")
        _annotations.pop()
        _save_annotations(_annotations)
    return get_state()


# ── HTML ───────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roman Urdu Annotator</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #e2e8f0;
    padding: 24px;
  }

  .shell {
    width: 100%;
    max-width: 760px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  /* Header row */
  .header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .title { font-size: .85rem; font-weight: 600; color: #94a3b8; letter-spacing: .06em; text-transform: uppercase; }
  .counter { font-size: 1.1rem; font-weight: 700; color: #cbd5e1; }

  /* Progress bar */
  .progress-track {
    width: 100%;
    height: 6px;
    background: #1e293b;
    border-radius: 99px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    border-radius: 99px;
    transition: width .3s ease;
  }

  /* Model badge */
  .meta {
    display: flex;
    gap: 10px;
    align-items: center;
    font-size: .8rem;
  }
  .badge {
    padding: 3px 10px;
    border-radius: 99px;
    font-weight: 600;
    font-size: .75rem;
    letter-spacing: .03em;
  }
  .badge-pos  { background: #14532d; color: #86efac; }
  .badge-neg  { background: #7f1d1d; color: #fca5a5; }
  .conf       { color: #64748b; }

  /* Comment card */
  .card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 28px 32px;
    font-size: 1.25rem;
    line-height: 1.75;
    color: #f1f5f9;
    min-height: 140px;
    word-break: break-word;
    transition: opacity .15s;
  }
  .card.flash { opacity: 0; }

  /* Buttons */
  .buttons {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }
  .btn {
    padding: 14px 8px;
    border: none;
    border-radius: 10px;
    font-size: .95rem;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    transition: transform .08s, filter .08s;
  }
  .btn:hover  { filter: brightness(1.15); }
  .btn:active { transform: scale(.96); }
  .btn kbd {
    font-size: .7rem;
    opacity: .65;
    font-family: monospace;
    font-weight: 400;
  }
  .btn-pos  { background: #16a34a; color: #fff; }
  .btn-neg  { background: #dc2626; color: #fff; }
  .btn-neu  { background: #2563eb; color: #fff; }
  .btn-mix  { background: #9333ea; color: #fff; }

  /* Undo row */
  .undo-row {
    display: flex;
    justify-content: flex-end;
  }
  .btn-undo {
    background: #1e293b;
    border: 1px solid #475569;
    color: #94a3b8;
    padding: 8px 18px;
    border-radius: 8px;
    font-size: .85rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: background .12s;
  }
  .btn-undo:hover { background: #334155; }
  .btn-undo kbd   { font-size: .7rem; opacity: .6; }

  /* Done screen */
  .done {
    text-align: center;
    padding: 48px 0;
  }
  .done h2 { font-size: 2rem; color: #86efac; margin-bottom: 8px; }
  .done p  { color: #94a3b8; }
</style>
</head>
<body>
<div class="shell" id="app">
  <div class="header">
    <span class="title">Roman Urdu Annotator</span>
    <span class="counter" id="counter">— / —</span>
  </div>

  <div class="progress-track">
    <div class="progress-fill" id="progress" style="width:0%"></div>
  </div>

  <div class="meta">
    <span style="color:#64748b;font-size:.8rem">Model predicted:</span>
    <span class="badge" id="model-badge">—</span>
    <span class="conf" id="model-conf"></span>
  </div>

  <div class="card" id="comment-card">Loading…</div>

  <div class="buttons">
    <button class="btn btn-pos" onclick="label('Positive')">
      Positive <kbd>P</kbd>
    </button>
    <button class="btn btn-neg" onclick="label('Negative')">
      Negative <kbd>N</kbd>
    </button>
    <button class="btn btn-neu" onclick="label('Neutral')">
      Neutral <kbd>U</kbd>
    </button>
    <button class="btn btn-mix" onclick="label('Sarcastic/Mixed')">
      Sarcastic / Mixed <kbd>S</kbd>
    </button>
  </div>

  <div class="undo-row">
    <button class="btn-undo" onclick="undo()">↩ Undo <kbd>Z</kbd></button>
  </div>
</div>

<script>
let busy = false;

async function fetchState() {
  const r = await fetch('/api/state');
  return r.json();
}

function renderState(s) {
  if (s.done) {
    document.getElementById('app').innerHTML = `
      <div class="done">
        <h2>✓ All done!</h2>
        <p>Labelled ${s.labeled} comments. Find your annotations in <code>annotations.csv</code>.</p>
      </div>`;
    return;
  }

  document.getElementById('counter').textContent = `${s.labeled + 1} / ${s.total}`;
  document.getElementById('progress').style.width = `${(s.labeled / s.total) * 100}%`;

  const badge = document.getElementById('model-badge');
  badge.textContent = s.model_prediction;
  badge.className = 'badge ' + (s.model_prediction === 'Positive' ? 'badge-pos' : 'badge-neg');
  document.getElementById('model-conf').textContent = `(${s.model_confidence.toFixed(1)}% confidence)`;

  const card = document.getElementById('comment-card');
  card.classList.add('flash');
  setTimeout(() => {
    card.textContent = s.text;
    card.classList.remove('flash');
  }, 120);
}

async function label(value) {
  if (busy) return;
  busy = true;
  try {
    const r = await fetch('/api/label', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: value }),
    });
    renderState(await r.json());
  } finally {
    busy = false;
  }
}

async function undo() {
  if (busy) return;
  busy = true;
  try {
    const r = await fetch('/api/undo', { method: 'POST' });
    if (r.ok) renderState(await r.json());
  } finally {
    busy = false;
  }
}

document.addEventListener('keydown', e => {
  if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
  const map = { p: 'Positive', n: 'Negative', u: 'Neutral', s: 'Sarcastic/Mixed' };
  if (map[e.key.toLowerCase()]) { e.preventDefault(); label(map[e.key.toLowerCase()]); }
  if (e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); }
});

// Boot
fetchState().then(renderState);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
