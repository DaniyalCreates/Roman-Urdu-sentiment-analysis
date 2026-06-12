"""
weekly_report.py — weekly Roman Urdu sentiment evaluation agent

Designed to run on GitHub Actions (no local model files required).
Loads the fine-tuned model directly from Hugging Face Hub.

Outputs:
    reports/report_YYYY-MM-DD.md   — markdown report committed back to the repo
"""

import os
import re
import time
import datetime
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

API_KEY  = os.getenv("YOUTUBE_API_KEY")
MODEL_ID = "DaniyalCreates/roman-urdu-sentiment-xlmroberta"  # loaded from HF Hub

MAX_VIDEOS_PER_TERM    = 5
MAX_COMMENTS_PER_VIDEO = 100

SEARCH_TERMS = {
    "dramas": [
        "Pakistani drama 2024",
        "Pakistani drama review ARY",
        "best Pakistani drama Geo",
        "Pakistani drama 2025 HUM TV",
        "Pakistani drama emotional scene",
        "drama OST Pakistani",
    ],
    "cricket": [
        "Pakistan cricket 2024",
        "Pakistan vs cricket highlights",
        "PCB cricket news Pakistan",
        "Pakistan T20 cricket 2025",
        "Babar Azam batting highlights",
        "Pakistan Super League PSL 2025",
    ],
    "news": [
        "Pakistan news today ARY",
        "Geo news Pakistan latest",
        "Pakistan breaking news 2024",
        "Samaa news Pakistan",
        "Pakistan economy news Urdu",
    ],
    "vlogs": [
        "Pakistan vlog 2024",
        "Karachi vlog Pakistan",
        "Lahore vlog Pakistan",
        "Islamabad vlog Pakistan",
        "Pakistani street food vlog",
        "Pakistan travel 2025",
    ],
    "music": [
        "Coke Studio Pakistan 2024",
        "Pakistani song new 2024",
        "Pakistani OST best",
        "nescafe basement Pakistan",
        "Pakistani music reaction",
    ],
    "comedy": [
        "Pakistani comedy video",
        "Ducky Bhai Pakistan",
        "Pakistani funny clips",
        "Totla Pakistan comedy",
        "Mooroo Pakistan vlog",
    ],
    "food": [
        "Pakistani street food Karachi",
        "Pakistani food review",
        "desi food Pakistan cooking",
        "Pakistani biryani recipe",
    ],
}

# ── Roman Urdu detection (identical to collect_data.py) ───────────────────────

_ARABIC_RE = re.compile(r'[؀-ۿݐ-ݿࢠ-ࣿ]')

_URDU_RE = re.compile(
    r'\b('
    r'hai|hain|nahi|nahin|na|acha|accha|achi|acchi|yaar|bohat|bahut|'
    r'kya|mein|main|mujhe|hum|tum|aap|tumhara|tumhari|apna|apni|'
    r'bhai|dost|yar|aur|lekin|magar|kyun|kyunke|phir|abhi|kal|aaj|aj|'
    r'bilkul|zaroor|zarur|theek|thik|sahi|galat|bura|buri|'
    r'tera|mera|teri|meri|woh|wo|yeh|ye|jo|jab|tab|toh|to|'
    r'se|ke|ki|ka|ne|pe|par|liye|wala|wali|'
    r'dekho|dekh|suno|sun|laga|lagta|lagti|karo|karna|karta|karti|'
    r'pasand|dil|zindagi|maza|mazaa|pyar|mohabbat|'
    r'mashallah|subhanallah|inshallah|alhamdulillah|'
    r'zyada|thora|thoda|pata|samajh|hua|hui|hoga|hogi'
    r')\b',
    re.IGNORECASE,
)


def is_roman_urdu(text: str) -> bool:
    text = text.strip()
    if len(text) < 6:
        return False
    if _ARABIC_RE.search(text):
        return False
    return bool(_URDU_RE.search(text))


# ── YouTube helpers ────────────────────────────────────────────────────────────

def build_youtube():
    if not API_KEY:
        raise ValueError("YOUTUBE_API_KEY environment variable not set.")
    return build("youtube", "v3", developerKey=API_KEY)


def search_videos(youtube, query: str, max_results: int = MAX_VIDEOS_PER_TERM):
    try:
        resp = (
            youtube.search()
            .list(q=query, part="snippet", type="video",
                  maxResults=max_results, relevanceLanguage="ur")
            .execute()
        )
    except HttpError as e:
        print(f"    [search error] '{query}': {e}")
        return []
    return [(item["id"]["videoId"], item["snippet"]["title"])
            for item in resp.get("items", [])]


def fetch_comments(youtube, video_id: str, max_comments: int = MAX_COMMENTS_PER_VIDEO):
    comments, page_token = [], None
    while len(comments) < max_comments:
        try:
            kwargs = dict(videoId=video_id, part="snippet",
                          maxResults=min(100, max_comments - len(comments)),
                          textFormat="plainText", order="relevance")
            if page_token:
                kwargs["pageToken"] = page_token
            resp = youtube.commentThreads().list(**kwargs).execute()
        except HttpError as e:
            if e.resp.status not in (403, 404):
                print(f"    [comment error] {video_id}: {e}")
            break
        for item in resp.get("items", []):
            comments.append(
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            )
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return comments


# ── Model helpers ──────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading model from Hub: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    mdl.eval()
    print("Model ready.")
    return tok, mdl


def predict_batch(texts, tokenizer, model, batch_size=32):
    sentiments, confidences = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                           padding=True, max_length=128)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1)
        preds = probs.argmax(dim=1).tolist()
        confs = probs.max(dim=1).values.tolist()
        sentiments.extend(["Positive" if p == 1 else "Negative" for p in preds])
        confidences.extend([round(c * 100, 1) for c in confs])
    return sentiments, confidences


# ── Report generation ──────────────────────────────────────────────────────────

def generate_report(df: pd.DataFrame, total_fetched: int,
                    category_stats: dict, date: str) -> str:
    kept        = len(df)
    pass_rate   = kept / total_fetched * 100 if total_fetched else 0
    pos         = (df["sentiment"] == "Positive").sum()
    neg         = (df["sentiment"] == "Negative").sum()

    # Confidence bands
    bands = [
        ("90%+ (high confidence)", 90,  101),
        ("70–90%",                 70,  90),
        ("60–70%",                 60,  70),
        ("50–60% (low confidence)",50,  60),
    ]
    band_rows = ""
    for label, lo, hi in bands:
        n   = int(((df["confidence"] >= lo) & (df["confidence"] < hi)).sum())
        pct = n / kept * 100 if kept else 0
        band_rows += f"| {label} | {n:,} | {pct:.1f}% |\n"

    # 10 lowest-confidence comments
    low10 = df.nsmallest(10, "confidence")[["confidence", "sentiment", "text"]]
    low_rows = ""
    for _, row in low10.iterrows():
        text = row["text"].replace("|", "\\|").replace("\n", " ")[:90]
        low_rows += f"| {row['confidence']}% | {row['sentiment']} | {text} |\n"

    # Top categories by density, sorted descending
    cat_rows = ""
    for cat, stats in sorted(category_stats.items(),
                              key=lambda x: x[1]["kept"] / max(x[1]["fetched"], 1),
                              reverse=True):
        f = stats["fetched"]
        k = stats["kept"]
        d = k / f * 100 if f else 0
        cat_rows += f"| {cat} | {f:,} | {k:,} | {d:.1f}% |\n"

    report = f"""# Weekly Roman Urdu Sentiment Report — {date}

**Model:** [DaniyalCreates/roman-urdu-sentiment-xlmroberta](https://huggingface.co/DaniyalCreates/roman-urdu-sentiment-xlmroberta)
**Generated:** {date}

---

## 📊 Collection Summary

| Metric | Value |
|--------|-------|
| Total comments fetched | {total_fetched:,} |
| Kept after Roman Urdu filter | {kept:,} ({pass_rate:.1f}%) |

## 😊 Sentiment Split

| Sentiment | Count | Share |
|-----------|-------|-------|
| Positive | {pos:,} | {pos/kept*100:.1f}% |
| Negative | {neg:,} | {neg/kept*100:.1f}% |

## 🎯 Confidence Distribution

| Band | Count | Share |
|------|-------|-------|
{band_rows}
## ⚠️ 10 Lowest-Confidence Predictions

| Conf. | Sentiment | Comment |
|-------|-----------|---------|
{low_rows}
## 🏆 Top Categories by Roman Urdu Density

| Category | Fetched | Kept | Density |
|----------|---------|------|---------|
{cat_rows}
---
*Report auto-generated by `weekly_report.py` and committed by GitHub Actions.*
"""
    return report


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    date_today     = datetime.date.today().isoformat()
    youtube        = build_youtube()
    rows           = []
    seen           = set()
    total_fetched  = 0
    category_stats = {cat: {"fetched": 0, "kept": 0} for cat in SEARCH_TERMS}

    for category, terms in SEARCH_TERMS.items():
        print(f"\n── {category.upper()} ──")
        for term in terms:
            print(f"  Searching: \"{term}\"")
            videos = search_videos(youtube, term)
            time.sleep(0.5)

            for vid_id, title in videos:
                print(f"    [{vid_id}] {title[:70]}")
                comments = fetch_comments(youtube, vid_id)
                category_stats[category]["fetched"] += len(comments)
                total_fetched += len(comments)
                kept = 0

                for text in comments:
                    text = text.strip()
                    if text in seen:
                        continue
                    seen.add(text)
                    if is_roman_urdu(text):
                        rows.append({"text": text, "video_title": title,
                                     "category": category,
                                     "date_collected": date_today})
                        kept += 1

                category_stats[category]["kept"] += kept
                print(f"      {len(comments)} fetched, {kept} kept")
                time.sleep(0.3)

    print(f"\nTotal fetched: {total_fetched} | Kept: {len(rows)}")

    if not rows:
        print("No Roman Urdu comments collected. Exiting.")
        return

    df = pd.DataFrame(rows)

    # Classify
    tokenizer, model = load_model()
    print(f"Running inference on {len(df)} comments…")
    sentiments, confidences = predict_batch(df["text"].tolist(), tokenizer, model)
    df["sentiment"]  = sentiments
    df["confidence"] = confidences

    # Write report
    Path("reports").mkdir(exist_ok=True)
    report_path = f"reports/report_{date_today}.md"
    report_text = generate_report(df, total_fetched, category_stats, date_today)
    Path(report_path).write_text(report_text, encoding="utf-8")
    print(f"Report written: {report_path}")

    # Print summary
    pos = (df["sentiment"] == "Positive").sum()
    neg = (df["sentiment"] == "Negative").sum()
    print(f"\n{'═'*50}")
    print(f"  Collected  : {len(df)}")
    print(f"  Positive   : {pos} ({pos/len(df)*100:.1f}%)")
    print(f"  Negative   : {neg} ({neg/len(df)*100:.1f}%)")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()
