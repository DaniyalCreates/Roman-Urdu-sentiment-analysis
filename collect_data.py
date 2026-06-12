"""
collect_data.py — YouTube Roman Urdu comment collector + sentiment predictor

Usage:
    python collect_data.py

Reads YOUTUBE_API_KEY from .env.
Outputs:
    collected_comments.csv          — filtered Roman Urdu comments
    collected_with_predictions.csv  — same + sentiment & confidence columns
"""

import os
import re
import time
import datetime

import pandas as pd
import torch
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

API_KEY   = os.getenv("YOUTUBE_API_KEY")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "xlmroberta_finetuned")

MAX_VIDEOS_PER_TERM   = 5    # videos fetched per search query
MAX_COMMENTS_PER_VIDEO = 100  # top-level comments per video

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

# ── Roman Urdu detection ───────────────────────────────────────────────────────

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
    r'zyada|thora|thoda|pata|nahi|pata|samajh|hua|hui|hoga|hogi'
    r')\b',
    re.IGNORECASE,
)


def is_roman_urdu(text: str) -> bool:
    """True if text is Latin-script containing common Urdu vocabulary."""
    text = text.strip()
    if len(text) < 6:
        return False
    if _ARABIC_RE.search(text):   # Nastaliq script → not Roman Urdu
        return False
    return bool(_URDU_RE.search(text))


# ── YouTube helpers ────────────────────────────────────────────────────────────

def build_youtube():
    if not API_KEY:
        raise ValueError(
            "YOUTUBE_API_KEY not found. Create a .env file with:\n"
            "  YOUTUBE_API_KEY=your_key_here"
        )
    return build("youtube", "v3", developerKey=API_KEY)


def search_videos(youtube, query: str, max_results: int = MAX_VIDEOS_PER_TERM):
    """Return [(video_id, title), ...] for a search query."""
    try:
        resp = (
            youtube.search()
            .list(
                q=query,
                part="snippet",
                type="video",
                maxResults=max_results,
                relevanceLanguage="ur",
            )
            .execute()
        )
    except HttpError as e:
        print(f"    [search error] '{query}': {e}")
        return []

    return [
        (item["id"]["videoId"], item["snippet"]["title"])
        for item in resp.get("items", [])
    ]


def fetch_comments(youtube, video_id: str, max_comments: int = MAX_COMMENTS_PER_VIDEO):
    """Return list of top-level comment strings for a video."""
    comments = []
    page_token = None

    while len(comments) < max_comments:
        try:
            kwargs = dict(
                videoId=video_id,
                part="snippet",
                maxResults=min(100, max_comments - len(comments)),
                textFormat="plainText",
                order="relevance",
            )
            if page_token:
                kwargs["pageToken"] = page_token
            resp = youtube.commentThreads().list(**kwargs).execute()
        except HttpError as e:
            if e.resp.status not in (403, 404):
                print(f"    [comment error] {video_id}: {e}")
            break  # disabled comments or quota hit — move on

        for item in resp.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(snippet["textDisplay"])

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return comments


# ── Sentiment prediction ───────────────────────────────────────────────────────

def load_model():
    print("\nLoading XLM-RoBERTa model…")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    mdl.eval()
    print("Model ready.")
    return tok, mdl


def predict_batch(texts, tokenizer, model, batch_size=32):
    sentiments, confidences = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1)
        preds = probs.argmax(dim=1).tolist()
        confs = probs.max(dim=1).values.tolist()
        sentiments.extend(["Positive" if p == 1 else "Negative" for p in preds])
        confidences.extend([round(c * 100, 1) for c in confs])
    return sentiments, confidences


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    youtube     = build_youtube()
    date_today  = datetime.date.today().isoformat()
    rows        = []
    seen        = set()
    total_fetched = 0

    for category, terms in SEARCH_TERMS.items():
        print(f"\n── {category.upper()} ──────────────────────────────────")
        for term in terms:
            print(f"  Searching: \"{term}\"")
            videos = search_videos(youtube, term)
            time.sleep(0.5)

            for vid_id, title in videos:
                print(f"    [{vid_id}] {title[:70]}")
                comments = fetch_comments(youtube, vid_id)
                total_fetched += len(comments)
                kept = 0

                for text in comments:
                    text = text.strip()
                    if text in seen:
                        continue
                    seen.add(text)
                    if is_roman_urdu(text):
                        rows.append({
                            "text":           text,
                            "video_title":    title,
                            "date_collected": date_today,
                        })
                        kept += 1

                print(f"      fetched {len(comments)} comments, kept {kept} Roman Urdu")
                time.sleep(0.3)

    # ── Save raw collection ────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Total fetched  : {total_fetched}")
    print(f"Kept (filtered): {len(rows)}")

    if not rows:
        print("No Roman Urdu comments found. Check search terms or API key.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("collected_comments.csv", index=False)
    print(f"Saved collected_comments.csv ({len(df)} rows)")

    # ── Run sentiment predictions ──────────────────────────────────────────────
    tokenizer, model = load_model()
    print(f"Running inference on {len(df)} comments…")
    sentiments, confidences = predict_batch(df["text"].tolist(), tokenizer, model)

    df["sentiment"]  = sentiments
    df["confidence"] = confidences
    df.to_csv("collected_with_predictions.csv", index=False)
    print(f"Saved collected_with_predictions.csv")

    # ── Summary ────────────────────────────────────────────────────────────────
    pos = (df["sentiment"] == "Positive").sum()
    neg = (df["sentiment"] == "Negative").sum()

    print(f"\n{'═'*50}")
    print(f"  Comments collected  : {len(df)}")
    print(f"  Positive            : {pos:>5}  ({pos / len(df) * 100:.1f}%)")
    print(f"  Negative            : {neg:>5}  ({neg / len(df) * 100:.1f}%)")
    print(f"{'═'*50}")


if __name__ == "__main__":
    main()
