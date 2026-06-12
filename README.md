# Sentiment Analysis of Roman Urdu Social Media Text
### MSc Dissertation — Artificial Intelligence & Data Science | University of Hull (2025)

## 🚀 Live Demo

**[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/DaniyalCreates/roman-urdu-sentiment)**

## 📌 Project Overview
Roman Urdu is a low-resource language written in Latin script, widely used 
across South Asian social media platforms but severely underrepresented in 
NLP research. This project builds and compares two deep learning models for 
sentiment classification on Roman Urdu social media text — addressing a 
real-world NLP challenge with no standardised corpus or pre-trained resources.

## 🧠 Models Built & Compared
| Model | Accuracy |
|-------|----------|
| Bi-LSTM (baseline) | ~79% |
| [XLM-RoBERTa (fine-tuned)](https://huggingface.co/DaniyalCreates/roman-urdu-sentiment-xlmroberta) | 82% |

XLM-RoBERTa outperformed the Bi-LSTM baseline, demonstrating the viability
of multilingual transformer models for low-resource language sentiment analysis.

## 🗂️ Repository Structure

### Notebooks
| Notebook | Description |
|----------|-------------|
| `Notebook1Final_Data_Preprocessing` | Data cleaning, normalisation, and label encoding of raw social media text |
| `Notebook2Final_BiLSTM` | Architecture, training, and evaluation of the Bi-LSTM baseline model |
| `Notebook3Final_XLM_RoBERTa` | Fine-tuning XLM-RoBERTa using HuggingFace Transformers |
| `Notebook3_XLM_RoBERTa_Colab.ipynb` | Colab-ready version — mounts Google Drive, saves and zips the fine-tuned model for download |
| `Notebook4Final_Stress_Test` | Cross-domain robustness test against the RU-EN Emotion Dataset |

### Scripts
| File | Description |
|------|-------------|
| `run_bilstm.py` | Runs Notebook 2 end-to-end locally against `rusad_cleaned.csv` |
| `run_stress_test.py` | Runs Notebook 4 end-to-end locally against the emotion dataset |
| `test_xlmroberta.py` | Loads the fine-tuned model and classifies a set of example sentences |
| `app.py` | FastAPI demo app — serves a single-page UI for live sentiment inference |
| `collect_data.py` | YouTube data collection agent — searches Pakistani content, filters Roman Urdu comments, and runs sentiment predictions |
| `annotate.py` | Local annotation tool — label low-confidence comments via a keyboard-driven web UI, saving results to `annotations.csv` |
| `weekly_report.py` | Weekly evaluation agent — collects, classifies, and commits a markdown report; run by GitHub Actions every Monday |

## 🏷️ Annotation Tool

`annotate.py` is a FastAPI web app for manually labelling the 432 low-confidence comments identified during analysis.

- Loads `low_confidence_comments.csv` and presents one comment at a time
- Shows the model's prediction and confidence as context
- Four label buttons with **keyboard shortcuts** for fast labelling:

| Key | Label |
|-----|-------|
| `P` | Positive |
| `N` | Negative |
| `U` | Neutral |
| `S` | Sarcastic / Mixed |
| `Z` | Undo last label |

- Progress bar (e.g. 37 / 432) and immediate save to `annotations.csv`
- Automatically resumes where you left off on restart

**Run:**
```bash
uvicorn annotate:app --port 8001
```
Then open **http://localhost:8001**

The resulting `annotations.csv` (text, model_prediction, model_confidence, human_label) forms the candidate training set for a future neutral-class model extension.

## 📡 Data Collection

`collect_data.py` uses the YouTube Data API v3 to gather real-world Roman Urdu comments for further analysis or model retraining.

- Searches across 7 categories: **dramas, cricket, news, vlogs, music, comedy, food** (configurable `SEARCH_TERMS` dict)
- Filters to Roman Urdu: Latin script + presence of common Urdu vocabulary markers
- Deduplicates across all fetched videos in a run
- Saves `collected_comments.csv` (text, video\_title, date\_collected)
- Runs `xlmroberta_finetuned` inference and saves `collected_with_predictions.csv`
- Prints a summary: total fetched, kept after filtering, Positive/Negative split

**Setup:**
```bash
cp .env.example .env          # then paste your YouTube Data API v3 key inside
python collect_data.py
```

> A free YouTube Data API v3 key gives 10,000 units/day. One full run costs ~1,300 units.

## 🤖 Automated Weekly Reports

Every **Monday at 09:00 UTC** a GitHub Actions workflow runs `weekly_report.py`, which:

1. Collects fresh Roman Urdu comments from YouTube across all 7 categories
2. Loads the fine-tuned model directly from [Hugging Face Hub](https://huggingface.co/DaniyalCreates/roman-urdu-sentiment-xlmroberta) — no local files needed
3. Classifies every comment and generates a markdown report in `reports/report_YYYY-MM-DD.md` containing:
   - Total collected and pass rate
   - Positive / Negative sentiment split
   - Confidence distribution (50–60% / 60–70% / 70–90% / 90%+)
   - 10 lowest-confidence predictions
   - Per-category Roman Urdu density table
4. Commits and pushes the report back to this repository automatically

Reports are stored in the [`reports/`](reports/) folder and build up a longitudinal view of sentiment trends in Pakistani social media content.

The workflow requires a `YOUTUBE_API_KEY` repository secret (Settings → Secrets and variables → Actions). It can also be triggered manually from the [Actions tab](../../actions/workflows/weekly_report.yml).

## 🚀 Demo App

A lightweight web app lets you type Roman Urdu text and get an instant
Positive / Negative prediction with confidence score.

**Setup:**
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
Then open **http://localhost:8000** in your browser.

> The fine-tuned model (`xlmroberta_finetuned/`) must be present in the
> project root. It is excluded from version control due to size — download
> it from the [Hugging Face model repo](https://huggingface.co/DaniyalCreates/roman-urdu-sentiment-xlmroberta)
> or the Colab notebook and unzip it here.

## 🛠️ Tech Stack
- **Language:** Python 3.9
- **Frameworks:** PyTorch, HuggingFace Transformers, TensorFlow / Keras
- **Models:** XLM-RoBERTa (`xlm-roberta-base`), Bi-LSTM
- **Web:** FastAPI, Uvicorn
- **Libraries:** pandas, NumPy, scikit-learn, Matplotlib, Seaborn, openpyxl
- **Tools:** Jupyter Notebook, Google Colab, Git

## 📊 Key Results
- Fine-tuned XLM-RoBERTa achieved **82% test accuracy** on the RUSAD dataset
- Bi-LSTM baseline reached **79% test accuracy**
- Stress test on the RU-EN Emotion Dataset (8,147 tweets): **79% accuracy**,
  confirming cross-domain generalisation
- 1,726 misclassified examples saved to `misclassified_examples.csv` for
  error analysis

## 💡 Why This Matters
Most NLP research focuses on high-resource languages like English. 
Roman Urdu represents hundreds of millions of speakers whose digital 
communication is largely invisible to AI systems. This work demonstrates 
that multilingual transformer models can be effectively fine-tuned for 
low-resource, code-switched languages — with direct applications in 
customer feedback analysis, content moderation, and brand monitoring.

## 👤 Author
**Daniyal Tariq**  
MSc Artificial Intelligence & Data Science — University of Hull  
📧 tariqdaniyal048@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/daniyal-tariq-771115185/)
