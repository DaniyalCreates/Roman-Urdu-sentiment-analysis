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
