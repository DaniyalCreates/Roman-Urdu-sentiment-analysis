# Sentiment Analysis of Roman Urdu Social Media Text
### MSc Dissertation — Artificial Intelligence & Data Science | University of Hull (2025)

## 📌 Project Overview
Roman Urdu is a low-resource language written in Latin script, widely used 
across South Asian social media platforms but severely underrepresented in 
NLP research. This project builds and compares two deep learning models for 
sentiment classification on Roman Urdu social media text — addressing a 
real-world NLP challenge with no standardised corpus or pre-trained resources.

## 🧠 Models Built & Compared
| Model | Accuracy |
|-------|----------|
| Bi-LSTM (baseline) | ~80% |
| XLM-RoBERTa (fine-tuned) | 82% |

XLM-RoBERTa outperformed the Bi-LSTM baseline by ~15% relative improvement, 
demonstrating the viability of multilingual transformer models for 
low-resource language sentiment analysis.

## 🗂️ Repository Structure
| Notebook | Description |
|----------|-------------|
| `Notebook1Final_Data_Preprocessing` | Data cleaning, normalisation, and tokenisation of raw social media text |
| `Notebook2Final_BiLSTM` | Architecture, training and evaluation of the Bi-LSTM baseline model |
| `Notebook3Final_XLM_RoBERTa` | Fine-tuning XLM-RoBERTa using HuggingFace Transformers on custom dataset |
| `Notebook4Final_Stress_Test` | Robustness testing and stress evaluation of both models |

## 🛠️ Tech Stack
- **Language:** Python
- **Frameworks:** PyTorch, HuggingFace Transformers
- **Models:** XLM-RoBERTa, Bi-LSTM
- **Libraries:** pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- **Tools:** Jupyter Notebook, Git

## 📊 Key Results
- Fine-tuned XLM-RoBERTa achieved **82% classification accuracy**
- Outperformed Bi-LSTM baseline by a **15% relative improvement**
- Validated using cross-validation and confusion matrix analysis
- Demonstrated strong generalisation on unseen social media text

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