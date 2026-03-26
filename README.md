# 🗣️ Tamil Hybrid Question Answering System

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scrapy](https://img.shields.io/badge/Scrapy-2.11-60A917?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Research Project** — A hybrid Tamil QA system combining extractive span prediction (mBERT), generative answer synthesis (Seq2Seq LSTM), and a custom Tamil Transformer trained from scratch with SentencePiece BPE tokenisation.

---

## 🧠 System Overview

This system answers questions in **Tamil** using three complementary strategies:

| Strategy | Model | When used |
|----------|-------|-----------|
| **Extractive QA** | `bert-base-multilingual-cased` | Primary — finds answer span in a given context |
| **Transformer QA** | Custom Tamil Transformer | Fallback 1 — span prediction from question alone |
| **Generative QA** | Seq2Seq LSTM (Keras) | Fallback 2 — generates answer token-by-token |

The hybrid approach ensures maximum coverage: when the extractive model fails to find a confident span, the system gracefully falls back to generation.

---

## 🏗️ Project Structure

```
tamil-hybrid-qa/
│
├── chatbot/
│   ├── hybrid_qa.py              # Hybrid QA engine (orchestrates all models)
│   └── chat.py                   # Interactive CLI chatbot
│
├── configs/
│   └── config.py                 # All hyperparameters, paths, model names
│
├── data/
│   └── utils/
│       └── convert_dataset.py    # Convert .txt / JSONL → standard QA JSON
│
├── models/
│   ├── transformer_qa.py         # Custom Tamil Transformer model (PyTorch)
│   ├── seq2seq_qa.py             # Seq2Seq LSTM model (Keras/TF)
│   ├── extractive_qa.py          # mBERT fine-tuning + inference
│   └── train_transformer.py      # Training script for the Transformer model
│
├── preprocessing/
│   ├── text_cleaner.py           # Tamil Unicode filtering + grammar correction
│   ├── tokenizer_trainer.py      # SentencePiece BPE tokenizer training
│   └── dataset.py                # PyTorch Dataset + collate_fn
│
├── scraper/
│   └── spider.py                 # Scrapy spider (Dinamalar Tamil news)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔄 Architecture

```
Tamil Question (Input)
        │
        ▼
┌─────────────────────────────────┐
│  preprocessing/text_cleaner.py  │
│  • Unicode NFKC normalisation   │
│  • Tamil script filtering       │
│  • LanguageTool grammar check   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│               Hybrid QA Engine (chatbot/hybrid_qa.py)   │
│                                                         │
│  ┌──────────────────────────────────────────────┐       │
│  │  1. mBERT Extractive QA (PRIMARY)            │       │
│  │     bert-base-multilingual-cased             │       │
│  │     → Span prediction on context passages    │       │
│  └──────────────────┬───────────────────────────┘       │
│                     │ No valid answer?                   │
│                     ▼                                    │
│  ┌──────────────────────────────────────────────┐       │
│  │  2. Custom Transformer QA (FALLBACK 1)       │       │
│  │     SentencePiece BPE + Transformer Encoder  │       │
│  │     → Span prediction from question tokens   │       │
│  └──────────────────┬───────────────────────────┘       │
│                     │ Still no answer?                   │
│                     ▼                                    │
│  ┌──────────────────────────────────────────────┐       │
│  │  3. Seq2Seq LSTM (FALLBACK 2)                │       │
│  │     Encoder-Decoder LSTM (Keras)             │       │
│  │     → Generates answer token by token        │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
               │
               ▼
         Tamil Answer
```

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/your-username/tamil-hybrid-qa.git
cd tamil-hybrid-qa
```

### 2. Virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Stanza Tamil model

```python
import stanza
stanza.download("ta")
```

---

## 📂 Data Preparation

Place your QA JSON files under `data/`. Expected format:

```json
[
  {
    "question": "சோலார் மண்டலத்தின் மையத்தில் உள்ள நட்சத்திரம் எது?",
    "answers":  ["சூரியன்"],
    "context":  "சூரியன் சோலார் மண்டலத்தின் மையத்தில் உள்ள நட்சத்திரமாகும்."
  }
]
```

To convert raw `.txt` or `.jsonl` files:

```bash
# From tab-separated text
python -m data.utils.convert_dataset --mode txt   --input dev_doc.txt  --output data/qa_data.json

# From JSONL
python -m data.utils.convert_dataset --mode jsonl --input test.jsonl   --output data/qa_data.json
```

---

## 🕷️ Scraping Tamil Corpus

```bash
scrapy crawl tamil_news -o data/scraped_articles.json
```

Use the scraped `content` fields to build a larger training corpus.

---

## 🚀 Training

### Train the custom Transformer model

```bash
python -m models.train_transformer --data data/qa_data.json --epochs 5
```

### Fine-tune mBERT

```python
from models.extractive_qa import finetune
import json

with open("data/qa_data.json") as f:
    qa_data = json.load(f)

finetune(qa_data, epochs=3, device="cuda")
```

---

## 💬 Running the Chatbot

```bash
python -m chatbot.chat \
    --model outputs/tamil_qa_transformer.pth \
    --data  data/qa_data.json \
    --spm   tamil_tokenizer.model
```

**Interactive session:**
```
─────────────────────────────────────────────────────
  தமிழ் கேள்வி-பதிலி மென்பொருள் (Tamil QA Chatbot)
─────────────────────────────────────────────────────
  Type 'exit' to quit.

நீங்கள்: சூரியன் என்ன?
பதில்: சூரியன் சோலார் மண்டலத்தின் மையத்தில் உள்ள நட்சத்திரமாகும்.
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Custom Transformer training & inference |
| `transformers` | mBERT fine-tuning + HuggingFace tokenizer |
| `sentencepiece` | Tamil BPE tokenizer |
| `stanza` | Tamil NLP pipeline (tokenisation, POS) |
| `tensorflow` / `keras` | Seq2Seq LSTM model |
| `scrapy` | Tamil news web scraping |
| `language-tool-python` | Tamil grammar correction |

---

## 🔮 Roadmap

- [ ] Replace mBERT with [IndicBERT](https://huggingface.co/ai4bharat/indic-bert) for better Tamil coverage
- [ ] Add Streamlit web interface
- [ ] Evaluate with Exact Match (EM) and F1 metrics
- [ ] Publish trained models to HuggingFace Hub
- [ ] Docker containerisation

---

## 📝 Dataset Format Notes

The system expects `answers` as a **list** (supporting multiple reference answers for evaluation). During inference, `answers[0]` is used as the primary target.

---

## 👤 Author

## Saravanan.B

[![Portfolio](https://img.shields.io/badge/🌐%20Portfolio-Visit%20Now-6366f1?style=for-the-badge)](https://v0-portfolio-saravanan-b.vercel.app/)

[![Email](https://img.shields.io/badge/📧%20Email%20Me-D14836?style=for-the-badge)](mailto:Mrsaravananb@gmail.com)

[![LinkedIn](https://img.shields.io/badge/🔗%20Connect-0077B5?style=for-the-badge)](https://www.linkedin.com/in/saravanan-b-46244b290)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
