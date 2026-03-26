"""
configs/config.py
-----------------
Centralised configuration for the Tamil Hybrid QA System.
All file paths and hyperparameters are defined here.
Update paths to match your local environment.
"""

import os

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = os.getenv("DATA_DIR", "data")
QA_DATA_FILE    = os.path.join(DATA_DIR, "qa_data.json")       # main QA dataset
CORPUS_FILE     = os.path.join(DATA_DIR, "tamil_corpus.txt")   # raw text for tokenizer
DEV_FILE        = os.path.join(DATA_DIR, "dev.json")
TEST_FILE       = os.path.join(DATA_DIR, "test.json")

# ── Scraper ───────────────────────────────────────────────────────────────────
SCRAPER_START_URL  = "https://www.dinamalar.com/"
SCRAPER_OUTPUT     = os.path.join(DATA_DIR, "scraped_articles.json")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
TOKENIZER_PREFIX   = "tamil_tokenizer"
TOKENIZER_MODEL    = f"{TOKENIZER_PREFIX}.model"
VOCAB_SIZE         = 1000          # BPE vocab size for SentencePiece
CHAR_COVERAGE      = 0.9995

# ── Sequence lengths ──────────────────────────────────────────────────────────
SEQ_LENGTH         = 128
MAX_BERT_LENGTH    = 512

# ── Custom Transformer model ──────────────────────────────────────────────────
D_MODEL            = 128
N_HEAD             = 4
NUM_ENCODER_LAYERS = 2
DIM_FEEDFORWARD    = 512

# ── Seq2Seq LSTM model ────────────────────────────────────────────────────────
EMBEDDING_DIM      = 256
HIDDEN_UNITS       = 512
MAX_QUESTION_LEN   = 50
MAX_ANSWER_LEN     = 50

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE         = 4
EPOCHS             = 5
LEARNING_RATE      = 1e-4
GRAD_CLIP          = 1.0
SCHEDULER_STEP     = 3
SCHEDULER_GAMMA    = 0.1

# ── mBERT extractive QA ───────────────────────────────────────────────────────
MBERT_MODEL_NAME   = "bert-base-multilingual-cased"

# ── Model save paths ──────────────────────────────────────────────────────────
TRANSFORMER_SAVE   = "outputs/tamil_qa_transformer.pth"
SEQ2SEQ_SAVE       = "outputs/tamil_qa_seq2seq.h5"
MBERT_SAVE_DIR     = "outputs/tamil_qa_mbert"
TOKENIZER_SAVE_DIR = "outputs/tamil_qa_tokenizer"
