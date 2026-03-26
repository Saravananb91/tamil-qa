"""
preprocessing/tokenizer_trainer.py
------------------------------------
Train and load a BPE SentencePiece tokenizer for Tamil text.

Usage
-----
    python -m preprocessing.tokenizer_trainer --corpus data/tamil_corpus.txt
"""

import argparse
import sentencepiece as spm

from configs.config import (
    TOKENIZER_PREFIX, TOKENIZER_MODEL,
    VOCAB_SIZE, CHAR_COVERAGE
)


def train_tokenizer(
    corpus_file: str,
    model_prefix: str = TOKENIZER_PREFIX,
    vocab_size:   int = VOCAB_SIZE,
) -> spm.SentencePieceProcessor:
    """
    Train a BPE SentencePiece tokenizer on the given corpus.

    Parameters
    ----------
    corpus_file  : path to a plain-text Tamil corpus (one line per document)
    model_prefix : output file prefix (produces .model and .vocab files)
    vocab_size   : BPE vocabulary size

    Returns
    -------
    Loaded SentencePieceProcessor ready for encoding/decoding.
    """
    print(f"🔤 Training SentencePiece tokenizer (vocab_size={vocab_size})…")
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=CHAR_COVERAGE,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    print(f"✅ Tokenizer saved: {model_prefix}.model")
    return load_tokenizer(f"{model_prefix}.model")


def load_tokenizer(model_path: str = TOKENIZER_MODEL) -> spm.SentencePieceProcessor:
    """
    Load an existing SentencePiece model from disk.

    Parameters
    ----------
    model_path : path to the .model file

    Returns
    -------
    SentencePieceProcessor
    """
    sp = spm.SentencePieceProcessor(model_file=model_path)
    print(f"✅ Tokenizer loaded from: {model_path}  (vocab={sp.vocab_size()})")
    return sp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tamil SentencePiece tokenizer")
    parser.add_argument("--corpus", required=True, help="Path to corpus .txt file")
    parser.add_argument("--prefix", default=TOKENIZER_PREFIX, help="Model output prefix")
    parser.add_argument("--vocab",  type=int, default=VOCAB_SIZE, help="Vocabulary size")
    args = parser.parse_args()

    train_tokenizer(args.corpus, args.prefix, args.vocab)
