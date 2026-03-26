"""
models/train_transformer.py
----------------------------
Training script for the custom Tamil Transformer QA model.

Usage
-----
    python -m models.train_transformer --data data/qa_data.json
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import (
    CORPUS_FILE, TOKENIZER_MODEL, VOCAB_SIZE, SEQ_LENGTH,
    D_MODEL, N_HEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, GRAD_CLIP,
    SCHEDULER_STEP, SCHEDULER_GAMMA, TRANSFORMER_SAVE,
)
from preprocessing.text_cleaner  import build_corpus_file
from preprocessing.tokenizer_trainer import train_tokenizer, load_tokenizer
from preprocessing.dataset        import TamilQADataset, collate_fn
from models.transformer_qa        import TamilQAModel


def train(
    qa_data: list,
    epochs:  int   = EPOCHS,
    device:  torch.device = None,
) -> TamilQAModel:
    """
    Full training loop for the Transformer QA model.

    Parameters
    ----------
    qa_data : loaded QA dataset (list of dicts)
    epochs  : number of training epochs
    device  : torch device (auto-detected if None)

    Returns
    -------
    Trained TamilQAModel
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Build corpus and train tokenizer
    build_corpus_file(qa_data, CORPUS_FILE)
    tokenizer = train_tokenizer(CORPUS_FILE, vocab_size=VOCAB_SIZE)

    dataset    = TamilQADataset(qa_data, tokenizer, SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = TamilQAModel(
        vocab_size=tokenizer.vocab_size(),
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_seq_length=SEQ_LENGTH,
    ).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
    loss_fn    = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for src, start_label, end_label in dataloader:
            src          = src.to(device)
            start_label  = start_label.to(device)
            end_label    = end_label.to(device)

            optimizer.zero_grad()
            start_logits, end_logits = model(src)

            # Flatten for loss
            loss = (
                loss_fn(start_logits.view(-1), start_label.view(-1))
                + loss_fn(end_logits.view(-1),   end_label.view(-1))
            ) / 2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(dataloader):.4f}")

    os.makedirs(os.path.dirname(TRANSFORMER_SAVE), exist_ok=True)
    torch.save(model.state_dict(), TRANSFORMER_SAVE)
    print(f"✅ Model saved to: {TRANSFORMER_SAVE}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tamil Transformer QA model")
    parser.add_argument("--data",   required=True, help="Path to QA JSON data file")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        qa_data = json.load(f)

    train(qa_data, epochs=args.epochs)
