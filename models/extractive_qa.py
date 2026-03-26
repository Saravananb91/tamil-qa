"""
models/extractive_qa.py
------------------------
Extractive QA using multilingual BERT (bert-base-multilingual-cased).
Fine-tuned on Tamil QA data; falls back to Seq2Seq generation when
no confident span is found.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW

from configs.config import MBERT_MODEL_NAME, MAX_BERT_LENGTH, BATCH_SIZE, EPOCHS


# ── Inference ─────────────────────────────────────────────────────────────────

def load_extractive_model(model_path: str = MBERT_MODEL_NAME):
    """Load tokenizer and QA model from a HuggingFace checkpoint or local path."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForQuestionAnswering.from_pretrained(model_path)
    return tokenizer, model


def extract_answer(
    context:        str,
    question:       str,
    tokenizer,
    model,
    max_length:     int = MAX_BERT_LENGTH,
    device:         str = "cpu",
) -> str:
    """
    Run extractive QA on a (context, question) pair.

    Parameters
    ----------
    context  : passage containing the answer
    question : Tamil question string
    tokenizer: HuggingFace tokenizer
    model    : HuggingFace AutoModelForQuestionAnswering
    max_length: max token length for the input
    device   : "cpu" or "cuda"

    Returns
    -------
    str  Extracted answer span, or "No valid answer found." if span is empty.
    """
    model.eval()
    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx  = torch.argmax(outputs.start_logits).item()
    end_idx    = torch.argmax(outputs.end_logits).item() + 1
    token_ids  = inputs["input_ids"][0][start_idx:end_idx]

    if len(token_ids) == 0:
        return "No valid answer found."

    return tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(token_ids)
    )


# ── Fine-tuning dataset ───────────────────────────────────────────────────────

class MBertQADataset(Dataset):
    """
    Dataset for fine-tuning mBERT on Tamil QA pairs.
    Each item returns input_ids, attention_mask, and start/end positions.
    """

    def __init__(self, qa_data: list, tokenizer, max_length: int = MAX_BERT_LENGTH):
        self.data      = qa_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item     = self.data[idx]
        question = item["question"].strip()
        answer   = item["answers"][0].strip() if item.get("answers") else ""

        encoding = self.tokenizer(
            question, answer,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Approximate start/end using char_to_token mapping
        start_pos = encoding.char_to_token(0, 0) or 0
        end_pos   = encoding.char_to_token(0, max(0, len(answer) - 1)) or 0

        return (
            input_ids,
            attention_mask,
            torch.tensor(start_pos, dtype=torch.long),
            torch.tensor(end_pos,   dtype=torch.long),
        )


def finetune(
    qa_data:    list,
    model_name: str = MBERT_MODEL_NAME,
    epochs:     int = EPOCHS,
    device:     str = "cpu",
    save_dir:   str = "outputs/tamil_qa_mbert",
):
    """
    Fine-tune mBERT on the Tamil QA dataset.

    Parameters
    ----------
    qa_data    : list of {"question": ..., "answers": [...]} dicts
    model_name : HuggingFace model name or local path
    epochs     : number of training epochs
    device     : "cpu" or "cuda"
    save_dir   : directory to save the fine-tuned model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    dataset    = MBertQADataset(qa_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer  = AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, attn_mask, start_pos, end_pos in dataloader:
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            start_pos, end_pos   = start_pos.to(device), end_pos.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                start_positions=start_pos,
                end_positions=end_pos,
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(dataloader):.4f}")

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Fine-tuned model saved to: {save_dir}")
    return model, tokenizer
