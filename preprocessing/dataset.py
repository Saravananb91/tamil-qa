"""
preprocessing/dataset.py
-------------------------
PyTorch Dataset and collate function for the Tamil QA Transformer model.
Handles tokenisation, padding, and start/end span label generation.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from preprocessing.text_cleaner import preprocess_tamil_text


class TamilQADataset(Dataset):
    """
    Dataset for Tamil extractive QA.

    Each item is a (context, question) pair with a known answer span.
    Tokens are encoded as [question] + [SEP] + [answer], and start/end
    position labels point to the answer span within that sequence.

    Parameters
    ----------
    qa_data    : list of dicts — each must have "question" and "answers" keys
    tokenizer  : SentencePieceProcessor
    seq_length : maximum token sequence length (will truncate if exceeded)
    """

    def __init__(self, qa_data: list, tokenizer, seq_length: int):
        self.data       = qa_data
        self.tokenizer  = tokenizer
        self.seq_length = seq_length
        self._sep       = tokenizer.piece_to_id("</s>")
        self._pad       = tokenizer.pad_id()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item     = self.data[idx]
        question = preprocess_tamil_text(item["question"])
        answer   = preprocess_tamil_text(item["answers"][0])

        q_tokens = self.tokenizer.encode(question, out_type=int)
        a_tokens = self.tokenizer.encode(answer,   out_type=int)

        tokens = (q_tokens + [self._sep] + a_tokens)[: self.seq_length]

        # Pad to seq_length
        pad_len = self.seq_length - len(tokens)
        tokens  = tokens + [self._pad] * pad_len

        src = torch.tensor(tokens, dtype=torch.long)

        # Span labels (clamped to seq_length - 1)
        start_idx   = min(len(q_tokens) + 1,              self.seq_length - 1)
        end_idx     = min(len(q_tokens) + len(a_tokens),  self.seq_length - 1)
        start_label = torch.tensor(start_idx, dtype=torch.long)
        end_label   = torch.tensor(end_idx,   dtype=torch.long)

        return src, start_label, end_label


def collate_fn(batch):
    """
    Dynamic-padding collate function for DataLoader.
    Pads variable-length sequences to the longest in the batch.
    """
    src_list, start_list, end_list = zip(*batch)
    padded_src  = pad_sequence(src_list, batch_first=True, padding_value=0)
    start_labels = torch.stack(start_list)
    end_labels   = torch.stack(end_list)
    return padded_src, start_labels, end_labels
