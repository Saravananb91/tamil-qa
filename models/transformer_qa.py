"""
models/transformer_qa.py
-------------------------
Custom Transformer-based Tamil QA model.
Predicts answer span start and end positions within a tokenised sequence.
"""

import torch
import torch.nn as nn


class TamilQAModel(nn.Module):
    """
    Transformer encoder that outputs start/end span logits for extractive QA.

    Architecture
    ------------
    Embedding → Positional Encoding → Transformer → Linear heads (start, end)

    Parameters
    ----------
    vocab_size         : tokenizer vocabulary size
    d_model            : embedding / transformer hidden dimension
    nhead              : number of self-attention heads
    num_encoder_layers : depth of the transformer encoder (and decoder)
    dim_feedforward    : inner FF dimension
    max_seq_length     : maximum supported sequence length (for positional encoding)
    """

    def __init__(
        self,
        vocab_size:         int,
        d_model:            int,
        nhead:              int,
        num_encoder_layers: int,
        dim_feedforward:    int,
        max_seq_length:     int,
    ):
        super().__init__()
        self.embedding          = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, d_model)
        )
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.start_logits = nn.Linear(d_model, 1)
        self.end_logits   = nn.Linear(d_model, 1)

    def forward(self, input_tokens: torch.Tensor):
        """
        Parameters
        ----------
        input_tokens : (batch, seq_len) long tensor

        Returns
        -------
        start_logits : (batch, seq_len)
        end_logits   : (batch, seq_len)
        """
        seq_len    = input_tokens.size(1)
        embeddings = (
            self.embedding(input_tokens)
            + self.positional_encoding[:, :seq_len, :]
        )
        out         = self.transformer(embeddings, embeddings)
        start_logits = self.start_logits(out).squeeze(-1)
        end_logits   = self.end_logits(out).squeeze(-1)
        return start_logits, end_logits
