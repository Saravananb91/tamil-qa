"""
chatbot/hybrid_qa.py
---------------------
Hybrid Tamil QA engine.

Strategy
--------
1. Preprocess the question with Stanza (Tamil NLP pipeline).
2. Attempt extractive answer using mBERT (bert-base-multilingual-cased).
3. If mBERT returns "No valid answer found.", fall back to:
   a. The custom Transformer model (if loaded), OR
   b. The Seq2Seq LSTM model.
"""

import torch
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing.text_cleaner  import preprocess_tamil_text, correct_grammar
from models.extractive_qa        import extract_answer
from models.seq2seq_qa           import generate_answer
from configs.config              import MAX_QUESTION_LEN, SEQ_LENGTH


_NO_ANSWER = "No valid answer found."


class HybridQASystem:
    """
    Orchestrates extractive + generative QA.

    Parameters
    ----------
    extractive_tokenizer : HuggingFace tokenizer for mBERT
    extractive_model     : HuggingFace AutoModelForQuestionAnswering
    contexts             : list of context strings to search for answers
    device               : torch device

    Optional (Seq2Seq fallback)
    ---------------------------
    seq2seq_tokenizer    : fitted Keras Tokenizer
    encoder_model        : Keras encoder inference model
    decoder_model        : Keras decoder inference model
    max_answer_len       : max decoding steps for Seq2Seq

    Optional (Transformer fallback)
    --------------------------------
    transformer_model    : TamilQAModel
    spm_tokenizer        : SentencePieceProcessor
    seq_length           : max token length
    """

    def __init__(
        self,
        extractive_tokenizer,
        extractive_model,
        contexts:            list,
        device:              torch.device = None,
        # Seq2Seq
        seq2seq_tokenizer=None,
        encoder_model=None,
        decoder_model=None,
        max_answer_len:      int  = 50,
        # Transformer
        transformer_model=None,
        spm_tokenizer=None,
        seq_length:          int  = SEQ_LENGTH,
    ):
        self.ex_tokenizer  = extractive_tokenizer
        self.ex_model      = extractive_model
        self.contexts      = contexts
        self.device        = device or torch.device("cpu")

        self.seq2seq_tok   = seq2seq_tokenizer
        self.encoder       = encoder_model
        self.decoder       = decoder_model
        self.max_ans_len   = max_answer_len

        self.transformer   = transformer_model
        self.spm_tok       = spm_tokenizer
        self.seq_length    = seq_length

    def _extractive(self, context: str, question: str) -> str:
        return extract_answer(
            context, question,
            self.ex_tokenizer, self.ex_model,
            device=str(self.device),
        )

    def _seq2seq_fallback(self, question: str) -> str:
        if self.seq2seq_tok is None or self.encoder is None:
            return ""
        q_seq    = self.seq2seq_tok.texts_to_sequences([question])
        q_padded = pad_sequences(q_seq, maxlen=MAX_QUESTION_LEN, padding="post", truncating="post")
        return generate_answer(q_padded, self.encoder, self.decoder, self.seq2seq_tok, self.max_ans_len)

    def _transformer_fallback(self, question: str) -> str:
        if self.transformer is None or self.spm_tok is None:
            return ""
        self.transformer.eval()
        with torch.no_grad():
            q_clean = preprocess_tamil_text(question)
            tokens  = self.spm_tok.encode(q_clean, out_type=int)[: self.seq_length]
            src     = torch.tensor([tokens], dtype=torch.long, device=self.device)
            s_logits, e_logits = self.transformer(src)
            s_idx   = torch.argmax(s_logits, dim=-1).item()
            e_idx   = torch.argmax(e_logits, dim=-1).item()
            ans_ids = src[0, s_idx : e_idx + 1].cpu().numpy()
            return self.spm_tok.decode(ans_ids.tolist())

    def answer(self, question: str) -> str:
        """
        Answer a Tamil question using the hybrid strategy.

        Parameters
        ----------
        question : Tamil question string

        Returns
        -------
        str  Best available answer.
        """
        corrected = correct_grammar(question)

        # Try each context with extractive model
        for ctx in self.contexts:
            ans = self._extractive(ctx, corrected)
            if ans != _NO_ANSWER and ans.strip():
                return ans

        # Fallback: Transformer
        ans = self._transformer_fallback(corrected)
        if ans.strip():
            return ans

        # Fallback: Seq2Seq
        ans = self._seq2seq_fallback(corrected)
        if ans.strip():
            return ans

        return "மன்னிக்கவும், எனக்கு புரியவில்லை. தயவுசெய்து மறு முயற்சி செய்யவும்."
