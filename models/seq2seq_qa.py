"""
models/seq2seq_qa.py
--------------------
Sequence-to-Sequence LSTM model for generative Tamil QA.
Used as a fallback when the extractive mBERT model returns no answer.

Architecture: Encoder LSTM → Decoder LSTM with teacher forcing during training,
and greedy decoding during inference.
"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

from configs.config import EMBEDDING_DIM, HIDDEN_UNITS


def build_seq2seq(
    vocab_size:       int,
    max_question_len: int,
    max_answer_len:   int,
    embedding_dim:    int = EMBEDDING_DIM,
    hidden_units:     int = HIDDEN_UNITS,
) -> tuple[Model, Model, Model]:
    """
    Build training model, encoder inference model, and decoder inference model.

    Parameters
    ----------
    vocab_size       : vocabulary size
    max_question_len : padded question sequence length
    max_answer_len   : padded answer sequence length
    embedding_dim    : embedding dimension
    hidden_units     : LSTM hidden state size

    Returns
    -------
    (train_model, encoder_model, decoder_model)
    """
    # ── Shared layers (referenced by both train and inference models) ─────────
    encoder_inputs    = Input(shape=(max_question_len,), name="encoder_input")
    encoder_embedding = Embedding(vocab_size, embedding_dim, name="encoder_embed")
    encoder_lstm      = LSTM(hidden_units, return_state=True, name="encoder_lstm")

    decoder_inputs    = Input(shape=(max_answer_len,), name="decoder_input")
    decoder_embedding = Embedding(vocab_size, embedding_dim, name="decoder_embed")
    decoder_lstm      = LSTM(hidden_units, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_dense     = Dense(vocab_size, activation="softmax", name="decoder_dense")

    # ── Training model ────────────────────────────────────────────────────────
    enc_embedded              = encoder_embedding(encoder_inputs)
    enc_out, state_h, state_c = encoder_lstm(enc_embedded)
    encoder_states            = [state_h, state_c]

    dec_embedded              = decoder_embedding(decoder_inputs)
    dec_out, _, _             = decoder_lstm(dec_embedded, initial_state=encoder_states)
    decoder_outputs           = decoder_dense(dec_out)

    train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_train")
    train_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Encoder inference model ───────────────────────────────────────────────
    encoder_model = Model(encoder_inputs, encoder_states, name="encoder_inference")

    # ── Decoder inference model ───────────────────────────────────────────────
    dec_state_h   = Input(shape=(hidden_units,), name="dec_state_h")
    dec_state_c   = Input(shape=(hidden_units,), name="dec_state_c")
    dec_inf_input = Input(shape=(1,),            name="dec_token_input")

    dec_inf_embed            = decoder_embedding(dec_inf_input)
    dec_inf_out, h_inf, c_inf = decoder_lstm(
        dec_inf_embed, initial_state=[dec_state_h, dec_state_c]
    )
    dec_inf_logits = decoder_dense(dec_inf_out)

    decoder_model = Model(
        [dec_inf_input, dec_state_h, dec_state_c],
        [dec_inf_logits, h_inf, c_inf],
        name="decoder_inference",
    )

    return train_model, encoder_model, decoder_model


def generate_answer(
    input_seq:     np.ndarray,
    encoder_model: Model,
    decoder_model: Model,
    tokenizer,
    max_answer_len: int,
) -> str:
    """
    Greedy decode an answer from a padded question sequence.

    Parameters
    ----------
    input_seq      : (1, max_question_len) numpy array
    encoder_model  : Keras encoder inference model
    decoder_model  : Keras decoder inference model
    tokenizer      : fitted Keras Tokenizer
    max_answer_len : maximum decoding steps

    Returns
    -------
    str  Decoded answer text (without <start> / <end> tokens).
    """
    states_value = encoder_model.predict(input_seq, verbose=0)

    target_seq    = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get("<start>", 1)

    decoded_words: list[str] = []
    stop_condition = False

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )
        sampled_idx  = int(np.argmax(output_tokens[0, -1, :]))
        sampled_word = tokenizer.index_word.get(sampled_idx, "")

        if sampled_word == "<end>" or len(decoded_words) >= max_answer_len:
            stop_condition = True
        else:
            decoded_words.append(sampled_word)

        target_seq    = np.array([[sampled_idx]])
        states_value  = [h, c]

    return " ".join(decoded_words).strip()
