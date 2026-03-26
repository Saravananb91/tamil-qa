"""
chatbot/chat.py
---------------
Interactive Tamil QA chatbot (command-line interface).

Usage
-----
    python -m chatbot.chat --model outputs/tamil_qa_transformer.pth \
                           --data  data/qa_data.json
"""

import argparse
import json
import torch

from configs.config         import MBERT_MODEL_NAME, TOKENIZER_MODEL, SEQ_LENGTH
from models.extractive_qa   import load_extractive_model
from preprocessing.tokenizer_trainer import load_tokenizer
from models.transformer_qa  import TamilQAModel
from configs.config         import D_MODEL, N_HEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD
from chatbot.hybrid_qa      import HybridQASystem


_WELCOME = (
    "\n" + "─" * 55 + "\n"
    "  தமிழ் கேள்வி-பதிலி மென்பொருள் (Tamil QA Chatbot)\n"
    "─" * 55 + "\n"
    "  Type 'exit' to quit.\n"
    "─" * 55
)


def run_chat(
    qa_data:     list,
    model_path:  str,
    spm_path:    str = TOKENIZER_MODEL,
    device:      torch.device = None,
) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load mBERT
    print("🔄 Loading mBERT extractive model…")
    ex_tok, ex_model = load_extractive_model(MBERT_MODEL_NAME)
    ex_model.to(device)

    # Load SentencePiece + Transformer
    print("🔄 Loading SentencePiece tokenizer…")
    spm_tok = load_tokenizer(spm_path)

    transformer = TamilQAModel(
        vocab_size=spm_tok.vocab_size(),
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_seq_length=SEQ_LENGTH,
    ).to(device)
    transformer.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Transformer model loaded from: {model_path}")

    contexts = [item.get("context", "") for item in qa_data if item.get("context")]

    qa = HybridQASystem(
        extractive_tokenizer=ex_tok,
        extractive_model=ex_model,
        contexts=contexts,
        device=device,
        transformer_model=transformer,
        spm_tokenizer=spm_tok,
        seq_length=SEQ_LENGTH,
    )

    print(_WELCOME)
    while True:
        try:
            user_input = input("\nநீங்கள்: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user_input.lower() in ("exit", "quit"):
            print("\nவணக்கம்! சந்தோஷமாக இருங்கள்! 🙏")
            break
        if not user_input:
            continue
        answer = qa.answer(user_input)
        print(f"பதில்: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tamil Hybrid QA Chatbot")
    parser.add_argument("--model", required=True, help="Path to trained transformer .pth")
    parser.add_argument("--data",  required=True, help="Path to QA dataset .json")
    parser.add_argument("--spm",   default=TOKENIZER_MODEL, help="SentencePiece model path")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        qa_data = json.load(f)

    run_chat(qa_data, args.model, args.spm)
