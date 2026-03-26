"""
data/utils/convert_dataset.py
------------------------------
Utility scripts to convert raw text / JSONL files into the standard
QA JSON format expected by the training pipeline.

Expected output format:
    [
        {"question": "...", "answers": ["..."], "context": "..."},
        ...
    ]

Usage
-----
    # Convert a .txt file (one "question<TAB>answer" per line)
    python -m data.utils.convert_dataset --mode txt --input dev_doc.json --output data/qa_data.json

    # Convert a JSONL file (one JSON object per line)
    python -m data.utils.convert_dataset --mode jsonl --input test.txt --output data/qa_data.json
"""

import argparse
import json


def txt_to_json(input_path: str, output_path: str) -> list:
    """
    Convert a text file where each line is "question<space>answer"
    into the standard QA JSON list format.
    """
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                question, answer = parts
                records.append({"question": question, "answers": [answer]})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted {len(records)} records → {output_path}")
    return records


def jsonl_to_json(input_path: str, output_path: str) -> list:
    """
    Convert a JSONL file (one JSON object per line) to a JSON array.
    """
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted {len(records)} records → {output_path}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw data to QA JSON format")
    parser.add_argument("--mode",   choices=["txt", "jsonl"], required=True)
    parser.add_argument("--input",  required=True, help="Input file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    if args.mode == "txt":
        txt_to_json(args.input, args.output)
    else:
        jsonl_to_json(args.input, args.output)
