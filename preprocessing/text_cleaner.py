"""
preprocessing/text_cleaner.py
------------------------------
Tamil text preprocessing utilities:
  - Unicode normalisation
  - Tamil script filtering (retain only Tamil Unicode block)
  - Whitespace normalisation
  - Grammar correction via LanguageTool (optional)
"""

import re
import unicodedata


# Tamil Unicode range: U+0B80–U+0BFF (includes ஂ–ஔ, எ–௺)
_TAMIL_PATTERN = re.compile(r"[^ஂ-ஔஎ-௺\s]")
_SPACE_PATTERN = re.compile(r"\s+")


def preprocess_tamil_text(text: str) -> str:
    """
    Clean raw Tamil text for model input.

    Steps
    -----
    1. Unicode NFKC normalisation.
    2. Remove all characters outside the Tamil Unicode block.
    3. Collapse multiple whitespace characters into a single space.

    Parameters
    ----------
    text : str  Raw Tamil text.

    Returns
    -------
    str  Cleaned text containing only Tamil script and spaces.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _TAMIL_PATTERN.sub("", text)
    text = _SPACE_PATTERN.sub(" ", text).strip()
    return text


def correct_grammar(text: str) -> str:
    """
    Optionally correct grammar using LanguageTool for Tamil (ta-IN).
    Falls back to the raw text if LanguageTool is not installed.

    Parameters
    ----------
    text : str  Input Tamil text.

    Returns
    -------
    str  Grammar-corrected text (or original if tool unavailable).
    """
    try:
        import language_tool_python
        tool = language_tool_python.LanguageTool("ta-IN")
        matches = tool.check(text)
        return language_tool_python.utils.correct(text, matches)
    except ImportError:
        return text
    except Exception:
        return text


def build_corpus_file(qa_data: list, output_path: str) -> str:
    """
    Write a plain-text corpus file from QA data for SentencePiece training.

    Parameters
    ----------
    qa_data     : list of dicts with "question" and "answers" keys
    output_path : path to write the corpus .txt file

    Returns
    -------
    str  Path to the written corpus file.
    """
    lines = []
    for item in qa_data:
        q = preprocess_tamil_text(item.get("question", ""))
        a = preprocess_tamil_text(item["answers"][0]) if item.get("answers") else ""
        if q or a:
            lines.append(f"{q} {a}".strip())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Corpus written to: {output_path}  ({len(lines)} lines)")
    return output_path
