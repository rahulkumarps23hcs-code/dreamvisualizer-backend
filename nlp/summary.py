from typing import Optional

from transformers import pipeline


_SUMMARIZER: Optional[object] = None
_MODEL_NAME = "t5-small"


def _get_summarizer():
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = pipeline("summarization", model=_MODEL_NAME)
    return _SUMMARIZER


def generate_summary(text: str, max_length: int = 128, min_length: int = 24) -> str:
    """Generate a short abstractive summary for the given text.

    Uses a lightweight T5-based summarization model.
    """

    cleaned = text.strip()
    if not cleaned:
        return ""

    # For very short texts, just return the original
    if len(cleaned.split()) < 25:
        return cleaned

    summarizer = _get_summarizer()
    input_text = f"summarize: {cleaned}"

    result = summarizer(input_text, max_length=max_length, min_length=min_length, truncation=True)
    if not result:
        return cleaned

    return result[0].get("summary_text", cleaned)
