from typing import Optional


def generate_summary(text: str, max_length: int = 128, min_length: int = 24) -> str:
    """Generate a short summary for the given text without external ML models.

    Strategy:
    - If text is empty, return empty string.
    - If text is already short, return it unchanged.
    - Otherwise, return the first N words (bounded by max_length) as a simple
      extractive summary.
    """

    cleaned = text.strip()
    if not cleaned:
        return ""

    words = cleaned.split()

    # For very short texts, just return the original
    if len(words) <= max(min_length, 25):
        return cleaned

    # Truncate to at most max_length words
    limit = max(32, min(max_length, 128))
    summary_words = words[:limit]
    return " ".join(summary_words)
