from typing import Optional

from transformers import pipeline


# Small emotion classification model
_MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
_emotion_pipeline: Optional[object] = None


def _get_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        _emotion_pipeline = pipeline("text-classification", model=_MODEL_NAME)
    return _emotion_pipeline


_LABEL_MAP = {
    "joy": "joy",
    "happiness": "joy",
    "sadness": "sad",
    "fear": "fear",
    "anxiety": "fear",
    "surprise": "adventure",
    "excitement": "adventure",
    "love": "calm",
    "neutral": "calm",
}

_DEFAULT_EMOTION = "mystery"


def detect_emotion(text: str) -> str:
    """Detect a coarse emotion label for the given text.

    Returns one of: joy | fear | mystery | adventure | sad | calm.
    Uses a small transformer emotion classifier under the hood.
    """

    cleaned = text.strip()
    if not cleaned:
        return "calm"

    classifier = _get_emotion_pipeline()
    result = classifier(cleaned, truncation=True)[0]
    label = str(result["label"]).lower()

    mapped = _LABEL_MAP.get(label)
    if mapped is None:
        # Fall back to a generic "mystery" label for anything unknown
        if "sad" in label:
            return "sad"
        if "fear" in label or "anxiety" in label:
            return "fear"
        if "joy" in label or "happy" in label:
            return "joy"
        return _DEFAULT_EMOTION

    return mapped
