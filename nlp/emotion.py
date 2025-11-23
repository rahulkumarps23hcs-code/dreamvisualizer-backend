from typing import Dict


_DEFAULT_EMOTION = "mystery"


def _keyword_scores(text: str) -> Dict[str, int]:
    cleaned = text.lower()
    scores: Dict[str, int] = {"joy": 0, "sad": 0, "fear": 0, "adventure": 0, "calm": 0}

    joy_words = ["happy", "joy", "smile", "excited", "fun", "laugh"]
    sad_words = ["sad", "cry", "lonely", "tears", "hurt"]
    fear_words = ["scared", "fear", "afraid", "anxious", "anxiety", "danger"]
    adv_words = ["adventure", "explore", "mystery", "quest", "journey"]
    calm_words = ["peace", "calm", "relax", "quiet", "sleep", "rest"]

    for w in joy_words:
        if w in cleaned:
            scores["joy"] += 1
    for w in sad_words:
        if w in cleaned:
            scores["sad"] += 1
    for w in fear_words:
        if w in cleaned:
            scores["fear"] += 1
    for w in adv_words:
        if w in cleaned:
            scores["adventure"] += 1
    for w in calm_words:
        if w in cleaned:
            scores["calm"] += 1

    return scores


def detect_emotion(text: str) -> str:
    """Detect a coarse emotion label for the given text.

    Returns one of: joy | fear | mystery | adventure | sad | calm.
    Uses a lightweight keyword classifier instead of external ML models.
    """

    cleaned = text.strip()
    if not cleaned:
        return "calm"

    scores = _keyword_scores(cleaned)
    label, value = max(scores.items(), key=lambda kv: kv[1])

    if value == 0:
        # No strong keyword match; treat as generic "mystery" story
        return _DEFAULT_EMOTION

    return label
