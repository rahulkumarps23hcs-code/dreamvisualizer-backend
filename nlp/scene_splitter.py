from typing import Dict, List

import re


def _split_sentences(text: str) -> List[str]:
    """Very simple sentence splitter using punctuation.

    We avoid heavy NLTK models here to keep the setup light and deterministic.
    """

    cleaned = text.strip()
    if not cleaned:
        return []

    # Split on ., !, ? followed by whitespace
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def split_into_scenes(text: str, sentences_per_scene: int = 3) -> List[Dict[str, str]]:
    """Split story text into small 'scenes'.

    Uses a naive sentence splitter and groups a few sentences per scene.
    Returns a list of scene dicts: {"id": int, "text": str}.
    """

    sentences = _split_sentences(text)

    scenes: List[Dict[str, str]] = []
    current: List[str] = []
    scene_id = 1

    for sentence in sentences:
        current.append(sentence)
        if len(current) >= sentences_per_scene:
            scenes.append({"id": scene_id, "text": " ".join(current)})
            scene_id += 1
            current = []

    if current:
        scenes.append({"id": scene_id, "text": " ".join(current)})

    if not scenes and text.strip():
        scenes = [{"id": 1, "text": text.strip()}]

    return scenes
