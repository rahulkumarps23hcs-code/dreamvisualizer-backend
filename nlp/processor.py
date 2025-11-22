from typing import Any, Dict, List

import re

from .scene_splitter import split_into_scenes
from .emotion import detect_emotion
from .summary import generate_summary


def _extract_characters(text: str, max_characters: int = 10) -> List[str]:
    """Very naive character extraction based on capitalized tokens.

    This is intentionally simple: we look for unique capitalized words
    that are not common stopwords and treat them as character names.
    """

    tokens = re.findall(r"[A-Za-z][A-Za-z']+", text)

    stopwords = {"the", "and", "but", "with", "from", "into", "this", "that", "there", "here"}

    characters: List[str] = []
    for tok in tokens:
        if not tok[0].isupper():
            continue
        lower = tok.lower()
        if lower in stopwords or lower == "i":
            continue
        if tok not in characters:
            characters.append(tok)
        if len(characters) >= max_characters:
            break

    return characters


def process_story(text: str) -> Dict[str, Any]:
    """Main NLP pipeline for DreamVisualizer AI.

    - Split the story into scenes
    - For each scene: detect emotion + generate a mini-summary
    - Generate an overall summary
    - Extract a simple list of character-like nouns
    """

    scenes_raw = split_into_scenes(text)

    enriched_scenes: List[Dict[str, Any]] = []
    for scene in scenes_raw:
        scene_text = scene["text"]
        emotion = detect_emotion(scene_text)
        summary = generate_summary(scene_text, max_length=64, min_length=16)
        enriched_scenes.append(
            {
                "id": scene["id"],
                "text": scene_text,
                "emotion": emotion,
                "summary": summary,
            }
        )

    overall_summary = generate_summary(text)
    characters = _extract_characters(text)
    emotions = [scene["emotion"] for scene in enriched_scenes]

    return {
        "scenes": enriched_scenes,
        "emotions": emotions,
        "overall_summary": overall_summary,
        "characters": characters,
    }
