from __future__ import annotations

import os
from pathlib import Path
from time import time
from typing import Optional

from gtts import gTTS


_TTS_CACHE: dict[tuple[str, str], str] = {}


def _get_gtts_language_code(language: str) -> str:
    lang_map = {
        "english": "en",
        "hindi": "hi",
        "marathi": "mr",
        "tamil": "ta",
    }

    key = language.lower()
    if key not in lang_map:
        raise ValueError(f"Unsupported language: {language}")

    return lang_map[key]


def generate_tts(
    text: str,
    language: str,
    voice_style: str = "default",
    output_dir: str = "generated_audio",
) -> str:
    """Generate TTS audio for a given text and language.

    Returns the absolute path to a WAV file.
    """

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Text is empty")

    lang_key = language.lower()
    cache_key = (lang_key, cleaned)

    cached_path = _TTS_CACHE.get(cache_key)
    if cached_path and os.path.isfile(cached_path):
        return cached_path

    os.makedirs(output_dir, exist_ok=True)

    lang_code = _get_gtts_language_code(language)

    # Note: voice_style is accepted for future extension; many Coqui models
    # are single-voice, so we simply ignore it for now.

    timestamp = int(time())
    safe_lang = lang_key.replace(" ", "_")
    filename = f"scene_{safe_lang}_{timestamp}.wav"
    path = Path(output_dir) / filename

    tts = gTTS(text=cleaned, lang=lang_code)
    tts.save(str(path))

    resolved = str(path.resolve())
    _TTS_CACHE[cache_key] = resolved

    return resolved
