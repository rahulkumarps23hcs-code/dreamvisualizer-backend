from __future__ import annotations

import os
from pathlib import Path
from time import time
from typing import Optional

from .language_map import LANGUAGE_MODEL_MAP

try:
    from TTS.api import TTS as CoquiTTS  # type: ignore
except Exception:  # pragma: no cover - import error handled at runtime
    CoquiTTS = None  # type: ignore


_TTS_INSTANCES: dict[str, "CoquiTTS"] = {}
_TTS_CACHE: dict[tuple[str, str], str] = {}


def _get_tts_backend(language: str) -> "CoquiTTS":
    if CoquiTTS is None:
        raise RuntimeError("TTS library is not installed or failed to import.")

    key = language.lower()
    if key not in LANGUAGE_MODEL_MAP:
        raise ValueError(f"Unsupported language: {language}")

    model_name = LANGUAGE_MODEL_MAP[key]

    if model_name not in _TTS_INSTANCES:
        # Lazy-load model once and reuse
        tts = CoquiTTS(model_name=model_name)
        _TTS_INSTANCES[model_name] = tts

    return _TTS_INSTANCES[model_name]


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

    tts = _get_tts_backend(language)

    # Note: voice_style is accepted for future extension; many Coqui models
    # are single-voice, so we simply ignore it for now.

    timestamp = int(time())
    safe_lang = lang_key.replace(" ", "_")
    filename = f"scene_{safe_lang}_{timestamp}.wav"
    path = Path(output_dir) / filename

    # Coqui TTS high-level API
    tts.tts_to_file(text=cleaned, file_path=str(path))

    resolved = str(path.resolve())
    _TTS_CACHE[cache_key] = resolved

    return resolved
