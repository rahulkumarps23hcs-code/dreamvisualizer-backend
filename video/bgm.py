from __future__ import annotations

from pathlib import Path
from time import time
from typing import Optional, Sequence

from pydub import AudioSegment

# Point pydub to the exact ffmpeg.exe location on Windows
AudioSegment.converter = r"C:\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

_DEFAULT_OUTPUT_DIR = "generated_videos"


def mix_scenes_with_bgm(
    audio_paths: Sequence[str],
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    bgm_path: Optional[str] = None,
    bgm_gain_db: float = -18.0,
) -> str:
    """Merge scene narration audio files and optionally overlay soft BGM.

    Returns the path to a WAV file containing the final mixed audio.
    """

    if not audio_paths:
        raise ValueError("At least one narration audio path is required.")

    combined = AudioSegment.empty()
    for path in audio_paths:
        segment = AudioSegment.from_file(path)
        combined += segment

    if bgm_path is not None and Path(bgm_path).is_file():
        bgm = AudioSegment.from_file(bgm_path)

        if len(bgm) < len(combined):
            loops = int(len(combined) / len(bgm)) + 1
            bgm = (bgm * loops)[: len(combined)]
        else:
            bgm = bgm[: len(combined)]

        bgm = bgm + bgm_gain_db
        final_audio = bgm.overlay(combined)
    else:
        final_audio = combined

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = out_dir / f"final_audio_{int(time())}.wav"
    final_audio.export(filename, format="wav")
    return str(filename)
