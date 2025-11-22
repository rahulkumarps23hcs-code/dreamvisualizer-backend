from __future__ import annotations

import os
from pathlib import Path
from time import time

from moviepy.editor import AudioFileClip, ImageClip

_DEFAULT_OUTPUT_DIR = "generated_videos"


def lip_sync(image_path: str, audio_path: str, output_dir: str = _DEFAULT_OUTPUT_DIR) -> str:
    """Create a short video clip from a single image and narration audio.

    This is implemented as a static image + audio clip so that the pipeline
    works even without a local Wav2Lip installation. The function signature
    and output format are compatible with a future Wav2Lip-based
    implementation.
    """

    image_file = Path(image_path)
    audio_file = Path(audio_path)

    if not image_file.is_file():
        raise FileNotFoundError(f"Image not found: {image_file}")
    if not audio_file.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_file}")

    os.makedirs(output_dir, exist_ok=True)

    audio_clip = AudioFileClip(str(audio_file))
    video_clip = None

    try:
        duration = float(audio_clip.duration or 0.0)
        if duration <= 0:
            raise ValueError("Audio duration must have a positive duration.")

        video_clip = ImageClip(str(image_file)).set_duration(duration).set_audio(audio_clip)
        video_clip = video_clip.set_fps(25)

        timestamp = int(time())
        filename = f"scene_{image_file.stem}_{timestamp}.mp4"
        output_path = Path(output_dir) / filename

        video_clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )
    finally:
        audio_clip.close()
        if video_clip is not None:
            video_clip.close()

    return str(output_path)
