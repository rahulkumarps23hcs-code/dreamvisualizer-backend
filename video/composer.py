from __future__ import annotations

from pathlib import Path
from time import time
from typing import List, Optional, Sequence
import gc

from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips

from .bgm import mix_scenes_with_bgm

_DEFAULT_OUTPUT_DIR = "generated_videos"


def compose_video(
    clip_paths: Sequence[str],
    audio_paths: Sequence[str],
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    bgm_path: Optional[str] = None,
) -> str:
    """Combine per-scene clips and audio into a single MP4 video."""

    if not clip_paths:
        raise ValueError("At least one video clip is required.")
    if len(clip_paths) != len(audio_paths):
        raise ValueError("clip_paths and audio_paths must have the same length.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips: List[VideoFileClip] = []
    composite_clip: Optional[VideoFileClip] = None
    audio_clip: Optional[AudioFileClip] = None
    final_video_path: Optional[Path] = None

    try:
        for path in clip_paths:
            clip = VideoFileClip(str(path))
            clip = clip.fadein(0.3).fadeout(0.3)
            clips.append(clip)

        composite_clip = concatenate_videoclips(clips, method="compose")

        mixed_audio_path = mix_scenes_with_bgm(audio_paths, output_dir=str(out_dir), bgm_path=bgm_path)
        audio_clip = AudioFileClip(mixed_audio_path)
        composite_clip = composite_clip.set_audio(audio_clip)

        filename = out_dir / f"final_video_{int(time())}.mp4"
        composite_clip.write_videofile(
            str(filename),
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )
        final_video_path = filename
    finally:
        for clip in clips:
            clip.close()
        if audio_clip is not None:
            audio_clip.close()
        if composite_clip is not None:
            composite_clip.close()
        gc.collect()

    if final_video_path is None:
        raise RuntimeError("Video composition failed.")

    return str(final_video_path)
