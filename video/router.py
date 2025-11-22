from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from analytics.events import log_event
from auth.jwt_handler import get_current_user
from journal.saver import save_video
from .lip_sync import lip_sync
from .composer import compose_video

_IMAGE_DIR = "generated_images"
_AUDIO_DIR = "generated_audio"
_VIDEO_DIR = "generated_videos"

router = APIRouter(prefix="/api/video", tags=["video"])


class GenerateVideoRequest(BaseModel):
    image_urls: List[str]
    audio_urls: List[str]


class GenerateVideoResponse(BaseModel):
    video_url: str


def _resolve_local_path(url_or_path: str, base_dir: str, expected_prefix: str) -> Path:
    """Turn a public URL or relative path into a local filesystem path."""

    path_part = urlparse(url_or_path).path if "://" in url_or_path else url_or_path

    if path_part.startswith(expected_prefix):
        filename = Path(path_part).name
    else:
        filename = Path(path_part).name

    return Path(base_dir) / filename


@router.post("/generate", response_model=GenerateVideoResponse)
async def generate_video(
    payload: GenerateVideoRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> GenerateVideoResponse:
    if not payload.image_urls or not payload.audio_urls:
        raise HTTPException(status_code=400, detail="image_urls and audio_urls are required")

    if len(payload.image_urls) != len(payload.audio_urls):
        raise HTTPException(status_code=400, detail="image_urls and audio_urls must have the same length")

    clip_paths: List[str] = []
    audio_paths: List[str] = []

    for image_url, audio_url in zip(payload.image_urls, payload.audio_urls):
        image_path = _resolve_local_path(image_url, _IMAGE_DIR, "/generated/")
        audio_path = _resolve_local_path(audio_url, _AUDIO_DIR, "/audio-files/")

        if not image_path.is_file():
            raise HTTPException(status_code=400, detail=f"Image file not found: {image_path.name}")
        if not audio_path.is_file():
            raise HTTPException(status_code=400, detail=f"Audio file not found: {audio_path.name}")

        clip_path = lip_sync(str(image_path), str(audio_path), output_dir=_VIDEO_DIR)
        clip_paths.append(clip_path)
        audio_paths.append(str(audio_path))

    # Optional BGM file location (if you add one later, place it here)
    bgm_path: str | None = None

    final_video_path = compose_video(clip_paths, audio_paths, output_dir=_VIDEO_DIR, bgm_path=bgm_path)
    filename = Path(final_video_path).name
    video_url = f"/videos/{filename}"

    # Analytics: track video render operations
    log_event("video_rendered", meta={"clip_count": len(clip_paths)})

    save_video(current_user["id"], video_url)

    return GenerateVideoResponse(video_url=video_url)
