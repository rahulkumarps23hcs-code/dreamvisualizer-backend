from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from analytics.events import log_event
from auth.jwt_handler import get_current_user
from journal.saver import save_pdf, save_zip
from .storybook import generate_storybook
from .comicbook import generate_comic
from .zip_export import generate_zip


_IMAGE_DIR = "generated_images"
_AUDIO_DIR = "generated_audio"
_VIDEO_DIR = "generated_videos"
_EXPORT_DIR = "exports"


router = APIRouter(prefix="/api/export", tags=["export"])


class Scene(BaseModel):
    id: Optional[int] = None
    text: str
    emotion: Optional[str] = None
    summary: Optional[str] = None


class StorybookRequest(BaseModel):
    scenes: List[Scene]
    image_urls: List[str]
    overall_summary: Optional[str] = None


class ComicRequest(BaseModel):
    scenes: List[Scene]
    image_urls: List[str]


class BundleRequest(BaseModel):
    image_urls: List[str] = []
    audio_urls: List[str] = []
    video_url: Optional[str] = None
    metadata: dict[str, Any] | None = None


class ExportResponse(BaseModel):
    url: str


def _resolve_local_path(url_or_path: str, base_dir: str, expected_prefix: str) -> Path:
    path_part = urlparse(url_or_path).path if "://" in url_or_path else url_or_path

    if path_part.startswith(expected_prefix):
        filename = Path(path_part).name
    else:
        filename = Path(path_part).name

    return Path(base_dir) / filename


@router.post("/storybook", response_model=ExportResponse)
async def export_storybook(
    payload: StorybookRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ExportResponse:
    if not payload.scenes:
        raise HTTPException(status_code=400, detail="At least one scene is required.")

    if len(payload.scenes) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 scenes are allowed.")

    image_paths: List[str] = []
    for url in payload.image_urls:
        image_paths.append(str(_resolve_local_path(url, _IMAGE_DIR, "/generated/")))

    pdf_path = generate_storybook(
        scenes=payload.scenes,
        image_paths=image_paths,
        overall_summary=payload.overall_summary,
        output_dir=_EXPORT_DIR,
    )

    filename = Path(pdf_path).name
    # Analytics: track storybook exports
    log_event("export_storybook", meta={"filename": filename})

    public_url = f"/exports/{filename}"
    save_pdf(current_user["id"], public_url, kind="pdf")

    return ExportResponse(url=public_url)


@router.post("/comic", response_model=ExportResponse)
async def export_comic(
    payload: ComicRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ExportResponse:
    if not payload.scenes:
        raise HTTPException(status_code=400, detail="At least one scene is required.")

    image_paths: List[str] = []
    for url in payload.image_urls:
        image_paths.append(str(_resolve_local_path(url, _IMAGE_DIR, "/generated/")))

    pdf_path = generate_comic(
        scenes=payload.scenes,
        image_paths=image_paths,
        output_dir=_EXPORT_DIR,
    )

    filename = Path(pdf_path).name
    # Analytics: track comic exports
    log_event("export_comic", meta={"filename": filename})

    public_url = f"/exports/{filename}"
    save_pdf(current_user["id"], public_url, kind="comic")

    return ExportResponse(url=public_url)


@router.post("/bundle", response_model=ExportResponse)
async def export_bundle(
    payload: BundleRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ExportResponse:
    image_paths: List[str] = []
    for url in payload.image_urls:
        image_paths.append(str(_resolve_local_path(url, _IMAGE_DIR, "/generated/")))

    audio_paths: List[str] = []
    for url in payload.audio_urls:
        audio_paths.append(str(_resolve_local_path(url, _AUDIO_DIR, "/audio-files/")))

    video_path: Optional[str] = None
    if payload.video_url:
        video_path = str(_resolve_local_path(payload.video_url, _VIDEO_DIR, "/videos/"))

    all_assets: dict[str, Any] = {
        "images": image_paths,
        "audio": audio_paths,
        "video": video_path,
        "metadata": payload.metadata or {},
    }

    zip_path = generate_zip(all_assets, output_dir=_EXPORT_DIR)

    filename = Path(zip_path).name
    # Analytics: track bundle exports
    log_event("export_bundle", meta={"filename": filename})

    public_url = f"/exports/{filename}"
    save_zip(current_user["id"], public_url)

    return ExportResponse(url=public_url)
