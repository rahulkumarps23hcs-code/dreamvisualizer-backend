from __future__ import annotations

from datetime import datetime
from time import sleep
from typing import Any, Dict, List, Literal, Optional

from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel

from analytics.events import log_event
from auth.jwt_handler import get_current_user
from audio.tts_engine import generate_tts
from db.mongo import get_database
from imagegen.consistency import ConsistencyState, adjust_prompt_for_consistency, init_consistency_state
from imagegen.sd15 import generate_sd15
from imagegen.sdxl import generate_sdxl
from journal.saver import save_audio, save_image, save_video
from video.composer import compose_video
from video.lip_sync import lip_sync


router = APIRouter(prefix="/api/tasks", tags=["tasks"])

_TASKS_COLLECTION = "tasks"


def _tasks_collection():
    db = get_database()
    return db[_TASKS_COLLECTION]


def _update_task(task_id: str, fields: Dict[str, Any]) -> None:
    col = _tasks_collection()
    col.update_one(
        {"_id": ObjectId(task_id)},
        {
            "$set": {
                **fields,
                "updatedAt": datetime.utcnow(),
            }
        },
    )


class SceneIn(BaseModel):
    id: Optional[int] = None
    text: str
    emotion: Optional[str] = None


class ImageTaskRequest(BaseModel):
    model: Literal["sd15", "sdxl"] = "sd15"
    scenes: List[SceneIn]
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None


class AudioSceneIn(BaseModel):
    id: Optional[int] = None
    text: str


class AudioTaskRequest(BaseModel):
    scenes: List[AudioSceneIn]
    language: str = "english"
    voice: str = "default"


class VideoTaskRequest(BaseModel):
    image_urls: List[str]
    audio_urls: List[str]


class TaskCreateResponse(BaseModel):
    task_id: str


class TaskStatusResponse(BaseModel):
    id: str
    type: Literal["image", "audio", "video"]
    status: Literal["queued", "running", "finishing", "complete", "failed"]
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None


def _build_prompt(scene: SceneIn, state: ConsistencyState) -> str:
    base_prompt = scene.text
    if scene.emotion:
        base_prompt = f"{base_prompt}. Mood: {scene.emotion}"
    adjusted = adjust_prompt_for_consistency(base_prompt, state)
    return (
        f"{adjusted}, ultra detailed, cinematic lighting, 4k concept art, "
        f"artstation, trending, highly detailed, unreal engine render"
    )


def _run_image_task(task_id: str, payload: Dict[str, Any], user_id: str) -> None:
    from pathlib import Path
    from time import time as now_time

    col = _tasks_collection()
    _update_task(task_id, {"status": "running", "progress": 0.0})

    try:
        req = ImageTaskRequest(**payload)
        if not req.scenes:
            raise ValueError("At least one scene is required")
        if len(req.scenes) > 10:
            raise ValueError("Maximum 10 scenes are allowed")

        output_dir = Path("generated_images")
        output_dir.mkdir(parents=True, exist_ok=True)

        state = init_consistency_state()
        urls: List[str] = []

        total = len(req.scenes)
        for index, scene in enumerate(req.scenes, start=1):
            prompt = _build_prompt(scene, state)

            if req.model == "sd15":
                images = generate_sd15(
                    prompt=prompt,
                    negative_prompt=req.negative_prompt,
                    width=req.width or 768,
                    height=req.height or 512,
                    steps=req.steps or 25,
                )
            else:
                images = generate_sdxl(
                    prompt=prompt,
                    negative_prompt=req.negative_prompt,
                    width=req.width or 1024,
                    height=req.height or 1024,
                    steps=req.steps or 30,
                )

            if not images:
                continue

            image = images[0]
            timestamp = int(now_time())
            filename = f"{req.model}_scene_{index}_{timestamp}.png"
            filepath = output_dir / filename
            image.save(filepath)

            public_url = f"/generated/{filename}"
            urls.append(public_url)

            scene_id = scene.id if scene.id is not None else index
            log_event(
                "image_generated",
                meta={
                    "model": req.model,
                    "scene_id": scene_id,
                },
            )

            save_image(user_id, public_url, scene_index=scene_id)

            progress = (index / total) * 100.0
            _update_task(task_id, {"progress": progress})

        _update_task(
            task_id,
            {
                "status": "complete",
                "progress": 100.0,
                "result": {"images": urls},
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        _update_task(task_id, {"status": "failed", "error": str(exc)})


def _run_audio_task(task_id: str, payload: Dict[str, Any], user_id: str) -> None:
    from pathlib import Path

    col = _tasks_collection()
    _update_task(task_id, {"status": "running", "progress": 0.0})

    try:
        req = AudioTaskRequest(**payload)
        if not req.scenes:
            raise ValueError("At least one scene is required")
        if len(req.scenes) > 10:
            raise ValueError("Maximum 10 scenes are allowed")

        audio_files: List[str] = []
        total = len(req.scenes)

        for index, scene in enumerate(req.scenes, start=1):
            scene_id = scene.id if scene.id is not None else index

            audio_path = generate_tts(scene.text, req.language, req.voice)
            filename = Path(audio_path).name
            public_url = f"/audio-files/{filename}"
            audio_files.append(public_url)

            log_event(
                "audio_generated",
                meta={
                    "language": req.language,
                    "voice": req.voice,
                },
            )

            save_audio(user_id, public_url, scene_index=scene_id)

            progress = (index / total) * 100.0
            _update_task(task_id, {"progress": progress})

        _update_task(
            task_id,
            {
                "status": "complete",
                "progress": 100.0,
                "result": {"audio_files": audio_files},
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        _update_task(task_id, {"status": "failed", "error": str(exc)})


def _run_video_task(task_id: str, payload: Dict[str, Any], user_id: str) -> None:
    from pathlib import Path

    col = _tasks_collection()
    _update_task(task_id, {"status": "running", "progress": 0.0})

    try:
        req = VideoTaskRequest(**payload)
        if not req.image_urls or not req.audio_urls:
            raise ValueError("image_urls and audio_urls are required")
        if len(req.image_urls) != len(req.audio_urls):
            raise ValueError("image_urls and audio_urls must have the same length")

        _IMAGE_DIR = "generated_images"
        _AUDIO_DIR = "generated_audio"
        _VIDEO_DIR = "generated_videos"

        clip_paths: List[str] = []
        audio_paths: List[str] = []

        total = len(req.image_urls)

        for index, (image_url, audio_url) in enumerate(zip(req.image_urls, req.audio_urls), start=1):
            image_path = Path(_IMAGE_DIR) / Path(image_url).name
            audio_path = Path(_AUDIO_DIR) / Path(audio_url).name

            if not image_path.is_file():
                raise ValueError(f"Image file not found: {image_path.name}")
            if not audio_path.is_file():
                raise ValueError(f"Audio file not found: {audio_path.name}")

            clip_path = lip_sync(str(image_path), str(audio_path), output_dir=_VIDEO_DIR)
            clip_paths.append(clip_path)
            audio_paths.append(str(audio_path))

            progress = (index / total) * 70.0  # first 70% while per-scene clips are built
            _update_task(task_id, {"progress": progress})

        bgm_path: Optional[str] = None

        _update_task(task_id, {"status": "finishing", "progress": 85.0})

        final_video_path = compose_video(clip_paths, audio_paths, output_dir=_VIDEO_DIR, bgm_path=bgm_path)
        filename = Path(final_video_path).name
        video_url = f"/videos/{filename}"

        log_event("video_rendered", meta={"clip_count": len(clip_paths)})
        save_video(user_id, video_url)

        _update_task(
            task_id,
            {
                "status": "complete",
                "progress": 100.0,
                "result": {"video_url": video_url},
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        _update_task(task_id, {"status": "failed", "error": str(exc)})


@router.post("/image", response_model=TaskCreateResponse)
async def create_image_task(
    payload: ImageTaskRequest,
    background: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> TaskCreateResponse:
    if not payload.scenes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one scene is required")
    if len(payload.scenes) > 10:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum 10 scenes are allowed")

    col = _tasks_collection()
    now = datetime.utcnow()
    doc = {
        "type": "image",
        "status": "queued",
        "progress": 0.0,
        "result": None,
        "error": None,
        "userId": current_user["id"],
        "createdAt": now,
        "updatedAt": now,
    }
    result = col.insert_one(doc)
    task_id = str(result.inserted_id)

    background.add_task(_run_image_task, task_id, payload.dict(), current_user["id"])

    return TaskCreateResponse(task_id=task_id)


@router.post("/audio", response_model=TaskCreateResponse)
async def create_audio_task(
    payload: AudioTaskRequest,
    background: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> TaskCreateResponse:
    if not payload.scenes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one scene is required")
    if len(payload.scenes) > 10:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum 10 scenes are allowed")

    col = _tasks_collection()
    now = datetime.utcnow()
    doc = {
        "type": "audio",
        "status": "queued",
        "progress": 0.0,
        "result": None,
        "error": None,
        "userId": current_user["id"],
        "createdAt": now,
        "updatedAt": now,
    }
    result = col.insert_one(doc)
    task_id = str(result.inserted_id)

    background.add_task(_run_audio_task, task_id, payload.dict(), current_user["id"])

    return TaskCreateResponse(task_id=task_id)


@router.post("/video", response_model=TaskCreateResponse)
async def create_video_task(
    payload: VideoTaskRequest,
    background: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> TaskCreateResponse:
    if not payload.image_urls or not payload.audio_urls:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="image_urls and audio_urls are required")
    if len(payload.image_urls) != len(payload.audio_urls):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image_urls and audio_urls must have the same length",
        )

    col = _tasks_collection()
    now = datetime.utcnow()
    doc = {
        "type": "video",
        "status": "queued",
        "progress": 0.0,
        "result": None,
        "error": None,
        "userId": current_user["id"],
        "createdAt": now,
        "updatedAt": now,
    }
    result = col.insert_one(doc)
    task_id = str(result.inserted_id)

    background.add_task(_run_video_task, task_id, payload.dict(), current_user["id"])

    return TaskCreateResponse(task_id=task_id)


@router.get("/status", response_model=TaskStatusResponse)
async def get_task_status(id: str = Query(..., description="Task id")) -> TaskStatusResponse:
    col = _tasks_collection()
    try:
        obj_id = ObjectId(id)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid task id") from exc

    doc = col.find_one({"_id": obj_id})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    return TaskStatusResponse(
        id=str(doc.get("_id")),
        type=doc.get("type"),
        status=doc.get("status"),
        progress=float(doc.get("progress", 0.0)),
        result=doc.get("result"),
        error=doc.get("error"),
        createdAt=doc.get("createdAt"),
        updatedAt=doc.get("updatedAt"),
    )
