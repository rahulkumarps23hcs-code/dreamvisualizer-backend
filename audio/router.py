from pathlib import Path
from typing import Any, Dict, List, Optional
import wave

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from analytics.events import log_event
from auth.jwt_handler import get_current_user
from journal.saver import save_audio
from .tts_engine import generate_tts


router = APIRouter(prefix="/api/audio", tags=["audio"])


class SceneIn(BaseModel):
    id: Optional[int] = None
    text: str


class GenerateAudioRequest(BaseModel):
    scenes: List[SceneIn]
    language: str = "english"
    voice: str = "default"


class GenerateAudioResponse(BaseModel):
    audio_files: List[str]


@router.post("/generate", response_model=GenerateAudioResponse)
async def generate_audio(
    payload: GenerateAudioRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> GenerateAudioResponse:
    if not payload.scenes:
        raise HTTPException(status_code=400, detail="At least one scene is required")

    if len(payload.scenes) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 scenes are allowed")

    audio_files: List[str] = []

    for index, scene in enumerate(payload.scenes, start=1):
        scene_id = scene.id if scene.id is not None else index
        try:
            audio_path = generate_tts(scene.text, payload.language, payload.voice)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - unexpected
            raise HTTPException(status_code=500, detail="Audio generation failed") from exc

        filename = Path(audio_path).name
        public_url = f"/audio-files/{filename}"
        audio_files.append(public_url)

        # Derive audio duration in seconds for analytics
        duration_seconds = 0.0
        try:
            with wave.open(audio_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate() or 1
                duration_seconds = frames / float(rate)
        except Exception:
            duration_seconds = 0.0

        log_event(
            "audio_generated",
            meta={
                "language": payload.language,
                "voice": payload.voice,
                "duration_seconds": duration_seconds,
                "scene_id": scene_id,
            },
        )

        save_audio(
            current_user["id"],
            public_url,
            scene_index=scene_id,
        )

    return GenerateAudioResponse(audio_files=audio_files)
