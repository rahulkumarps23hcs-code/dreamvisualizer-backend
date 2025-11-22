from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from analytics.events import log_event
from auth.jwt_handler import get_current_user
from journal.saver import save_image
from .sd15 import generate_sd15
from .sdxl import generate_sdxl
from .consistency import ConsistencyState, adjust_prompt_for_consistency, init_consistency_state


router = APIRouter(prefix="/api/image", tags=["image"])


class SceneIn(BaseModel):
    id: Optional[int] = None
    text: str
    emotion: Optional[str] = None


class GenerateImagesRequest(BaseModel):
    model: Literal["sd15", "sdxl"] = "sd15"
    scenes: List[SceneIn]
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None


class GenerateImagesResponse(BaseModel):
    images: List[str]


def _build_prompt(scene: SceneIn, state: ConsistencyState) -> str:
    base_prompt = scene.text

    # Emotion hint to influence mood
    if scene.emotion:
        base_prompt = f"{base_prompt}. Mood: {scene.emotion}"

    # Maintain character consistency
    adjusted = adjust_prompt_for_consistency(base_prompt, state)

    # Global style
    return (
        f"{adjusted}, ultra detailed, cinematic lighting, 4k concept art, "
        f"artstation, trending, highly detailed, unreal engine render"
    )


@router.post("/generate", response_model=GenerateImagesResponse)
async def generate_images(
    payload: GenerateImagesRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> GenerateImagesResponse:
    if not payload.scenes:
        raise HTTPException(status_code=400, detail="At least one scene is required")

    if len(payload.scenes) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 scenes are allowed")

    output_dir = Path("generated_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    state = init_consistency_state()
    urls: List[str] = []

    for index, scene in enumerate(payload.scenes, start=1):
        prompt = _build_prompt(scene, state)

        if payload.model == "sd15":
            images = generate_sd15(
                prompt=prompt,
                negative_prompt=payload.negative_prompt,
                width=payload.width or 768,
                height=payload.height or 512,
                steps=payload.steps or 25,
            )
        else:
            images = generate_sdxl(
                prompt=prompt,
                negative_prompt=payload.negative_prompt,
                width=payload.width or 1024,
                height=payload.height or 1024,
                steps=payload.steps or 30,
            )

        if not images:
            continue

        image = images[0]
        timestamp = int(time())
        filename = f"{payload.model}_scene_{index}_{timestamp}.png"
        filepath = output_dir / filename
        image.save(filepath)

        public_url = f"/generated/{filename}"

        # Log analytics event for this generated image
        scene_id = scene.id if scene.id is not None else index
        log_event(
            "image_generated",
            meta={
                "model": payload.model,
                "scene_id": scene_id,
            },
        )

        # URLs will be served from FastAPI static mount, e.g. /generated/{filename}
        urls.append(public_url)

        save_image(
            current_user["id"],
            public_url,
            scene_index=scene_id,
        )

    return GenerateImagesResponse(images=urls)
