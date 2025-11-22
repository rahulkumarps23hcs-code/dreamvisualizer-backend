from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline


_SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
_sd15_pipe: Optional[StableDiffusionPipeline] = None


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_sd15_pipe() -> StableDiffusionPipeline:
    global _sd15_pipe

    if _sd15_pipe is None:
        device = _get_device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            _SD15_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
        )

        if device == "cuda":
            pipe.to(device)
            pipe.enable_attention_slicing()
        else:
            # CPU-friendly configuration
            pipe.to(device)
            pipe.enable_attention_slicing()

        _sd15_pipe = pipe

    return _sd15_pipe


def generate_sd15(
    prompt: str,
    negative_prompt: str | None = None,
    width: int = 768,
    height: int = 512,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
) -> List["PIL.Image.Image"]:
    """Generate one or more images using Stable Diffusion 1.5.

    Returns a list of PIL images (usually a single image).
    """

    pipe = _get_sd15_pipe()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=1,
    )

    images = result.images
    del result

    if pipe.device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return images
