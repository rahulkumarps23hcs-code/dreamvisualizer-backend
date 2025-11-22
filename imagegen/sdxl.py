from typing import List, Optional

import torch
from diffusers import StableDiffusionXLPipeline


_SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_sdxl_pipe: Optional[StableDiffusionXLPipeline] = None


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_sdxl_pipe() -> StableDiffusionXLPipeline:
    global _sdxl_pipe

    if _sdxl_pipe is None:
        device = _get_device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionXLPipeline.from_pretrained(
            _SDXL_MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        if device == "cuda":
            pipe.to(device)
            pipe.enable_attention_slicing()
        else:
            pipe.to(device)
            pipe.enable_attention_slicing()

        _sdxl_pipe = pipe

    return _sdxl_pipe


def generate_sdxl(
    prompt: str,
    negative_prompt: str | None = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 30,
    guidance_scale: float = 7.0,
    seed: Optional[int] = None,
) -> List["PIL.Image.Image"]:
    """Generate images using SDXL (higher quality, slower).

    Returns a list of PIL images (usually a single image).
    """

    pipe = _get_sdxl_pipe()

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
