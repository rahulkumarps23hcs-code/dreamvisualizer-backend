from typing import List, Optional


try:  # Optional heavy deps: present locally, absent on Railway lite deploy
    import torch  # type: ignore
    from diffusers import StableDiffusionXLPipeline  # type: ignore
except Exception:  # pragma: no cover - defensive import guard
    torch = None  # type: ignore
    StableDiffusionXLPipeline = None  # type: ignore


_SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_sdxl_pipe: Optional["StableDiffusionXLPipeline"] = None


def _get_device() -> str:
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():  # type: ignore[union-attr]
        return "cuda"
    return "cpu"


def _get_sdxl_pipe():
    global _sdxl_pipe

    if StableDiffusionXLPipeline is None or torch is None:
        raise RuntimeError("SDXL image generation is not available in this deployment (torch/diffusers not installed).")

    if _sdxl_pipe is None:
        device = _get_device()
        dtype = torch.float16 if device == "cuda" else torch.float32  # type: ignore[union-attr]

        pipe = StableDiffusionXLPipeline.from_pretrained(  # type: ignore[operator]
            _SDXL_MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
        )

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

    If torch/diffusers are not installed (e.g. on Railway lite deploy), this
    will raise a RuntimeError instead of crashing the whole app at import
    time.
    """

    if StableDiffusionXLPipeline is None or torch is None:
        raise RuntimeError("SDXL image generation is not available in this deployment (torch/diffusers not installed).")

    pipe = _get_sdxl_pipe()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)  # type: ignore[union-attr]

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

    if getattr(pipe, "device", None) is not None and getattr(pipe.device, "type", None) == "cuda":
        try:
            torch.cuda.empty_cache()  # type: ignore[union-attr]
        except Exception:
            pass

    return images
