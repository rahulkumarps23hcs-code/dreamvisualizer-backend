from typing import List, Optional


try:  # Optional heavy deps: present locally, absent on Railway lite deploy
    import torch  # type: ignore
    from diffusers import StableDiffusionPipeline  # type: ignore
except Exception:  # pragma: no cover - defensive import guard
    torch = None  # type: ignore
    StableDiffusionPipeline = None  # type: ignore


_SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
_sd15_pipe: Optional["StableDiffusionPipeline"] = None


def _get_device() -> str:
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():  # type: ignore[union-attr]
        return "cuda"
    return "cpu"


def _get_sd15_pipe():
    global _sd15_pipe

    if StableDiffusionPipeline is None or torch is None:
        raise RuntimeError("SD15 image generation is not available in this deployment (torch/diffusers not installed).")

    if _sd15_pipe is None:
        device = _get_device()
        dtype = torch.float16 if device == "cuda" else torch.float32  # type: ignore[union-attr]

        pipe = StableDiffusionPipeline.from_pretrained(  # type: ignore[operator]
            _SD15_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
        )

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

    If torch/diffusers are not installed (e.g. on Railway lite deploy), this
    will raise a RuntimeError instead of crashing the whole app at import
    time.
    """

    if StableDiffusionPipeline is None or torch is None:
        raise RuntimeError("SD15 image generation is not available in this deployment (torch/diffusers not installed).")

    pipe = _get_sd15_pipe()

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
