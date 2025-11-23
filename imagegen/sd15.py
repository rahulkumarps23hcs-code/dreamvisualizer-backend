from typing import List, Optional

from config.settings import settings


def generate_sd15(
    prompt: str,
    negative_prompt: str | None = None,
    width: int = 768,
    height: int = 512,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
) -> List["PIL.Image.Image"]:
    """Entry point for SD15 image generation.

    In lite mode (LITE_MODE=true), this function raises a RuntimeError so the
    backend can run without heavy ML dependencies. In full mode it is
    overwritten below with the real implementation backed by torch+diffusers.
    """

    raise RuntimeError("SD15 image generation is not available in lite mode.")


if not settings.lite_mode:
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

            pipe.to(device)
            pipe.enable_attention_slicing()

            _sd15_pipe = pipe

        return _sd15_pipe

    def _generate_sd15_impl(
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 768,
        height: int = 512,
        steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List["PIL.Image.Image"]:
        """Generate one or more images using Stable Diffusion 1.5."""

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

    # Override the lite stub with the full implementation in non-lite mode.
    def generate_sd15(
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 768,
        height: int = 512,
        steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List["PIL.Image.Image"]:
        return _generate_sd15_impl(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
