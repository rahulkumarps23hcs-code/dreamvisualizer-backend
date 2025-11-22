from dataclasses import dataclass
from typing import Optional

import torch
from transformers import CLIPModel, CLIPTokenizer


_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
_clip_model: Optional[CLIPModel] = None
_clip_tokenizer: Optional[CLIPTokenizer] = None


def _get_clip() -> tuple[CLIPModel, CLIPTokenizer]:
  global _clip_model, _clip_tokenizer

  if _clip_model is None or _clip_tokenizer is None:
      _clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_ID)
      _clip_tokenizer = CLIPTokenizer.from_pretrained(_CLIP_MODEL_ID)
      _clip_model.to("cpu")

  return _clip_model, _clip_tokenizer


def _encode_text(text: str) -> torch.Tensor:
    model, tokenizer = _get_clip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    features = torch.nn.functional.normalize(features, dim=-1)
    return features.squeeze(0)


@dataclass
class ConsistencyState:
    base_description: Optional[str] = None
    base_embedding: Optional[torch.Tensor] = None


def init_consistency_state() -> ConsistencyState:
    return ConsistencyState()


def adjust_prompt_for_consistency(scene_text: str, state: ConsistencyState) -> str:
    """Return a prompt adjusted to keep the main character visually consistent.

    This is a lightweight approximation using CLIP text embeddings. For the
    first scene we capture a base embedding; for later scenes we nudge the
    prompt to mention keeping the same main character.
    """

    cleaned = scene_text.strip()
    if not cleaned:
        return scene_text

    if state.base_embedding is None:
        # Initialize base character description from the first scene
        state.base_description = cleaned
        state.base_embedding = _encode_text(cleaned)
        return cleaned

    current_emb = _encode_text(cleaned)
    similarity = torch.nn.functional.cosine_similarity(current_emb, state.base_embedding, dim=0).item()

    if similarity >= 0.7:
        # Already similar enough; just lightly reinforce consistency
        return f"{cleaned}, same main character as previous scenes"

    # Scene drifts too far; explicitly reference the base character description
    return f"{cleaned}. Keep the main character looking like: {state.base_description}"
