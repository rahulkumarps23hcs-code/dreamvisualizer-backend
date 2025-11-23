from dataclasses import dataclass
from typing import Optional


@dataclass
class ConsistencyState:
    """Lightweight state used to keep prompts roughly consistent.

    In lite mode we avoid heavy CLIP models and instead just remember the
    first scene's description as the "base" character description.
    """

    base_description: Optional[str] = None


def init_consistency_state() -> ConsistencyState:
    return ConsistencyState()


def adjust_prompt_for_consistency(scene_text: str, state: ConsistencyState) -> str:
    """Return a prompt adjusted to keep the main character visually consistent.

    Strategy without CLIP:
    - First non-empty scene becomes the base_description.
    - Later scenes get a small textual hint to reuse the same main character.
    """

    cleaned = scene_text.strip()
    if not cleaned:
        return scene_text

    if state.base_description is None:
        # Initialize base character description from the first scene
        state.base_description = cleaned
        return cleaned

    # For subsequent scenes, gently nudge the prompt towards the same character
    return f"{cleaned}, same main character as previous scenes ({state.base_description})"
