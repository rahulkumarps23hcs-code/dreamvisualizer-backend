from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from analytics.events import log_event
from nlp.processor import process_story


router = APIRouter(prefix="/api/nlp", tags=["nlp"])


class NLPRequest(BaseModel):
    text: str


@router.post("/process")
async def nlp_process(payload: NLPRequest) -> dict:
    """Process a story text and return structured scenes, summary, and characters."""
    text = payload.text or ""
    if len(text) > 3000:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text is too long (max 3000 characters).")

    result = process_story(text)

    # Analytics: count how many stories (dreams) are created/processed
    text_length = len(text)
    log_event("dream_created", meta={"text_length": text_length})

    return result
