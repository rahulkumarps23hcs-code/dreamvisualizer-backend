from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from auth.jwt_handler import get_current_user
from .fetcher import filter_assets, get_user_assets


router = APIRouter(prefix="/api/journal", tags=["journal"])


AssetType = Literal["image", "audio", "video", "pdf", "comic", "zip"]


class AssetOut(BaseModel):
    id: str
    userId: str
    type: str
    url: str
    scene_index: Optional[int] = None
    createdAt: Optional[datetime] = None


@router.get("/all", response_model=List[AssetOut])
async def list_all_assets(current_user=Depends(get_current_user)) -> List[AssetOut]:
    user_id = current_user["id"]
    assets = get_user_assets(user_id)
    return [AssetOut(**asset) for asset in assets]


@router.get("/type", response_model=List[AssetOut])
async def list_assets_by_type(
    type: AssetType = Query(..., description="Asset type: image | audio | video | pdf | comic | zip"),
    current_user=Depends(get_current_user),
) -> List[AssetOut]:
    user_id = current_user["id"]
    assets = filter_assets(user_id, asset_type=type)
    return [AssetOut(**asset) for asset in assets]
