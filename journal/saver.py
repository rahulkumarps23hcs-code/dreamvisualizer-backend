from __future__ import annotations

from datetime import datetime
from typing import Optional

from db.mongo import get_database


_COLLECTION_NAME = "user_assets"


def _get_collection():
    db = get_database()
    return db[_COLLECTION_NAME]


def _normalize_user_id(user_id: str) -> str:
    return str(user_id)


def _insert_asset(user_id: str, asset_type: str, url: str, scene_index: Optional[int] = None) -> str:
    """Insert a single asset document into the user_assets collection.

    Document fields strictly follow the requested schema:
    userId, type, url, scene_index (optional), createdAt.
    """

    col = _get_collection()
    now = datetime.utcnow()

    doc: dict = {
        "userId": _normalize_user_id(user_id),
        "type": asset_type,
        "url": url,
        "createdAt": now,
    }

    if scene_index is not None:
        doc["scene_index"] = scene_index

    result = col.insert_one(doc)
    return str(result.inserted_id)


def save_image(user_id: str, url: str, scene_index: Optional[int] = None) -> str:
    return _insert_asset(user_id=user_id, asset_type="image", url=url, scene_index=scene_index)


def save_audio(user_id: str, url: str, scene_index: Optional[int] = None) -> str:
    return _insert_asset(user_id=user_id, asset_type="audio", url=url, scene_index=scene_index)


def save_video(user_id: str, url: str) -> str:
    return _insert_asset(user_id=user_id, asset_type="video", url=url)


def save_pdf(user_id: str, url: str, kind: str = "pdf") -> str:
    """Save a PDF-like asset.

    kind is mapped onto the "type" field and should typically be
    either "pdf" (storybook) or "comic".
    """

    asset_type = kind if kind in {"pdf", "comic"} else "pdf"
    return _insert_asset(user_id=user_id, asset_type=asset_type, url=url)


def save_zip(user_id: str, url: str) -> str:
    return _insert_asset(user_id=user_id, asset_type="zip", url=url)
