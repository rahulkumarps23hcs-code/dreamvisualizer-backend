from __future__ import annotations

from typing import List, Optional

from db.mongo import get_database


_COLLECTION_NAME = "user_assets"


def _get_collection():
    db = get_database()
    return db[_COLLECTION_NAME]


def get_user_assets(user_id: str) -> List[dict]:
    col = _get_collection()
    cursor = col.find({"userId": str(user_id)}).sort("createdAt", -1)
    return [
        {
            "id": str(doc.get("_id")),
            "userId": doc.get("userId"),
            "type": doc.get("type"),
            "url": doc.get("url"),
            "scene_index": doc.get("scene_index"),
            "kind": doc.get("kind"),
            "createdAt": doc.get("createdAt"),
        }
        for doc in cursor
    ]


def filter_assets(user_id: str, asset_type: str) -> List[dict]:
    col = _get_collection()
    cursor = col.find({"userId": str(user_id), "type": asset_type}).sort("createdAt", -1)
    return [
        {
            "id": str(doc.get("_id")),
            "userId": doc.get("userId"),
            "type": doc.get("type"),
            "url": doc.get("url"),
            "scene_index": doc.get("scene_index"),
            "kind": doc.get("kind"),
            "createdAt": doc.get("createdAt"),
        }
        for doc in cursor
    ]
