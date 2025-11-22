from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional

from db.mongo import get_database


_COLLECTION_NAME = "analytics_events"


def log_event(
    event_type: str,
    user_id: Optional[str] = None,
    dream_id: Optional[str] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> None:
    """Insert a single analytics event into the analytics_events collection.

    The event schema is intentionally simple and flexible so we can attach
    arbitrary metadata in the future without schema migrations.
    """

    db = get_database()
    collection = db[_COLLECTION_NAME]

    now = datetime.utcnow()

    doc: dict[str, Any] = {
        "event_type": event_type,
        "created_at": now,
        "meta": dict(meta) if meta is not None else {},
    }

    if user_id is not None:
        doc["user_id"] = str(user_id)
    if dream_id is not None:
        doc["dream_id"] = str(dream_id)

    collection.insert_one(doc)
