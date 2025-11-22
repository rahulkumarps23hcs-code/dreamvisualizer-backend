from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence

from db.mongo import get_database


_COLLECTION_NAME = "analytics_events"


def _events_collection():
    db = get_database()
    return db[_COLLECTION_NAME]


def _single_count(match: Dict[str, Any]) -> int:
    pipeline = [{"$match": match}, {"$count": "value"}]
    results = list(_events_collection().aggregate(pipeline))
    return int(results[0]["value"]) if results else 0


def get_total_dreams() -> int:
    return _single_count({"event_type": "dream_created"})


def get_total_images() -> int:
    return _single_count({"event_type": "image_generated"})


def get_audio_minutes() -> float:
    """Return total audio duration in minutes based on audio_generated events.

    Expects events of type "audio_generated" to store a
    meta.duration_seconds numeric field (fallback to 0 if missing).
    """

    pipeline = [
        {"$match": {"event_type": "audio_generated"}},
        {
            "$group": {
                "_id": None,
                "total_seconds": {
                    "$sum": {"$ifNull": ["$meta.duration_seconds", 0]}
                },
            }
        },
    ]

    results = list(_events_collection().aggregate(pipeline))
    if not results:
        return 0.0

    total_seconds = float(results[0].get("total_seconds", 0.0))
    return total_seconds / 60.0


def get_video_render_count() -> int:
    return _single_count({"event_type": "video_rendered"})


def get_exports_count() -> int:
    return _single_count(
        {"event_type": {"$in": ["export_storybook", "export_comic", "export_bundle"]}}
    )


def get_active_users(days: int) -> int:
    """Count distinct users with any event in the last N days."""

    since = datetime.utcnow() - timedelta(days=days)

    pipeline = [
        {"$match": {"created_at": {"$gte": since}, "user_id": {"$exists": True}}},
        {"$group": {"_id": "$user_id"}},
        {"$count": "value"},
    ]

    results = list(_events_collection().aggregate(pipeline))
    return int(results[0]["value"]) if results else 0


def get_timeseries(metric: str) -> List[Dict[str, Any]]:
    """Return a daily timeseries for the requested metric.

    Supported metrics:
      - "dreams": count of dream_created events per day
      - "images": count of image_generated events per day
      - "video_renders": count of video_rendered events per day
      - "exports": count of export_* events per day
      - "audio_minutes": total audio minutes per day (from meta.duration_seconds)
    """

    metric_defs: Dict[str, Dict[str, Any]] = {
        "dreams": {
            "event_types": ["dream_created"],
            "value_expr": 1,
        },
        "images": {
            "event_types": ["image_generated"],
            "value_expr": 1,
        },
        "video_renders": {
            "event_types": ["video_rendered"],
            "value_expr": 1,
        },
        "exports": {
            "event_types": ["export_storybook", "export_comic", "export_bundle"],
            "value_expr": 1,
        },
        "audio_minutes": {
            "event_types": ["audio_generated"],
            "value_expr": {"$ifNull": ["$meta.duration_seconds", 0]},
        },
    }

    if metric not in metric_defs:
        raise ValueError(f"Unsupported metric: {metric}")

    definition = metric_defs[metric]

    match_stage = {
        "$match": {"event_type": {"$in": definition["event_types"]}},
    }

    group_stage = {
        "$group": {
            "_id": {
                "$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}
            },
            "value": {"$sum": definition["value_expr"]},
        }
    }

    pipeline = [match_stage, group_stage, {"$sort": {"_id": 1}}]

    rows = list(_events_collection().aggregate(pipeline))

    timeseries: List[Dict[str, Any]] = []
    for row in rows:
        value = float(row.get("value", 0.0))
        if metric == "audio_minutes":
            value = value / 60.0
        timeseries.append({"date": row["_id"], "value": value})

    return timeseries


def get_top_models(limit: int = 5) -> List[Dict[str, Any]]:
    """Return top models based on usage counts.

    Expects events to optionally store a meta.model field for whichever
    operations have a model choice (e.g. image generation models).
    """

    pipeline = [
        {"$match": {"meta.model": {"$exists": True}}},
        {"$group": {"_id": "$meta.model", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": int(limit)},
    ]

    rows = list(_events_collection().aggregate(pipeline))
    return [{"model": row["_id"], "count": int(row["count"])} for row in rows]
