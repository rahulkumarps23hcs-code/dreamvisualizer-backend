from __future__ import annotations

import json
from io import StringIO
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from db.mongo import get_database
from .aggregator import (
    get_active_users,
    get_audio_minutes,
    get_exports_count,
    get_timeseries,
    get_top_models,
    get_total_dreams,
    get_total_images,
    get_video_render_count,
)


router = APIRouter(prefix="/api/analytics", tags=["analytics"])


class OverviewResponse(BaseModel):
    total_dreams: int
    total_images: int
    audio_minutes: float
    video_render_count: int
    exports_count: int
    active_users_7d: int
    active_users_30d: int


class TimeseriesPoint(BaseModel):
    date: str
    value: float


class TimeseriesResponse(BaseModel):
    metric: str
    points: List[TimeseriesPoint]


class TopModel(BaseModel):
    model: str
    count: int


class TopModelsResponse(BaseModel):
    models: List[TopModel]


@router.get("/overview", response_model=OverviewResponse)
async def analytics_overview() -> OverviewResponse:
    return OverviewResponse(
        total_dreams=get_total_dreams(),
        total_images=get_total_images(),
        audio_minutes=get_audio_minutes(),
        video_render_count=get_video_render_count(),
        exports_count=get_exports_count(),
        active_users_7d=get_active_users(7),
        active_users_30d=get_active_users(30),
    )


@router.get("/timeseries", response_model=TimeseriesResponse)
async def analytics_timeseries(metric: str = Query(..., description="Metric name, e.g. dreams, images, audio_minutes")) -> TimeseriesResponse:
    try:
        rows = get_timeseries(metric)
    except ValueError as exc:  # unsupported metric
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    points = [TimeseriesPoint(**row) for row in rows]
    return TimeseriesResponse(metric=metric, points=points)


@router.get("/top-models", response_model=TopModelsResponse)
async def analytics_top_models(limit: int = Query(5, ge=1, le=50)) -> TopModelsResponse:
    rows = get_top_models(limit=limit)
    return TopModelsResponse(models=[TopModel(**row) for row in rows])


@router.get("/export/csv")
async def analytics_export_csv() -> StreamingResponse:
    """Export raw analytics_events as a CSV file.

    This is intended for internal analysis. It limits the number of rows to
    avoid exporting an unbounded dataset.
    """

    db = get_database()
    collection = db["analytics_events"]

    cursor = collection.find().limit(10000)

    rows: List[Dict[str, Any]] = []
    for doc in cursor:
        rows.append(
            {
                "id": str(doc.get("_id")),
                "event_type": doc.get("event_type"),
                "user_id": doc.get("user_id"),
                "dream_id": doc.get("dream_id"),
                "created_at": doc.get("created_at"),
                "meta": json.dumps(doc.get("meta") or {}, ensure_ascii=False),
            }
        )

    if not rows:
        df = pd.DataFrame(columns=["id", "event_type", "user_id", "dream_id", "created_at", "meta"])
    else:
        df = pd.DataFrame(rows)

    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=analytics_events.csv"},
    )
