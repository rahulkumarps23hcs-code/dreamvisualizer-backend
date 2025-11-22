from __future__ import annotations

from datetime import datetime
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler

from db.mongo import get_database
from .aggregator import (
    get_active_users,
    get_audio_minutes,
    get_exports_count,
    get_total_dreams,
    get_total_images,
    get_video_render_count,
)


_scheduler: Optional[BackgroundScheduler] = None


def _take_daily_snapshot() -> None:
    """Store a daily analytics snapshot document.

    This is optional and does not affect the real-time metrics that are
    computed directly from events. It can be useful for long-term trending
    without scanning the full events collection.
    """

    db = get_database()
    collection = db["analytics_daily"]

    now = datetime.utcnow()
    day_start = datetime(now.year, now.month, now.day)

    doc = {
        "date": day_start,
        "captured_at": now,
        "total_dreams": get_total_dreams(),
        "total_images": get_total_images(),
        "audio_minutes": get_audio_minutes(),
        "video_render_count": get_video_render_count(),
        "exports_count": get_exports_count(),
        "active_users_7d": get_active_users(7),
        "active_users_30d": get_active_users(30),
    }

    collection.insert_one(doc)


def start_scheduler() -> None:
    """Start a background scheduler to run the daily snapshot job.

    Call this from application startup if you want scheduled snapshots.
    """

    global _scheduler

    if _scheduler is not None:
        return

    scheduler = BackgroundScheduler()
    scheduler.add_job(_take_daily_snapshot, "cron", hour=1, minute=0)
    scheduler.start()

    _scheduler = scheduler


def shutdown_scheduler() -> None:
    global _scheduler

    if _scheduler is not None:
        _scheduler.shutdown()
        _scheduler = None
