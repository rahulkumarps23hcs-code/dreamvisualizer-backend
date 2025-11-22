from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from config.settings import settings
from db.mongo import connect_to_mongo, close_mongo_connection, get_database
from auth.routes import router as auth_router
from routes.nlp import router as nlp_router
from imagegen.router import router as image_router
from audio.router import router as audio_router
from video.router import router as video_router
from exporter.router import router as export_router
from analytics.router import router as analytics_router
from journal.router import router as journal_router
from tasks.router import router as tasks_router


app = FastAPI(title=settings.app_name)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://dream-visualizer-frontend-4jcndhs0b.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(nlp_router)
app.include_router(image_router)
app.include_router(audio_router)
app.include_router(video_router)
app.include_router(export_router)
app.include_router(analytics_router)
app.include_router(journal_router)
app.include_router(tasks_router)

_GENERATED_DIR = "generated_images"
_AUDIO_DIR = "generated_audio"
_VIDEO_DIR = "generated_videos"
_EXPORT_DIR = "exports"
os.makedirs(_GENERATED_DIR, exist_ok=True)
os.makedirs(_AUDIO_DIR, exist_ok=True)
os.makedirs(_VIDEO_DIR, exist_ok=True)
os.makedirs(_EXPORT_DIR, exist_ok=True)
app.mount("/generated", StaticFiles(directory=_GENERATED_DIR), name="generated")
app.mount("/audio-files", StaticFiles(directory=_AUDIO_DIR), name="audio-files")
app.mount("/videos", StaticFiles(directory=_VIDEO_DIR), name="videos")
app.mount("/exports", StaticFiles(directory=_EXPORT_DIR), name="exports")


@app.on_event("startup")
def on_startup() -> None:
    connect_to_mongo()


@app.on_event("shutdown")
def on_shutdown() -> None:
    close_mongo_connection()


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "app": settings.app_name}


@app.get("/ping")
async def ping() -> dict:
    db = get_database()
    db.command("ping")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
