from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings

app = FastAPI(title="Manufacturing Maintenance Agent")
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.mount(
    "/chat-images",
    StaticFiles(directory=settings.chat_image_dir),
    name="chat-images",
)
app.mount(
    "/uploaded-documents",
    StaticFiles(directory=settings.upload_dir),
    name="uploaded-documents",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
