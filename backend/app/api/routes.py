import json
import logging
from collections import Counter
from pathlib import Path
from uuid import uuid4
from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import get_settings, update_runtime_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkPreviewItem,
    ChunkPreviewResponse,
    ConversationInfo,
    DocumentInfo,
    DocumentUploadResponse,
    HistoryMessage,
    HistoryToolCall,
    ImageAttachment,
)
from app.services.agent import answer_message
from app.services.chunker import chunk_text
from app.services.conversation_store import (
    conversation_exists,
    create_conversation,
    delete_conversation,
    list_conversations,
    list_history_messages,
    list_tool_calls,
)
from app.services.document_loader import load_document_sections, load_document_text
from app.services.vision import describe_image
from app.services.vector_store import get_vector_store

router = APIRouter()
logger = logging.getLogger("uvicorn.error")
ALLOWED_CHAT_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DATASET_PATH = PROJECT_ROOT / "evaluation" / "golden" / "golden_dataset.json"
RAGAS_REPORT_PATH = PROJECT_ROOT / "evaluation" / "reports" / "ragas_report_v1.json"


class ChatLlmSettingsRequest(BaseModel):
    provider: str
    model: str


def _llm_providers(settings) -> list[dict]:
    active_model = _active_chat_model(settings)
    return [
        {
            "id": "groq",
            "label": "Groq",
            "model": active_model if settings.chat_llm_provider == "groq" else settings.groq_model,
            "configured": bool(settings.groq_api_key),
            "models": [
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
            ],
        },
        {
            "id": "openrouter",
            "label": "OpenRouter",
            "model": active_model if settings.chat_llm_provider == "openrouter" else settings.openrouter_model,
            "configured": bool(settings.openrouter_api_key),
            "models": list(dict.fromkeys([
                active_model if settings.chat_llm_provider == "openrouter" else settings.openrouter_model,
                settings.openrouter_model,
                "google/gemma-4-31b-it:free",
                "openai/gpt-oss-20b:free",
            ])),
        },
    ]


def _active_chat_model(settings) -> str:
    if settings.chat_llm_model:
        return settings.chat_llm_model
    if settings.chat_llm_provider == "openrouter":
        return settings.openrouter_model
    return settings.groq_model


def _settings_payload(settings) -> dict:
    providers = _llm_providers(settings)
    return {
        "chat_model": _active_chat_model(settings),
        "vision_model": settings.groq_vision_model,
        "embedding_model": settings.embedding_model,
        "top_k": settings.top_k,
        "rerank_top_n": settings.rerank_top_n,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chat_llm": {
            "active_provider": settings.chat_llm_provider,
            "active_model": _active_chat_model(settings),
            "providers": providers,
        },
        "ragas_llm": {
            "default_provider": settings.ragas_llm_provider,
            "providers": providers,
        },
    }


@router.post("/conversations", response_model=ConversationInfo)
def new_conversation() -> ConversationInfo:
    return ConversationInfo(**create_conversation())


@router.get("/conversations", response_model=list[ConversationInfo])
def conversations() -> list[ConversationInfo]:
    return [ConversationInfo(**item) for item in list_conversations()]


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[HistoryMessage],
)
def conversation_messages(conversation_id: str) -> list[HistoryMessage]:
    if not conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Sohbet bulunamadı.")

    return [
        HistoryMessage(**message)
        for message in list_history_messages(conversation_id)
    ]


@router.get(
    "/conversations/{conversation_id}/tool-calls",
    response_model=list[HistoryToolCall],
)
def conversation_tool_calls(conversation_id: str) -> list[HistoryToolCall]:
    if not conversation_exists(conversation_id):
        raise HTTPException(status_code=404, detail="Sohbet bulunamadı.")

    return [HistoryToolCall(**item) for item in list_tool_calls(conversation_id)]


@router.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str) -> dict[str, str]:
    if not delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Sohbet bulunamadı.")

    return {
        "conversation_id": conversation_id,
        "status": "deleted",
    }


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    logger.info(
        "chat.query conversation_id=%s query=%r",
        request.conversation_id,
        request.message,
    )
    return answer_message(
        request.message,
        conversation_id=request.conversation_id,
    )


@router.post("/chat/image", response_model=ChatResponse)
async def chat_with_image(
    message: str = Form(default=""),
    conversation_id: str = Form(...),
    image: UploadFile = File(...),
) -> ChatResponse:
    content_type = image.content_type or ""
    if content_type not in ALLOWED_CHAT_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Sadece JPG, PNG, WEBP veya GIF görselleri yükleyebilirsin.",
        )

    cleaned_message = message.strip() or "Görseldeki soruyu/konuyu inceleyip cevapla."
    settings = get_settings()
    filename = Path(image.filename or "chat-image").name
    extension = ALLOWED_CHAT_IMAGE_TYPES[content_type]
    image_id = f"img_{uuid4().hex[:12]}"
    stored_name = f"{image_id}{extension}"
    target_path = settings.chat_image_dir / stored_name
    target_path.write_bytes(await image.read())

    attachment = ImageAttachment(
        filename=filename,
        content_type=content_type,
        url=f"/chat-images/{stored_name}",
    )

    logger.info(
        "chat.image conversation_id=%s query=%r image=%s",
        conversation_id,
        cleaned_message,
        filename,
    )
    image_descriptions = []
    try:
        image_descriptions.append(describe_image(target_path, content_type, cleaned_message))
    except Exception as exc:
        logger.warning("vision.describe failed image=%s error=%s", filename, exc)

    return answer_message(
        cleaned_message,
        conversation_id=conversation_id,
        attachments=[attachment],
        image_descriptions=image_descriptions,
    )


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    settings = get_settings()
    documents = []

    for path in settings.upload_dir.iterdir():
        if path.is_file() and path.name != ".gitkeep":
            parts = path.name.split("_", 2)
            document_id = f"{parts[0]}_{parts[1]}" if len(parts) == 3 else path.stem
            document_name = parts[2] if len(parts) == 3 else path.name

            documents.append(
                DocumentInfo(
                    document_id=document_id,
                    document_name=document_name,
                    url=f"/uploaded-documents/{path.name}",
                )
            )

    return documents


@router.get("/settings")
def app_settings() -> dict:
    settings = get_settings()
    return _settings_payload(settings)


@router.patch("/settings/chat-llm")
def update_chat_llm_settings(request: ChatLlmSettingsRequest) -> dict:
    settings = get_settings()
    providers = {provider["id"]: provider for provider in _llm_providers(settings)}

    if request.provider not in providers:
        raise HTTPException(
            status_code=422,
            detail="Desteklenmeyen LLM provider.",
        )

    provider = providers[request.provider]
    if not provider["configured"]:
        raise HTTPException(
            status_code=400,
            detail=f"{provider['label']} API key tanımlı değil.",
        )

    if request.model not in provider["models"]:
        raise HTTPException(
            status_code=422,
            detail="Seçilen model bu provider için tanımlı değil.",
        )

    updated_settings = update_runtime_settings(
        {
            "chat_llm_provider": request.provider,
            "chat_llm_model": request.model,
        }
    )
    return _settings_payload(updated_settings)


@router.get("/evaluation/status")
def evaluation_status() -> dict:
    if not GOLDEN_DATASET_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Golden dataset bulunamadı.",
        )

    settings = get_settings()
    rows = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    categories = Counter(row.get("category", "belirsiz") for row in rows)
    behaviors = Counter(row.get("expected_behavior", "belirsiz") for row in rows)
    expected_sources = Counter(row.get("expected_source", "belirsiz") for row in rows)

    report = {
        "exists": RAGAS_REPORT_PATH.exists(),
        "path": str(RAGAS_REPORT_PATH.relative_to(PROJECT_ROOT)),
        "size_bytes": 0,
        "updated_at": None,
        "scores": None,
    }

    if RAGAS_REPORT_PATH.exists():
        report_stat = RAGAS_REPORT_PATH.stat()
        report["size_bytes"] = report_stat.st_size
        report["updated_at"] = report_stat.st_mtime
        try:
            report_payload = json.loads(RAGAS_REPORT_PATH.read_text(encoding="utf-8"))
            report["scores"] = report_payload.get("scores")
        except json.JSONDecodeError:
            report["scores"] = None

    return {
        "dataset_path": str(GOLDEN_DATASET_PATH.relative_to(PROJECT_ROOT)),
        "total_questions": len(rows),
        "categories": dict(categories),
        "expected_behaviors": dict(behaviors),
        "expected_sources": dict(expected_sources),
        "ragas_llm": {
            "default_provider": settings.ragas_llm_provider,
            "providers": _llm_providers(settings),
        },
        "questions": [
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "category": row.get("category"),
                "expected_behavior": row.get("expected_behavior"),
                "expected_source": row.get("expected_source"),
            }
            for row in rows
        ],
        "report": report,
    }
@router.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    settings = get_settings()

    matching_files = list(settings.upload_dir.glob(f"{document_id}_*"))

    if not matching_files:
        raise HTTPException(
            status_code=404,
            detail="Doküman bulunamadı.",
        )

    for path in matching_files:
        path.unlink()

    get_vector_store().delete_document(document_id)

    return {
        "document_id": document_id,
        "status": "deleted",
    }
@router.post(
    "/documents/preview",
    response_model=ChunkPreviewResponse,
)
async def preview_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=350, gt=0),
    overlap: int = Form(default=50, ge=0),
) -> ChunkPreviewResponse:
    if overlap >= chunk_size:
        raise HTTPException(
            status_code=422,
            detail="overlap, chunk_size değerinden küçük olmalıdır.",
        )

    filename = Path(file.filename or "preview_document").name

    with TemporaryDirectory() as temp_dir:
        temporary_path = Path(temp_dir) / filename
        temporary_path.write_bytes(await file.read())

        try:
            text = load_document_text(temporary_path)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    chunks = chunk_text(
        text=text,
        document_id="preview",
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return ChunkPreviewResponse(
        document_name=filename,
        chunk_size=chunk_size,
        overlap=overlap,
        chunks_count=len(chunks),
        chunks=[
            ChunkPreviewItem(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                length=len(chunk.text),
            )
            for chunk in chunks
        ],
    )
@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    settings = get_settings()
    document_id = f"doc_{uuid4().hex[:10]}"
    filename = Path(file.filename or "uploaded_document").name
    target_path = settings.upload_dir / f"{document_id}_{filename}"

    target_path.write_bytes(await file.read())

    try:
        chunks = build_chunks_from_sections(
            target_path,
            document_id=document_id,
        )
        get_vector_store().add_chunks(
            document_id=document_id,
            document_name=filename,
            chunks=chunks,
        )
    except ValueError as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("document.upload.embedding_failed filename=%s", filename)
        target_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail=f"Embedding servisi kullanılamıyor: {exc}",
        ) from exc

    return DocumentUploadResponse(
        document_id=document_id,
        document_name=filename,
        chunks_count=len(chunks),
        status="indexed",
    )
def build_chunks_from_sections(path: Path, document_id: str):
    settings = get_settings()
    chunks = []

    for section in load_document_sections(path):
        section_chunks = chunk_text(
            text=section.text,
            document_id=document_id,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            metadata=section.metadata,
            start_index=len(chunks) + 1,
        )

        chunks.extend(section_chunks)

    return chunks
