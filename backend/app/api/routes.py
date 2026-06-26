from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.models.schemas import ChatRequest, ChatResponse, DocumentUploadResponse
from app.services.agent import answer_message
from app.services.chunker import chunk_text
from app.services.document_loader import load_document_text
from app.services.vector_store import get_vector_store

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return answer_message(request.message)


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    settings = get_settings()
    document_id = f"doc_{uuid4().hex[:10]}"
    filename = Path(file.filename or "uploaded_document").name
    target_path = settings.upload_dir / f"{document_id}_{filename}"

    target_path.write_bytes(await file.read())

    try:
        text = load_document_text(target_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = chunk_text(text, document_id=document_id)
    get_vector_store().add_chunks(document_id=document_id, document_name=filename, chunks=chunks)

    return DocumentUploadResponse(
        document_id=document_id,
        document_name=filename,
        chunks_count=len(chunks),
        status="indexed",
    )
