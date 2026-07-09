from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)


class Source(BaseModel):
    document_name: str
    chunk_id: str
    score: float
    page_number: int | None = None


class ImageAttachment(BaseModel):
    filename: str
    content_type: str
    url: str


class ToolCall(BaseModel):
    name: str
    input: dict
    output: dict
    turn_id: str | None = None


class HistoryToolCall(ToolCall):
    id: int
    created_at: datetime


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    tool_calls: list[ToolCall]
    turn_id: str | None = None
    attachments: list[ImageAttachment] = []


class ConversationInfo(BaseModel):
    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    turn_id: str | None = None
    sources: list[Source] = []
    attachments: list[ImageAttachment] = []

class ChunkPreviewItem(BaseModel):
    chunk_id: str
    text: str
    length: int

class ChunkPreviewResponse(BaseModel):
    document_name: str
    chunk_size: int
    overlap: int
    chunks_count: int
    chunks: list[ChunkPreviewItem]



class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    url: str | None = None

class DocumentUploadResponse(BaseModel):
    document_id: str
    document_name: str
    chunks_count: int
    status: Literal["indexed"]
