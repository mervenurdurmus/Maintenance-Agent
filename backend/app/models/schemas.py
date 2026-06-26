from typing import Literal

from pydantic import BaseModel, Field

RouteName = Literal[
    "alarm_question",
    "maintenance_question",
    "safety_question",
    "date_question",
    "period_calculation",
    "general_question",
    "smalltalk",
    "out_of_scope",
]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str = "demo-session-1"


class Source(BaseModel):
    document_name: str
    chunk_id: str
    score: float


class ToolCall(BaseModel):
    name: str
    input: dict
    output: dict


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    tool_calls: list[ToolCall]
    route: RouteName


class DocumentUploadResponse(BaseModel):
    document_id: str
    document_name: str
    chunks_count: int
    status: Literal["indexed"]
