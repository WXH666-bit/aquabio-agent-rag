from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator
from aquabio_mrag.conversation import ConversationStore


class ChatOptions(BaseModel):
    memory_enabled: bool = True
    rag_enabled: bool = True
    vision_enabled: bool = True
    pdf_enabled: bool = True
    mcp_enabled: bool = True
    mcp_retrieval_enabled: bool = False
    image_search_enabled: bool = True
    citation_enabled: bool = True
    log_enabled: bool = True
    hitl_enabled: bool = False
    detection_enabled: bool = True


class AttachmentRef(BaseModel):
    file_id: str
    type: str


class SessionModel(BaseModel):
    @field_validator("session_id", check_fields=False)
    @classmethod
    def validate_session_id(cls, value):
        return ConversationStore.normalize_session_id(value) if value is not None else value


class ChatRequest(SessionModel):
    session_id: str
    query: str = ""
    attachments: list[AttachmentRef] = Field(default_factory=list)
    options: ChatOptions = Field(default_factory=ChatOptions)


class SessionCreate(SessionModel):
    title: str = "新会话"
    session_id: str | None = None


class SessionUpdate(BaseModel):
    title: str | None = None
    is_favorite: bool | None = None
    tags: list[str] | None = None


class FeedbackRequest(SessionModel):
    session_id: str
    turn_id: str
    rating: int = Field(ge=-1, le=1)
    comment: str = Field(default="", max_length=1000)


class ChatResponse(BaseModel):
    session_id: str
    turn_id: str
    answer: str
    answer_type: str
    images: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    warnings: list[str]
    memory: dict[str, Any]
    route: dict[str, Any]
    model: dict[str, Any] = Field(default_factory=dict)
    pending_review: bool = False
    raw_state: dict[str, Any] | None = None
    detection: dict[str, Any] = Field(default_factory=dict)
    detection_comparison: dict[str, Any] = Field(default_factory=dict)
    detection_visualization_url: str = ""
