"""Pydantic-схемы HTTP API веб-чата."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChatRole = Literal["user", "assistant"]
AnswerStatus = Literal["answered", "insufficient", "refused", "clarify", "offtopic"]
ServiceStatus = Literal["loading", "ready", "error"]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ChatMessage(ApiModel):
    role: ChatRole
    content: str = Field(min_length=1, max_length=16_000)


class ChatRequest(ApiModel):
    message: str = Field(min_length=1, max_length=4_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)


class Source(ApiModel):
    chunk_id: str
    doc: str
    doc_title: str = ""
    doc_id: str = ""
    locator: str = ""
    quote: str = ""
    score: float = 0.0


class ChatResponse(ApiModel):
    answer: str
    status: AnswerStatus
    sources: list[Source] = Field(default_factory=list)
    missing_info: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    pipeline: Literal["agent", "simple"] = "agent"
    llm_calls: int = 0
    request_id: str = ""


class DocumentSummary(ApiModel):
    id: str
    filename: str
    title: str
    doc_type: str = ""
    doc_number: str = ""
    doc_date: str = ""
    edition: str = ""
    status: str = ""
    valid_from: str = ""
    valid_to: str = ""
    counterparties: list[str] = Field(default_factory=list)
    amends: str = ""
    amended_clauses: list[str] = Field(default_factory=list)
    chunk_count: int = 0


class DocumentChunk(ApiModel):
    chunk_id: str
    locator: str = ""
    quote: str = ""
    clauses: list[str] = Field(default_factory=list)
    start: int | None = None
    end: int | None = None


class DocumentDetail(DocumentSummary):
    text: str = ""
    chunks: list[DocumentChunk] = Field(default_factory=list)


class ChunkDetail(DocumentChunk):
    doc_id: str = ""
    filename: str = ""
    doc_title: str = ""


class ExampleQuestion(ApiModel):
    id: str
    question: str


class MetaResponse(ApiModel):
    app_name: str
    version: str
    examples: list[ExampleQuestion] = Field(default_factory=list)


class HealthResponse(ApiModel):
    status: ServiceStatus
    ready: bool
    documents: int = 0
    chunks: int = 0
    llm_model: str = ""
    emb_model: str = ""
    vectors: bool = False
    error: str | None = None
    warning: str | None = None
    version: str = ""
