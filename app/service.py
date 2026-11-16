"""Сервисный слой веб-чата: документы, индекс, агент и адаптер к UI."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import httpx

import solution
from rag import Chunk, VectorStore, get_or_create_vectorstore, load_documents

if TYPE_CHECKING:  
    from agent import AgentResult, LegalAgent

from .config import APP_VERSION, AppConfig, find_kb
from .models import (
    ChatResponse,
    ChunkDetail,
    DocumentChunk,
    DocumentDetail,
    DocumentSummary,
    HealthResponse,
    Source,
)

logger = logging.getLogger(__name__)

STATUS_MAP = {
    "ANSWERED": "answered",
    "PARTIAL": "insufficient",
    "REFUSED": "refused",
    "OFF_TOPIC": "offtopic",
    "CLARIFY": "clarify",
}

NO_LLM_MESSAGE = (
    "В веб-чате не настроены ключи доступа к модели (LLM_BASE_URL и LLM_API_KEY). "
    "Документы доступны для просмотра, но ответить на вопрос по ним пока нельзя."
)
SAFE_REFUSAL = (
    "Не удалось получить обоснованный ответ по документам. "
    "Попробуйте переформулировать вопрос или повторите запрос позже."
)

class NotReadyError(RuntimeError):
    """Индекс ещё не готов."""

class DocumentNotFoundError(KeyError):
    """Документ или фрагмент не найден."""

def doc_id_for(filename: str) -> str:
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]

def _short_error(exc: BaseException, limit: int = 300) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text[:limit]

def _chunk_order(chunk: Chunk) -> int:
    suffix = chunk.chunk_id.rsplit("::", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 0

def _collapse_with_map(text: str) -> tuple[str, list[int]]:
    """Схлопывает пробелы, сохраняя соответствие позиций исходному тексту."""
    chars: list[str] = []
    index_map: list[int] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            start = i
            while i < len(text) and text[i].isspace():
                i += 1
            chars.append(" ")
            index_map.append(start)
        else:
            chars.append(text[i])
            index_map.append(i)
            i += 1
    return "".join(chars), index_map

def locate_chunks(text: str, chunks: Iterable[Chunk]) -> dict[str, tuple[int | None, int | None]]:
    """Позиции body каждого чанка в полном тексте документа.

    Чанки идут в порядке документа, поэтому сначала ищем после предыдущего
    фрагмента — так одинаковые пункты не схлопываются в один и тот же offset.
    """
    ordered = sorted(chunks, key=_chunk_order)
    result: dict[str, tuple[int | None, int | None]] = {}
    if not text:
        return {chunk.chunk_id: (None, None) for chunk in ordered}

    cursor = 0
    for chunk in ordered:
        body = chunk.body or ""
        if not body:
            result[chunk.chunk_id] = (None, None)
            continue

        start = -1
        if chunk.doc_type == "law" and cursor > 0:
            start = text.find(body, max(0, cursor - 500))
        if start < 0:
            start = text.find(body, cursor)
        if start < 0:
            start = text.find(body)
        if start >= 0:
            end = start + len(body)
            result[chunk.chunk_id] = (start, end)
            cursor = max(cursor, end)
            continue

        collapsed_text, index_map = _collapse_with_map(text)
        collapsed_body = " ".join(body.split()).strip()
        pos = collapsed_text.find(collapsed_body) if collapsed_body else -1
        if pos < 0 or not index_map:
            result[chunk.chunk_id] = (None, None)
            continue
        last = pos + len(collapsed_body) - 1
        loose_start = index_map[pos]
        loose_end = index_map[last] + 1
        result[chunk.chunk_id] = (loose_start, loose_end)
        cursor = max(cursor, loose_end)
    return result

@dataclass
class DocumentInfo:
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
    counterparties: list[str] = field(default_factory=list)
    amends: str = ""
    amended_clauses: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)

    def summary(self) -> DocumentSummary:
        return DocumentSummary(
            id=self.id,
            filename=self.filename,
            title=self.title,
            doc_type=self.doc_type,
            doc_number=self.doc_number,
            doc_date=self.doc_date,
            edition=self.edition,
            status=self.status,
            valid_from=self.valid_from,
            valid_to=self.valid_to,
            counterparties=list(self.counterparties),
            amends=self.amends,
            amended_clauses=list(self.amended_clauses),
            chunk_count=len(self.chunks),
        )

class LegalService:
    """Один экземпляр на процесс: индекс и агент создаются один раз."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        store: VectorStore | None = None,
        agent: LegalAgent | Any | None = None,
        documents: dict[str, str] | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.store = store
        self.agent = agent
        self.documents: dict[str, str] = dict(documents or {})
        self.http = http
        self.status: str = "ready" if store is not None and documents is not None else "loading"
        self.error: str | None = None
        self.warning: str | None = None

        self._doc_infos: dict[str, DocumentInfo] = {}
        self._doc_by_id: dict[str, str] = {}
        self._offsets: dict[str, tuple[int | None, int | None]] = {}
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._owns_http = False
        if self.status == "ready":
            self._index_documents()

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def build(self) -> None:
        """Строит/загружает индекс и агент. Вызывается в фоновом потоке."""
        with self._build_lock:
            if self.status != "loading":
                return
            http: httpx.Client | None = None
            try:
                kb_path = find_kb(self.config.data_dir)
                documents = dict(load_documents(kb_path))
                http = httpx.Client(
                    timeout=httpx.Timeout(185.0, connect=15.0),
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                )
                store = get_or_create_vectorstore(
                    kb_path,
                    self.config.vectorstore_dir,
                    http,
                    solution.EMB_BASE,
                    solution.EMB_KEY,
                    model=solution.EMB_MODEL,
                    passage_task="",
                )

                agent = None
                warning = None
                if solution.API_BASE and solution.LLM_KEY:
                    try:
                        agent = solution.make_agent(store, http)
                    except Exception as exc:  # noqa: BLE001
                        warning = f"Агент не загрузился: {_short_error(exc)}"
                        logger.exception("agent init failed")
                else:
                    warning = (
                        "LLM_BASE_URL/LLM_API_KEY не заданы: документы доступны, "
                        "чат вернёт отказ"
                    )

                with self._lock:
                    self.documents = documents
                    self.store = store
                    self.http = http
                    self.agent = agent
                    self.warning = warning
                    self.error = None
                    self.status = "ready"
                    self._owns_http = True
                    self._index_documents()
                logger.info(
                    "service ready: documents=%s chunks=%s vectors=%s",
                    len(documents),
                    len(store.chunks),
                    store.vectors is not None,
                )
            except Exception as exc:  # noqa: BLE001
                if http is not None:
                    http.close()
                with self._lock:
                    self.status = "error"
                    self.error = _short_error(exc)
                logger.exception("service init failed")

    def mark_ready(self) -> None:
        """Для тестов и сценариев с уже готовым store."""
        with self._lock:
            self.status = "ready"
            self.error = None
            self._index_documents()

    def close(self) -> None:
        with self._lock:
            if self._owns_http and self.http is not None:
                self.http.close()
                self.http = None
                self._owns_http = False

    def _index_documents(self) -> None:
        by_file: dict[str, list[Chunk]] = {}
        if self.store is not None:
            for chunk in self.store.chunks:
                by_file.setdefault(chunk.source_filename, []).append(chunk)

        self._doc_infos = {}
        self._doc_by_id = {}
        self._offsets = {}
        filenames = sorted(set(by_file) | set(self.documents))
        for filename in filenames:
            chunks = sorted(by_file.get(filename, []), key=_chunk_order)
            info = self._make_info(filename, chunks)
            self._doc_infos[info.id] = info
            self._doc_by_id[filename] = info.id
            text = self.documents.get(filename)
            if text:
                self._offsets.update(locate_chunks(text, chunks))

    @staticmethod
    def _make_info(filename: str, chunks: list[Chunk]) -> DocumentInfo:
        first = chunks[0] if chunks else None
        return DocumentInfo(
            id=doc_id_for(filename),
            filename=filename,
            title=(first.doc_title if first else "") or filename,
            doc_type=(first.doc_type if first else ""),
            doc_number=(first.doc_number if first else ""),
            doc_date=(first.doc_date if first else ""),
            edition=(first.edition if first else ""),
            status=(first.status if first else ""),
            valid_from=(first.valid_from if first else ""),
            valid_to=(first.valid_to if first else ""),
            counterparties=list(first.counterparties if first else []),
            amends=(first.amends if first else ""),
            amended_clauses=list(first.amended_clauses if first else []),
            chunks=chunks,
        )

    def health(self) -> HealthResponse:
        with self._lock:
            return HealthResponse(
                status=self.status,  # type: ignore[arg-type]
                ready=self.status == "ready",
                documents=len(self._doc_infos),
                chunks=len(self.store.chunks) if self.store is not None else 0,
                llm_model=solution.LLM_MODEL,
                emb_model=solution.EMB_MODEL,
                vectors=bool(self.store is not None and self.store.vectors is not None),
                error=self.error,
                warning=self.warning,
                version=APP_VERSION,
            )

    def list_documents(self) -> list[DocumentSummary]:
        with self._lock:
            infos = sorted(
                self._doc_infos.values(),
                key=lambda item: (item.doc_type == "law", item.title.lower(), item.filename),
            )
            return [info.summary() for info in infos]

    def get_document(self, doc_id: str) -> DocumentDetail:
        with self._lock:
            info = self._doc_infos.get(doc_id)
            if info is None:
                raise DocumentNotFoundError(doc_id)
            text = self.documents.get(info.filename) or "\n\n".join(c.body for c in info.chunks)
            chunks = [
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    locator=chunk.locator,
                    quote=chunk.body,
                    clauses=list(chunk.clauses),
                    start=self._offsets.get(chunk.chunk_id, (None, None))[0],
                    end=self._offsets.get(chunk.chunk_id, (None, None))[1],
                )
                for chunk in info.chunks
            ]
            summary = info.summary()
            return DocumentDetail(**summary.model_dump(), text=text, chunks=chunks)

    def get_chunk(self, chunk_id: str) -> ChunkDetail:
        with self._lock:
            chunk = self.store.by_id.get(chunk_id) if self.store is not None else None
            if chunk is None:
                raise DocumentNotFoundError(chunk_id)
            info = self._doc_infos.get(doc_id_for(chunk.source_filename))
            start, end = self._offsets.get(chunk_id, (None, None))
            if start is None:
                text = self.documents.get(chunk.source_filename)
                if text:
                    start, end = locate_chunks(text, [chunk]).get(chunk_id, (None, None))
            return ChunkDetail(
                chunk_id=chunk.chunk_id,
                doc_id=info.id if info else doc_id_for(chunk.source_filename),
                filename=chunk.source_filename,
                doc_title=chunk.doc_title,
                locator=chunk.locator,
                quote=chunk.body,
                clauses=list(chunk.clauses),
                start=start,
                end=end,
            )

    def answer(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> ChatResponse:
        with self._lock:
            if self.status != "ready":
                raise NotReadyError(self.error or "Индекс ещё готовится")
            store = self.store
            agent = self.agent
            http = self.http

        request_id = request_id or uuid.uuid4().hex[:8]
        started = time.perf_counter()
        text = (message or "").strip()[: self.config.max_message_chars]
        if not text:
            raise ValueError("Пустой вопрос")
        messages = self._sanitize_history(history, text)

        result: AgentResult | None = None
        error: str | None = None
        if agent is not None:
            try:
                result = agent.answer(history=messages)
            except Exception as exc:  # noqa: BLE001
                error = _short_error(exc)
                logger.exception("agent failed request_id=%s", request_id)

        if result is None:
            return self._simple_fallback(messages, started, request_id, error)

        answer = solution.clean_answer(result.answer)
        if not answer:
            return self._simple_fallback(messages, started, request_id, error or "пустой ответ агента")

        trace = result.trace if isinstance(result.trace, dict) else {}
        timings = {
            str(key): float(value)
            for key, value in (trace.get("timings") or {}).items()
            if isinstance(value, (int, float))
        }
        timings.setdefault("total", round(time.perf_counter() - started, 2))
        missing_raw = trace.get("missing_info")
        missing = missing_raw if isinstance(missing_raw, str) and missing_raw.strip() else None
        return ChatResponse(
            answer=answer,
            status=self._map_status(result.status),  # type: ignore[arg-type]
            sources=self._sources_from_result(result.sources, store),
            missing_info=missing,
            timings=timings,
            pipeline="agent",
            llm_calls=int(trace.get("llm_calls") or 0),
            request_id=request_id,
        )

    def _simple_fallback(
        self,
        messages: list[dict[str, str]],
        started: float,
        request_id: str,
        error: str | None = None,
    ) -> ChatResponse:
        elapsed = round(time.perf_counter() - started, 2)
        if self.store is None or self.http is None:
            return ChatResponse(
                answer=SAFE_REFUSAL,
                status="refused",
                missing_info=error or "Индекс не готов",
                timings={"total": elapsed},
                pipeline="simple",
                request_id=request_id,
            )
        if not solution.API_BASE or not solution.LLM_KEY:
            return ChatResponse(
                answer=NO_LLM_MESSAGE,
                status="refused",
                missing_info="Не заданы LLM_BASE_URL и LLM_API_KEY",
                timings={"total": elapsed},
                pipeline="simple",
                request_id=request_id,
            )

        answer = ""
        try:
            answer = solution.clean_answer(
                solution.simple_answer(self.store, self.http, messages)
            )
        except Exception as exc:  # noqa: BLE001
            error = error or _short_error(exc)
            logger.exception("simple fallback failed request_id=%s", request_id)

        sources = self._search_sources(messages[-1]["content"]) if answer else []
        if not answer:
            return ChatResponse(
                answer=SAFE_REFUSAL,
                status="refused",
                sources=sources,
                missing_info=error or "Модель не вернула ответ",
                timings={"total": round(time.perf_counter() - started, 2)},
                pipeline="simple",
                request_id=request_id,
            )
        return ChatResponse(
            answer=answer,
            status="answered",
            sources=sources,
            timings={"total": round(time.perf_counter() - started, 2)},
            pipeline="simple",
            llm_calls=1,
            request_id=request_id,
        )

    def _search_sources(self, query: str) -> list[Source]:
        if self.store is None:
            return []
        kwargs: dict[str, Any] = {}
        if self.http is not None and solution.EMB_BASE and solution.EMB_KEY:
            kwargs = {
                "client": self.http,
                "base_url": solution.EMB_BASE,
                "api_key": solution.EMB_KEY,
                "model": solution.EMB_MODEL,
            }
        try:
            hits = self.store.search(query, top_k=4, **kwargs)
        except Exception:  # noqa: BLE001
            logger.exception("source search failed")
            return []
        return [self._source_from_chunk(chunk, score) for chunk, score in hits]

    @staticmethod
    def _map_status(status: str | None) -> str:
        return STATUS_MAP.get((status or "").upper(), "insufficient")

    def _sources_from_result(
        self, raw_sources: list[dict[str, Any]] | None, store: VectorStore | None
    ) -> list[Source]:
        sources: list[Source] = []
        seen: set[str] = set()
        for item in raw_sources or []:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk = store.by_id.get(chunk_id) if store is not None else None
            if chunk is not None:
                sources.append(self._source_from_chunk(chunk, self._score(item.get("score"))))
                continue
            filename = str(item.get("filename") or "")
            sources.append(
                Source(
                    chunk_id=chunk_id,
                    doc=filename,
                    doc_title=filename,
                    doc_id=self._doc_by_id.get(filename) or doc_id_for(filename),
                    locator=str(item.get("header") or ""),
                    quote=str(item.get("text") or ""),
                    score=self._score(item.get("score")),
                )
            )
        return sources

    def _source_from_chunk(self, chunk: Chunk, score: float) -> Source:
        return Source(
            chunk_id=chunk.chunk_id,
            doc=chunk.source_filename,
            doc_title=chunk.doc_title or chunk.source_filename,
            doc_id=self._doc_by_id.get(chunk.source_filename)
            or doc_id_for(chunk.source_filename),
            locator=chunk.locator,
            quote=chunk.body,
            score=round(float(score), 4),
        )

    @staticmethod
    def _score(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _sanitize_history(
        self, history: list[dict[str, Any]] | None, message: str
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for item in history or []:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            content = content.strip()
            if content:
                messages.append({"role": role, "content": content})

        messages = messages[-self.config.max_history_messages :]
        if not messages or not (
            messages[-1]["role"] == "user" and messages[-1]["content"] == message
        ):
            messages.append({"role": "user", "content": message})

        kept: list[dict[str, str]] = []
        total = 0
        for item in reversed(messages):
            total += len(item["content"])
            if kept and total > self.config.max_history_chars:
                break
            kept.append(item)
        kept.reverse()
        if kept[-1]["role"] != "user":
            kept.append({"role": "user", "content": message})
        return kept

