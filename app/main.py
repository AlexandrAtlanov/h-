"""FastAPI-приложение веб-чата."""

from __future__ import annotations

import asyncio
import csv
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import APP_NAME, APP_VERSION, STATIC_DIR, AppConfig
from .models import (
    ChatRequest,
    ChatResponse,
    ChunkDetail,
    DocumentDetail,
    DocumentSummary,
    ExampleQuestion,
    HealthResponse,
    MetaResponse,
)
from .service import DocumentNotFoundError, LegalService, NotReadyError

logger = logging.getLogger(__name__)

def _load_examples(data_dir: Path, limit: int = 6) -> list[ExampleQuestion]:
    path = Path(data_dir) / "questions.csv"
    if not path.is_file():
        return []
    examples: list[ExampleQuestion] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                question = (row.get("question") or "").strip()
                if not question:
                    continue
                qid = str(row.get("id") or len(examples) + 1).strip()
                examples.append(ExampleQuestion(id=qid, question=question))
                if len(examples) >= limit:
                    break
    except (OSError, csv.Error):
        logger.exception("cannot read examples from %s", path)
        return []
    return examples

def create_app(
    service_factory: Callable[[], LegalService] | None = None,
    *,
    auto_build: bool = True,
) -> FastAPI:
    defaults = AppConfig()
    service = service_factory() if service_factory is not None else LegalService(defaults)
    settings = service.config

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.chat_sem = asyncio.Semaphore(settings.max_concurrent_answers)
        init_task: asyncio.Task[Any] | None = None
        if auto_build and service.status == "loading":
            init_task = asyncio.create_task(asyncio.to_thread(service.build))
        try:
            yield
        finally:
            if init_task is not None and not init_task.done():
                init_task.cancel()
            else:
                service.close()

    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description="Чат по базе юридических документов с подтверждающими фрагментами.",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.settings = settings
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:8]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        logger.info(
            "%s %s -> %s %.2fs request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            time.perf_counter() - started,
            request_id,
        )
        return response

    def require_ready() -> LegalService:
        if not service.ready:
            raise HTTPException(
                status_code=503,
                detail=service.error or "Индекс готовится, попробуйте через несколько секунд",
            )
        return service

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> JSONResponse:
        health = service.health()
        return JSONResponse(
            status_code=200 if health.ready else 503,
            content=health.model_dump(mode="json"),
        )

    @app.get("/api/meta", response_model=MetaResponse)
    async def meta() -> MetaResponse:
        return MetaResponse(
            app_name=APP_NAME,
            version=APP_VERSION,
            examples=_load_examples(service.config.data_dir),
        )

    @app.get("/api/documents", response_model=list[DocumentSummary])
    async def documents() -> list[DocumentSummary]:
        return require_ready().list_documents()

    @app.get("/api/documents/{doc_id}", response_model=DocumentDetail)
    async def document(doc_id: str) -> DocumentDetail:
        try:
            return require_ready().get_document(doc_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Документ не найден") from exc

    @app.get("/api/chunks/{chunk_id:path}", response_model=ChunkDetail)
    async def chunk(chunk_id: str) -> ChunkDetail:
        try:
            return require_ready().get_chunk(chunk_id)
        except DocumentNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Фрагмент не найден") from exc

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        current = require_ready()
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])
        history = [message.model_dump() for message in payload.history]
        semaphore: asyncio.Semaphore = getattr(
            request.app.state, "chat_sem", None
        ) or asyncio.Semaphore(settings.max_concurrent_answers)
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        current.answer, payload.message, history, request_id
                    ),
                    timeout=settings.chat_timeout_sec,
                )
            except asyncio.TimeoutError as exc:
                raise HTTPException(
                    status_code=504,
                    detail="Модель не ответила за отведённое время. Попробуйте ещё раз.",
                ) from exc
            except NotReadyError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:8])
        logger.exception("unhandled error request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Внутренняя ошибка сервера. Попробуйте повторить запрос.",
                "request_id": request_id,
            },
        )

    return app

app = create_app()
