"""Test script for the agentic RAG pipeline in agent.py.

Loads .txt files from a folder (or .zip), builds/loads the FAISS vector store,
then runs sample legal questions through LegalAgent (planner -> retrieval ->
reranker -> critic/answer). Embedding model, reranker and .env variable names
follow the same contract used here: EMBEDDER_*, RERANKER_*, EMB_*, LLM_*.
"""

from __future__ import annotations

from dotenv import load_dotenv
import json
import os
from pathlib import Path
import zipfile

import httpx

from agent import AgentConfig, LegalAgent
from rag import get_or_create_vectorstore


load_dotenv()

# Configuration (defaults to Jina AI for embeddings)
KB_FOLDER = Path(os.getenv("KB_FOLDER", "data/knowledge-base"))
KB_ZIP = Path(os.getenv("KB_ZIP", "data/knowledge-base.zip"))
STORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstore"))

EMB_BASE_URL = os.getenv("EMBEDDER_BASE_URL", "https://api.jina.ai/v1").rstrip("/")
EMB_API_KEY = os.getenv("EMBEDDER_API_KEY") or os.getenv("JINA_API_KEY", "")
EMB_MODEL = os.getenv("EMB_MODEL", "jina-embeddings-v3")
EMB_TASK_PASSAGE = os.getenv("EMB_TASK_PASSAGE", "retrieval.passage")
EMB_TASK_QUERY = os.getenv("EMB_TASK_QUERY", "retrieval.query")

# Reranker (optional; configured via RERANKER_* in .env, falls back to the embedder)
RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL", EMB_BASE_URL).rstrip("/")
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY") or EMB_API_KEY
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


RERANKER_TOP_N = _env_int("RERANKER_TOP_N", 4)

# LLM is required by the agent (planner + critic/answer)
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")).rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-v4.1-flash")

SAMPLE_QUESTIONS = [
    "Какова арендная плата по договору аренды склада № А-08/2024 и до какого числа она вносится?",
    "По какому адресу доставляется товар по договору поставки № П-17/2024 после 1 октября 2024 года?",
    "Какая сумма договора требует согласования с финансовым директором по действующему регламенту 2025 года?",
    "В какой срок проводится санкционная проверка иностранного контрагента службой комплаенс?",
    "Какова основная ставка налога на прибыль организаций согласно статье 284 НК РФ?",
]


def ensure_txt_folder(folder: Path, zip_fallback: Path) -> Path:
    """Ensures a folder of .txt files exists (extracts from zip_fallback if needed)."""
    if folder.is_dir() and any(folder.rglob("*.txt")):
        return folder

    if zip_fallback.is_file():
        print(f"Распаковка {zip_fallback} -> {folder} ...")
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_fallback) as zf:
            zf.extractall(folder)
        return folder

    raise FileNotFoundError(
        f"Не найдена папка с .txt файлами ({folder}) или архив ({zip_fallback}). "
        "Укажите путь через KB_FOLDER или положите файлы в data/kb_txts."
    )


def build_config() -> AgentConfig:
    """Builds AgentConfig from the same env variables used by this script."""
    return AgentConfig(
        llm_base_url=LLM_BASE_URL,
        llm_api_key=LLM_API_KEY,
        llm_model=LLM_MODEL,
        emb_base_url=EMB_BASE_URL,
        emb_api_key=EMB_API_KEY,
        emb_model=EMB_MODEL,
        emb_query_task=EMB_TASK_QUERY,
        emb_passage_task=EMB_TASK_PASSAGE,
        reranker_base_url=RERANKER_BASE_URL,
        reranker_api_key=RERANKER_API_KEY,
        reranker_model=RERANKER_MODEL,
        reranker_top_n=RERANKER_TOP_N,
    )


def main() -> None:
    if not EMB_API_KEY:
        raise ValueError(
            "Не задан ключ эмбеддера: установите JINA_API_KEY или EMBEDDER_API_KEY в .env или окружении"
        )
    if not LLM_BASE_URL or not LLM_API_KEY:
        raise ValueError(
            "Агенту нужна LLM: установите LLM_BASE_URL и LLM_API_KEY (или OPENAI_*) в .env или окружении"
        )

    txt_folder = ensure_txt_folder(KB_FOLDER, KB_ZIP)
    print(f"Источник документов (папка): {txt_folder}")

    cfg = build_config()
    with httpx.Client(timeout=cfg.llm_timeout) as client:
        # 1. Load existing vectorstore from STORE_DIR or create a new one from txt_folder
        vectorstore = get_or_create_vectorstore(
            source_path=txt_folder,
            store_dir=STORE_DIR,
            client=client,
            base_url=cfg.emb_base_url,
            api_key=cfg.emb_api_key,
            model=cfg.emb_model,
            passage_task=cfg.emb_passage_task,
            chunk_size=1000,
            overlap=150,
        )
        print(
            f"Векторная база готова: документов={vectorstore.meta.get('num_documents')}, "
            f"чанков={len(vectorstore.chunks)}, модель={vectorstore.meta.get('model')}\n"
        )

        # 2. Run sample questions through the agentic pipeline (agent.py)
        agent = LegalAgent(vectorstore, cfg, client)
        if cfg.reranker_model and cfg.reranker_top_n > 0:
            print(f"Реранкер: {cfg.reranker_model} (top_n={cfg.reranker_top_n})\n")
        else:
            print("Реранкер отключён\n")

        for idx, question in enumerate(SAMPLE_QUESTIONS, 1):
            print("=" * 80)
            print(f"ВОПРОС {idx}: {question}")
            result = agent.answer(question=question)
            print(f"\n  [{result.status}] {result.answer}\n")

            for rank, source in enumerate(result.sources, 1):
                preview = (source.get("text") or "").replace("\n", " ")[:200]
                print(f"  [{rank}] id={source['chunk_id']} | score={source['score']}")
                print(f"      Файл: {source['filename']}")
                print(f"      Текст: {preview}...")

            print(f"\n  trace: {json.dumps(result.trace, ensure_ascii=False)}")
            print()


if __name__ == "__main__":
    main()