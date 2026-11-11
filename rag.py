"""RAG module for PSB Legal Assistant case: document loading, chunking with header prepending,
Jina / OpenAI-compatible embeddings, and FAISS vector store persistence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
import numpy as np

try:
    import faiss  # type: ignore[import-untyped]
except ImportError:
    faiss = None


SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx"}

# Matches numbered section headers in contracts/regulations (e.g., "1. Предмет договора", "1.1. ...")
SECTION_START_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.\s+\S")
# Matches law article headers (e.g., "Статья 284. Налоговые ставки")
ARTICLE_HEADER_RE = re.compile(r"^\s*Статья\s+\d+(?:\.\d+)*\.?\s*", re.IGNORECASE)
# Sentence / clause splitter for oversized single lines (e.g., long clauses in НК РФ)
CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")


@dataclass
class Chunk:
    """Single text chunk with source metadata and prepended document header."""

    chunk_id: str
    source_filename: str
    header: str
    body: str
    text: str


def read_docx(raw: bytes) -> str:
    """Extracts plain text from a .docx byte stream without external dependencies."""
    with zipfile.ZipFile(io.BytesIO(raw)) as doc:
        xml = doc.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join(
        "".join(x.text or "" for x in paragraph.iter(ns + "t"))
        for paragraph in root.iter(ns + "p")
    )


def normalize_text(text: str) -> str:
    """Normalizes line endings, non-breaking spaces, and strips empty lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def load_documents(source_path: str | Path) -> list[tuple[str, str]]:
    """Loads all .txt, .md, and .docx documents from a .zip file or a directory."""
    path = Path(source_path)
    documents: list[tuple[str, str]] = []

    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith("/") or name.startswith("__MACOSX/"):
                    continue
                suffix = Path(name).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    continue
                raw = archive.read(name)
                try:
                    text = read_docx(raw) if suffix == ".docx" else raw.decode("utf-8-sig")
                except (UnicodeDecodeError, KeyError, ValueError, zipfile.BadZipFile, ET.ParseError):
                    print(f"Пропущен нечитаемый документ: {name}", file=sys.stderr)
                    continue
                cleaned = normalize_text(text)
                if cleaned:
                    documents.append((name, cleaned))
    elif path.is_dir():
        for file_path in sorted(path.rglob("*")):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            suffix = file_path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue
            rel_name = str(file_path.relative_to(path)).replace("\\", "/")
            raw = file_path.read_bytes()
            try:
                text = read_docx(raw) if suffix == ".docx" else raw.decode("utf-8-sig")
            except (UnicodeDecodeError, KeyError, ValueError, zipfile.BadZipFile, ET.ParseError):
                print(f"Пропущен нечитаемый документ: {rel_name}", file=sys.stderr)
                continue
            cleaned = normalize_text(text)
            if cleaned:
                documents.append((rel_name, cleaned))
    else:
        raise FileNotFoundError(f"Источник базы знаний не найден или не поддерживается: {path}")

    if not documents:
        raise ValueError(f"В источнике {path} нет читаемых документов")
    return documents


def extract_header(lines: list[str], max_header_lines: int = 6) -> tuple[str, list[str]]:
    """Separates the document header (title, parties/dates/active status) from the body lines.

    Returns:
        (header_text, remaining_body_lines)
    """
    if not lines:
        return "", []

    # Case 1: Law files starting with "Источник:" / "URL:" / "Дата выгрузки:" / "Статья ..."
    if lines[0].startswith("Источник:") or any(
        ARTICLE_HEADER_RE.match(l) for l in lines[:max_header_lines]
    ):
        header_end = min(len(lines), 4)
        for idx, line in enumerate(lines[:max_header_lines]):
            if ARTICLE_HEADER_RE.match(line):
                header_end = idx + 1
                # Include amendment note right below article title if present
                if header_end < len(lines) and (
                    "изменен" in lines[header_end].lower()
                    or "редакции" in lines[header_end].lower()
                ):
                    header_end += 1
                break
        header = "\n".join(lines[:header_end])
        body_lines = lines[header_end:] if header_end < len(lines) else lines
        return header, body_lines

    # Case 2: Contracts / regulations with numbered sections ("1. Предмет договора")
    for idx, line in enumerate(lines[: max_header_lines + 2]):
        if idx > 0 and SECTION_START_RE.match(line):
            header = "\n".join(lines[:idx])
            return header, lines[idx:]

    # Case 3: Short letters, memos, or supplementary agreements without "1." sections
    header_end = min(len(lines), 4)
    header = "\n".join(lines[:header_end])
    body_lines = lines[header_end:] if len(lines) > header_end else lines
    return header, body_lines


def split_oversized_line(line: str, max_chars: int) -> list[str]:
    """Splits a single line that exceeds max_chars along clause/sentence boundaries (; or .)."""
    if len(line) <= max_chars:
        return [line]

    parts = CLAUSE_SPLIT_RE.split(line)
    sub_lines: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                sub_lines.append(current)
            # Hard-split if a single clause without ';' or '.' still exceeds max_chars
            while len(part) > max_chars:
                sub_lines.append(part[:max_chars].strip())
                part = part[max_chars:].strip()
            current = part
    if current:
        sub_lines.append(current)
    return sub_lines


def format_chunk_text(source_filename: str, header: str, body: str) -> str:
    """Formats the full chunk text with filename and document header prepended."""
    if header and header != body and not body.startswith(header):
        return f"Документ: {source_filename}\n{header}\n---\n{body}"
    return f"Документ: {source_filename}\n{body}"


def chunk_file(
    source_filename: str,
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    """Splits a single document into overlapping chunks with filename and header prepended.

    Args:
        source_filename: Name or relative path of the file.
        text: Raw document text.
        chunk_size: Target maximum character count for the chunk body (excluding prepended header).
        overlap: Target character overlap between consecutive chunks from the same file.
    """
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    raw_lines = cleaned.split("\n")
    header, body_raw_lines = extract_header(raw_lines)

    # Split any single line that is longer than chunk_size into smaller sentence/clause segments
    lines: list[str] = []
    for line in body_raw_lines:
        lines.extend(split_oversized_line(line, chunk_size))

    # If the entire document body fits in one chunk, return a single chunk
    full_body = "\n".join(lines)
    if len(full_body) <= chunk_size:
        chunk_text = format_chunk_text(source_filename, header, full_body)
        return [
            Chunk(
                chunk_id=f"{source_filename}::0",
                source_filename=source_filename,
                header=header,
                body=full_body,
                text=chunk_text,
            )
        ]

    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line)
        added_len = line_len if not current_lines else line_len + 1  # +1 for '\n'

        if current_lines and (current_len + added_len > chunk_size):
            body_str = "\n".join(current_lines)
            chunk_idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{source_filename}::{chunk_idx}",
                    source_filename=source_filename,
                    header=header,
                    body=body_str,
                    text=format_chunk_text(source_filename, header, body_str),
                )
            )

            # Retain trailing lines for overlap only if they fit alongside the next line
            overlap_lines: list[str] = []
            overlap_len = 0
            for prev_line in reversed(current_lines):
                add_prev = len(prev_line) if not overlap_lines else len(prev_line) + 1
                if overlap_len + add_prev > overlap:
                    break
                if overlap_len + add_prev + 1 + line_len > chunk_size:
                    break
                overlap_lines.append(prev_line)
                overlap_len += add_prev
            overlap_lines.reverse()

            current_lines = overlap_lines
            current_len = overlap_len
            added_len = line_len if not current_lines else line_len + 1

        current_lines.append(line)
        current_len += added_len

    if current_lines:
        body_str = "\n".join(current_lines)
        chunk_idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{source_filename}::{chunk_idx}",
                source_filename=source_filename,
                header=header,
                body=body_str,
                text=format_chunk_text(source_filename, header, body_str),
            )
        )

    return chunks


def get_embeddings(
    client: httpx.Client,
    texts: list[str],
    base_url: str,
    api_key: str,
    model: str = "jina-embeddings-v3",
    task: str = "",
    batch_size: int = 32,
) -> np.ndarray:
    """Computes L2-normalized embeddings via Jina AI or an OpenAI-compatible /embeddings endpoint."""
    if not texts:
        raise ValueError("Список текстов для эмбеддинга пуст")

    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload: dict = {"model": model, "input": batch}
        if task:
            payload["task"] = task

        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        batch_vectors: list[list[float] | None] = [None] * len(batch)
        for position, item in enumerate(result["data"]):
            index = int(item.get("index", position))
            if index < 0 or index >= len(batch):
                raise ValueError(f"Некорректный индекс embedding: {index}")
            vec = item["embedding"]
            norm = math.sqrt(sum(float(x) ** 2 for x in vec)) or 1.0
            batch_vectors[index] = [float(x) / norm for x in vec]

        if any(v is None for v in batch_vectors):
            raise ValueError("Получены не все embeddings в батче")

        all_vectors.extend(batch_vectors)  # type: ignore[arg-type]

    matrix = np.asarray(all_vectors, dtype=np.float32)
    return matrix


class VectorStore:
    """FAISS-backed vector store (with automatic NumPy fallback if faiss is not installed)."""

    def __init__(
        self,
        chunks: list[Chunk],
        index: object,
        vectors: np.ndarray | None = None,
        meta: dict | None = None,
    ) -> None:
        self.chunks = chunks
        self.index = index
        self.vectors = vectors
        self.meta = meta or {}

    def search(
        self,
        query: str,
        client: httpx.Client,
        base_url: str,
        api_key: str,
        model: str = "jina-embeddings-v3",
        task: str = "retrieval.query",
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Embeds query and returns top_k (Chunk, similarity_score) pairs."""
        if not self.chunks:
            return []

        k = min(top_k, len(self.chunks))
        query_vec = get_embeddings(
            client=client,
            texts=[query],
            base_url=base_url,
            api_key=api_key,
            model=model,
            task=task,
            batch_size=1,
        )

        if faiss is not None and self.index is not None:
            scores, indices = self.index.search(query_vec, k)
            results: list[tuple[Chunk, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.chunks):
                    results.append((self.chunks[int(idx)], float(score)))
            return results

        if self.vectors is None:
            raise RuntimeError("Ни FAISS индекс, ни матрица векторов недоступны")
        sims = (self.vectors @ query_vec[0]).tolist()
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        return [(self.chunks[i], float(sims[i])) for i in ranked]


def _compute_corpus_hash(
    documents: list[tuple[str, str]],
    chunk_size: int,
    overlap: int,
    model: str,
) -> str:
    """Computes a deterministic SHA-256 fingerprint of the documents and chunking config."""
    hasher = hashlib.sha256()
    hasher.update(f"model={model}|chunk_size={chunk_size}|overlap={overlap}\n".encode("utf-8"))
    for name, text in documents:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def create_vectorstore(
    source_path: str | Path,
    store_dir: str | Path,
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str = "jina-embeddings-v3",
    passage_task: str = "retrieval.passage",
    chunk_size: int = 1000,
    overlap: int = 150,
    batch_size: int = 32,
) -> VectorStore:
    """Reads a knowledge base (.zip or folder), chunks every file, embeds chunks, and saves FAISS store."""
    documents = load_documents(source_path)
    all_chunks: list[Chunk] = []
    for filename, text in documents:
        file_chunks = chunk_file(
            source_filename=filename,
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(file_chunks)

    if not all_chunks:
        raise ValueError("После разбиения документов не получено ни одного чанка")

    texts_to_embed = [c.text for c in all_chunks]
    vectors = get_embeddings(
        client=client,
        texts=texts_to_embed,
        base_url=base_url,
        api_key=api_key,
        model=model,
        task=passage_task,
        batch_size=batch_size,
    )

    dim = int(vectors.shape[1])
    store_path = Path(store_dir)
    store_path.mkdir(parents=True, exist_ok=True)

    faiss_index = None
    if faiss is not None:
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(vectors)
        faiss.write_index(faiss_index, str(store_path / "index.faiss"))

    # Always save vectors.npy alongside index.faiss for portability
    np.save(store_path / "vectors.npy", vectors)

    chunks_data = [asdict(c) for c in all_chunks]
    (store_path / "chunks.json").write_text(
        json.dumps(chunks_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta = {
        "model": model,
        "passage_task": passage_task,
        "dimension": dim,
        "num_documents": len(documents),
        "num_chunks": len(all_chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "corpus_hash": _compute_corpus_hash(documents, chunk_size, overlap, model),
    }
    (store_path / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return VectorStore(chunks=all_chunks, index=faiss_index, vectors=vectors, meta=meta)


def load_vectorstore(
    store_dir: str | Path,
    expected_model: str | None = None,
    source_path: str | Path | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> VectorStore | None:
    """Checks for an existing vectorstore in store_dir and loads it if valid.

    Args:
        store_dir: Directory containing chunks.json, meta.json, and index.faiss / vectors.npy.
        expected_model: Optional model name check (returns None if saved index used a different model).
        source_path: Optional path to knowledge base (.zip or folder) to verify corpus hash hasn't changed.
        chunk_size: Optional chunk_size to verify against meta.json.
        overlap: Optional overlap to verify against meta.json.

    Returns:
        Loaded VectorStore instance, or None if not found or incompatible.
    """
    store_path = Path(store_dir)
    chunks_file = store_path / "chunks.json"
    meta_file = store_path / "meta.json"
    faiss_file = store_path / "index.faiss"
    npy_file = store_path / "vectors.npy"

    if not chunks_file.is_file() or not meta_file.is_file():
        return None
    if not faiss_file.is_file() and not npy_file.is_file():
        return None

    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if expected_model and meta.get("model") != expected_model:
            print(
                f"Модель в индексе ({meta.get('model')}) не совпадает с текущей ({expected_model})",
                file=sys.stderr,
            )
            return None

        if chunk_size is not None and meta.get("chunk_size") != chunk_size:
            return None
        if overlap is not None and meta.get("overlap") != overlap:
            return None

        if source_path is not None:
            docs = load_documents(source_path)
            expected_hash = _compute_corpus_hash(
                docs,
                int(meta.get("chunk_size", 1000)),
                int(meta.get("overlap", 150)),
                str(meta.get("model", "")),
            )
            if meta.get("corpus_hash") != expected_hash:
                print("Хэш базы знаний изменился, требуется пересоздание индекса", file=sys.stderr)
                return None

        raw_chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw_chunks]

        faiss_index = None
        if faiss is not None and faiss_file.is_file():
            faiss_index = faiss.read_index(str(faiss_file))

        vectors = np.load(npy_file) if npy_file.is_file() else None
        if faiss_index is None and vectors is None:
            return None

        return VectorStore(chunks=chunks, index=faiss_index, vectors=vectors, meta=meta)
    except Exception as exc:
        print(f"Не удалось загрузить существующий индекс из {store_path}: {exc}", file=sys.stderr)
        return None


def get_or_create_vectorstore(
    source_path: str | Path,
    store_dir: str | Path,
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str = "jina-embeddings-v3",
    passage_task: str = "retrieval.passage",
    chunk_size: int = 1000,
    overlap: int = 150,
    batch_size: int = 32,
) -> VectorStore:
    """Convenience helper: loads existing vectorstore if valid, otherwise builds and saves a new one."""
    existing = load_vectorstore(
        store_dir=store_dir,
        expected_model=model,
        source_path=source_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if existing is not None:
        return existing

    return create_vectorstore(
        source_path=source_path,
        store_dir=store_dir,
        client=client,
        base_url=base_url,
        api_key=api_key,
        model=model,
        passage_task=passage_task,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
    )



