"""Agentic RAG для юридического ассистента ПСБ.

Пайплайн (2 вызова LLM для простых вопросов, до 4 в худшем случае):
    1. QueryPlanner          классификация + rewrite + decompose (1 LLM)
    2. Retrieval             параллельный поиск по подзапросам, дедупликация, соседние чанки,
                             связанные документы: допсоглашения и другие редакции того же документа (без LLM)
    3. CriticAndAnswer       либо финальный ответ, либо SEARCH_MORE (1 LLM)
    4. [Допоиск]             поиск по followup_queries + финальный ответ без права на новый поиск (1 LLM)
    5. SelfCheck             числа, пункты, статьи и номера документов ответа ищутся во фрагментах (без LLM);
                             при расхождении — 1 вызов исправления, затем удаление неподтверждённых предложений

rag.py отвечает за чанки/индекс/поиск, agent.py за LLM-логику, prompts.py за тексты промптов.
"""

from __future__ import annotations

from dotenv import load_dotenv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

from prompts import (ANSWER_FORMAT_ALLOW_SEARCH, ANSWER_FORMAT_FINAL, ANSWER_SYSTEM, NO_ANSWER_MESSAGE,
                     OFF_TOPIC_MESSAGE, PLANNER_SYSTEM, PROMPT_VERSION, REPAIR_INSTRUCTION, STYLE_EXAMPLES)
from rag import Chunk, VectorStore

load_dotenv()
# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #
@dataclass
class AgentConfig:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    emb_base_url: str
    emb_api_key: str
    emb_model: str = "jina-embeddings-v3"
    emb_query_task: str = "retrieval.query"     # Jina: "retrieval.query"; bge-m3 task не требует
    emb_passage_task: str = "retrieval.passage"  # Jina: "retrieval.passage"
    top_k: int = 4                  # чанков на один подзапрос
    max_iterations: int = 2         # сколько раз разрешен допоиск
    max_context_chunks: int = 16    # потолок чанков в промпт ответа
    neighbor_top_n: int = 3         # для скольких лучших чанков подтягиваем ::0 и ::N+1
    search_workers: int = 6         # потоки для эмбеддинг-поиска внутри одного вопроса
    batch_workers: int = 8          # параллельных вопросов в answer_many
    json_mode: bool = True          # response_format=json_object (авто-откат, если шлюз не поддерживает)
    llm_timeout: float = 120.0
    # Реранкер опционален: если reranker_model пуст или reranker_top_n == 0, не используется.
    reranker_base_url: str = ""
    reranker_api_key: str = ""
    reranker_model: str = ""
    reranker_top_n: int = 0

    @classmethod
    def from_env(cls) -> "AgentConfig":
        # Имена переменных совпадают с test.py / .env: EMBEDDER_* и RERANKER_*.
        emb_base_url = (
            os.environ.get("EMB_BASE_URL")
            or os.environ.get("EMBEDDER_BASE_URL")
            or os.environ["LLM_BASE_URL"]
        )
        emb_api_key = (
            os.environ.get("EMB_API_KEY")
            or os.environ.get("EMBEDDER_API_KEY")
            or os.environ["LLM_API_KEY"]
        )
        return cls(
            llm_base_url=os.environ["LLM_BASE_URL"],
            llm_api_key=os.environ["LLM_API_KEY"],
            llm_model=os.environ["LLM_MODEL"],
            emb_base_url=emb_base_url,
            emb_api_key=emb_api_key,
            emb_model=os.environ.get("EMB_MODEL", "baai/bge-m3"),
            emb_query_task=os.environ.get("EMB_TASK_QUERY", os.environ.get("EMB_QUERY_TASK", "")),
            emb_passage_task=os.environ.get("EMB_TASK_PASSAGE", ""),
            reranker_base_url=os.environ.get("RERANKER_BASE_URL", emb_base_url),
            reranker_api_key=os.environ.get("RERANKER_API_KEY", emb_api_key),
            reranker_model=os.environ.get("RERANKER_MODEL", ""),
            reranker_top_n=int(os.environ.get("RERANKER_TOP_N", "0") or "0"),
        )


# --------------------------------------------------------------------------- #
# LLM-клиент с ретраями и устойчивым разбором JSON
# --------------------------------------------------------------------------- #
def parse_json(text: str) -> dict | None:
    """Достает JSON-объект из ответа модели (с ```json-оберткой или текстом вокруг)."""
    if not text:
        return None
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


class LLM:
    def __init__(self, cfg: AgentConfig, http: httpx.Client) -> None:
        self.cfg = cfg
        self.http = http
        self._json_ok = cfg.json_mode
        self.calls = 0

    def chat(self, messages: list[dict], json_mode: bool = True,
             temperature: float = 0.0, max_tokens: int = 4096, retries: int = 3) -> str:
        url = self.cfg.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg.llm_api_key}"}
        last_err: Exception | None = None
        for attempt in range(retries):
            payload: dict = {
                "model": self.cfg.llm_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode and self._json_ok:
                payload["response_format"] = {"type": "json_object"}
            try:
                resp = self.http.post(url, headers=headers, json=payload, timeout=self.cfg.llm_timeout)
                if resp.status_code == 400 and "response_format" in payload:
                    self._json_ok = False       # шлюз не умеет json_object, дальше парсим сами
                    continue
                resp.raise_for_status()
                self.calls += 1
                return resp.json()["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM недоступна после {retries} попыток: {last_err}")


# --------------------------------------------------------------------------- #
# Самопроверка фактов (без LLM)
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)*")
_DOCNUM_RE = re.compile(r"(?<![А-ЯЁA-Zа-яё])([А-ЯЁA-Z]{1,3}-\d{1,4}(?:/\d{4})?)(?![\d/])")
_ARTICLE_RE = re.compile(r"(?:\bст\.|\bстать[яиейю]\w*)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"(?:\bпп?\.|\bпункт\w*)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)


def _norm(text: str) -> str:
    return (text or "").replace("\u00a0", " ").replace("\u202f", " ").replace("ё", "е").replace("Ё", "Е")


def _numbers(text: str) -> set[str]:
    out = set()
    for tok in _NUM_RE.findall(_norm(text)):
        tok = re.sub(r"[ ]", "", tok).replace(",", ".")
        out.add(tok)
        out.update(p for p in tok.split(".") if p)     # «3.1» подтверждает и «3», «01.03.2025» — части даты
    return out


class Evidence:
    """Что считается подтверждённым: тексты фрагментов, которые видела модель, + реплики пользователя."""

    def __init__(self, chunks: list[Chunk], user_text: str) -> None:
        self.text = _norm("\n".join(c.text for c in chunks))
        self.low = self.text.lower()
        self.numbers = _numbers(self.text) | _numbers(user_text)
        self.user_low = _norm(user_text).lower()
        self.docnums = {m.upper() for m in _DOCNUM_RE.findall(self.text + "\n" + _norm(user_text))}

    def has_article(self, num: str) -> bool:
        pat = r"(?:стать[яи]\s*|ст\.\s*)" + re.escape(num) + r"(?![\d.])"
        return re.search(pat, self.low) is not None or re.search(pat, self.user_low) is not None

    def has_clause(self, num: str) -> bool:
        esc = re.escape(num)
        return (re.search(r"(?m)^" + esc + r"\.?\s", self.text) is not None              # строка «3.1. …» / «3. …»
                or re.search(r"(?:\bпп?\.\s*|пункт\w*\s+)" + esc + r"(?![\d.])", self.low) is not None
                or re.search(r"(?:\bпп?\.\s*|пункт\w*\s+)" + esc + r"(?![\d.])", self.user_low) is not None)


def find_unsupported(answer: str, ev: Evidence) -> list[str]:
    """Факты ответа, которых нет во фрагментах. Пустой список — всё подтверждено."""
    text = _norm(answer)
    issues: list[str] = []
    for num in dict.fromkeys(_ARTICLE_RE.findall(text)):
        if not ev.has_article(num):
            issues.append(f"статья {num}")
    for num in dict.fromkeys(_CLAUSE_RE.findall(text)):
        if not ev.has_clause(num):
            issues.append(f"пункт {num}")
    for dn in dict.fromkeys(m.upper() for m in _DOCNUM_RE.findall(text)):
        if dn not in ev.docnums:
            issues.append(f"номер документа {dn}")
    # остальные числа: убираем уже проверенные ссылки, чтобы не дублировать
    rest = _ARTICLE_RE.sub(" ", _CLAUSE_RE.sub(" ", _DOCNUM_RE.sub(" ", text)))
    for tok in dict.fromkeys(_NUM_RE.findall(rest)):
        norm = re.sub(r"[ \u00a0\u202f]", "", tok).replace(",", ".")
        if norm not in ev.numbers:
            issues.append(f"число {tok.strip()}")
    return issues


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[«\"(]?[А-ЯЁA-Z])")
_ABBR_END = re.compile(r"(?:\b(?:п|пп|ст|ч|г|гг|руб|тыс|млн|млрд|им|т\.е|т\.к|др|см|ред|исх|№)\.|\b[А-ЯЁ]\.)$", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for piece in _SENT_SPLIT.split(text.strip()):
        if parts and _ABBR_END.search(parts[-1]):      # «… п. 3.1», «… ст. 70», «А. Б. Иванов» — не конец фразы
            parts[-1] += " " + piece
        elif piece:
            parts.append(piece)
    return parts


def drop_unsupported_sentences(answer: str, ev: Evidence) -> str:
    """Удаляет предложения с неподтверждёнными фактами. Если не осталось почти ничего — возвращает ""."""
    kept = [s for s in split_sentences(answer) if not find_unsupported(s, ev)]
    out = " ".join(kept).strip()
    return out if len(out) >= 40 else ""


def clean_answer_text(text: str) -> str:
    """Убирает markdown, технические id фрагментов и лишние пробелы."""
    text = re.sub(r"\[[^\]]*::\d+[^\]]*\]", "", text or "")          # [file.txt::3]
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"(?m)^\s*#+\s*", "", text)
    text = re.sub(r"(?m)^\s*[-•*]\s+", "", text)
    text = re.sub(r"\[([^\]]{3,120})\]", r"(\1)", text)                # [документ, п. 3.1] → (документ, п. 3.1)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text


# --------------------------------------------------------------------------- #
# Структуры данных
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    is_legal_domain: bool = True
    refusal_message: str = ""
    event_date: str | None = None
    sub_queries: list[str] = field(default_factory=list)
    llm_calls: int = 0              # сколько LLM-вызовов сделал планировщик


@dataclass
class AgentResult:
    answer: str
    status: str                                  # ANSWERED | PARTIAL | REFUSED | CLARIFY | OFF_TOPIC
    sources: list[dict] = field(default_factory=list)
    trace: dict = field(default_factory=dict)


class ChunkPool:
    """Пул найденных чанков: дедупликация по chunk_id, храним максимальный скор."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, float]] = {}

    def add(self, results: list[tuple[Chunk, float]]) -> int:
        new = 0
        for chunk, score in results:
            prev = self._items.get(chunk.chunk_id)
            if prev is None:
                new += 1
                self._items[chunk.chunk_id] = (chunk, score)
            elif score > prev[1]:
                self._items[chunk.chunk_id] = (chunk, score)
        return new

    def apply_order(self, ordered_ids: list[str]) -> None:
        """Поднимает указанные чанки наверх в заданном порядке (результат реранкера)."""
        max_score = max((s for _, s in self._items.values()), default=0.0)
        for rank, chunk_id in enumerate(ordered_ids):
            if chunk_id in self._items:
                chunk, _ = self._items[chunk_id]
                self._items[chunk_id] = (chunk, max_score + 1.0 - rank * 1e-3)

    def ranked(self, limit: int | None = None) -> list[tuple[Chunk, float]]:
        items = sorted(self._items.values(), key=lambda x: x[1], reverse=True)
        return items[:limit] if limit else items

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._items

    def __len__(self) -> int:
        return len(self._items)


# --------------------------------------------------------------------------- #
# Агент
# --------------------------------------------------------------------------- #
class LegalAgent:
    def __init__(self, vectorstore: VectorStore, cfg: AgentConfig, http: httpx.Client | None = None) -> None:
        self.vs = vectorstore
        self.cfg = cfg
        self.http = http or httpx.Client(timeout=cfg.llm_timeout)
        self.llm = LLM(cfg, self.http)
        self.by_id = {c.chunk_id: c for c in vectorstore.chunks}
        # Два РАЗНЫХ пула: во вложенном использовании одного пула легко получить дедлок.
        self._search_pool = ThreadPoolExecutor(max_workers=cfg.search_workers)
        self._batch_pool = ThreadPoolExecutor(max_workers=cfg.batch_workers)

    # ---- публичный API ---------------------------------------------------- #
    def answer(self, question: str | None = None, history: list[dict] | None = None) -> AgentResult:
        """Одиночный вопрос или история диалога [{'role': 'user'|'assistant', 'content': '...'}]."""
        t0 = time.perf_counter()
        messages = self._sanitize_history(history, question)
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        trace: dict = {"llm_calls": 0, "timings": {}, "prompt_version": PROMPT_VERSION}

        # Спекулятивный параллелизм: пока планировщик думает, уже ищем по сырому вопросу.
        plan_f = self._batch_pool.submit(self._plan, messages)
        raw_f = self._search_pool.submit(self._search_one, last_user)
        plan = plan_f.result()
        trace["plan"] = plan.__dict__
        trace["llm_calls"] = plan.llm_calls
        trace["timings"]["plan"] = round(time.perf_counter() - t0, 2)

        if not plan.is_legal_domain:
            raw_f.cancel()
            msg = clean_answer_text(plan.refusal_message) or OFF_TOPIC_MESSAGE
            return AgentResult(answer=msg, status="OFF_TOPIC", trace=trace)

        # --- Retrieval -----------------------------------------------------
        t1 = time.perf_counter()
        pool = ChunkPool()
        queries = [q for q in plan.sub_queries if q and q.strip() != last_user][:3]
        for res in self._search_many(queries):
            pool.add(res)
        try:
            pool.add(raw_f.result())
        except Exception as exc:  # noqa: BLE001, спекулятивный поиск не должен ронять запрос
            trace["raw_search_error"] = str(exc)
        self._add_neighbors(pool)
        self._rerank(last_user, pool)
        self._add_related(pool)
        trace["queries"] = queries
        trace["timings"]["retrieve"] = round(time.perf_counter() - t1, 2)

        # --- Critic + Answer ----------------------------------------------
        t2 = time.perf_counter()
        allow_more = self.cfg.max_iterations > 0
        answer_calls = [0]     # счетчик LLM-вызовов именно этого вопроса (потокобезопасно)
        out = self._critic_answer(messages, pool, plan, allow_more, counter=answer_calls)

        if out.get("action") == "SEARCH_MORE" and allow_more:
            followups = [q for q in out.get("followup_queries", []) if isinstance(q, str)][:3]
            trace["followup_queries"] = followups
            trace["missing_info"] = out.get("missing_info", "")
            for res in self._search_many(followups):
                pool.add(res)
            self._add_neighbors(pool)
            self._rerank(last_user, pool)
            self._add_related(pool)
            out = self._critic_answer(messages, pool, plan, allow_more=False, counter=answer_calls)
        trace["timings"]["answer"] = round(time.perf_counter() - t2, 2)

        # --- Самопроверка фактов -------------------------------------------
        answer = clean_answer_text(str(out.get("answer") or ""))
        if self.cfg.self_check and answer:
            t3 = time.perf_counter()
            user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
            ev = Evidence([c for c, _ in pool.ranked(self.cfg.max_context_chunks)], user_text)
            issues = find_unsupported(answer, ev)
            trace["selfcheck_issues"] = issues
            if issues:
                fixed = self._repair(messages, pool, plan, answer, issues, counter=answer_calls)
                if fixed is not None:
                    if not fixed.get("used_chunk_ids"):
                        fixed["used_chunk_ids"] = out.get("used_chunk_ids", [])
                    out = fixed
                    answer = clean_answer_text(str(out.get("answer") or ""))
                left = find_unsupported(answer, ev)
                trace["selfcheck_after_repair"] = left
                if left:
                    trimmed = drop_unsupported_sentences(answer, ev)
                    if trimmed:
                        answer = trimmed
                        trace["selfcheck_trimmed"] = True
            trace["timings"]["selfcheck"] = round(time.perf_counter() - t3, 2)

        # --- Сборка результата --------------------------------------------
        answer = answer or NO_ANSWER_MESSAGE
        status = out.get("status") if out.get("status") in {"ANSWERED", "PARTIAL", "REFUSED", "CLARIFY"} else "PARTIAL"
        trace["question_type"] = out.get("question_type", "")
        used = [cid for cid in out.get("used_chunk_ids", []) if isinstance(cid, str) and cid in pool]
        trace["pool_size"] = len(pool)
        if status == "ANSWERED" and not used:
            trace["warning"] = "ответ без валидных ссылок на чанки"
        sources = [
            {"chunk_id": c.chunk_id, "filename": c.source_filename, "header": c.header,
             "text": c.body, "score": round(s, 4), "locator": getattr(c, "locator", ""),
             "doc_title": getattr(c, "doc_title", "")}
            for c, s in pool.ranked() if c.chunk_id in used
        ]
        trace["timings"]["total"] = round(time.perf_counter() - t0, 2)
        trace["llm_calls"] = plan.llm_calls + answer_calls[0]
        return AgentResult(answer=answer, status=status, sources=sources, trace=trace)

    def answer_many(self, items: list[dict]) -> list[AgentResult]:
        """Батч независимых тест-кейсов: [{'question': ...} | {'history': [...]}, ...].
        Именно здесь параллелизм дает основной выигрыш по времени сабмита."""
        def run(item: dict) -> AgentResult:
            try:
                return self.answer(question=item.get("question"), history=item.get("history"))
            except Exception as exc:  # noqa: BLE001
                return AgentResult(answer="Внутренняя ошибка обработки запроса.", status="REFUSED",
                                   trace={"error": str(exc)})
        with ThreadPoolExecutor(max_workers=self.cfg.batch_workers) as ex:
            return list(ex.map(run, items))

    # ---- шаг 1: планировщик ---------------------------------------------- #
    def _plan(self, messages: list[dict]) -> Plan:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        try:
            raw = self.llm.chat(
                [{"role": "system", "content": PLANNER_SYSTEM},
                 {"role": "user", "content": f"Диалог:\n{transcript}\n\nВерни JSON."}],
                max_tokens=800,
            )
            llm_calls = 1
            data = parse_json(raw) or {}
        except Exception:  # noqa: BLE001
            llm_calls = 0
            data = {}
        queries = [q.strip() for q in data.get("sub_queries", []) if isinstance(q, str) and q.strip()]
        return Plan(
            is_legal_domain=bool(data.get("is_legal_domain", True)),   # при сбое, не отказываем
            refusal_message=str(data.get("refusal_message") or ""),
            event_date=data.get("event_date") if isinstance(data.get("event_date"), str) else None,
            sub_queries=queries or [last_user],
            llm_calls=llm_calls,
        )

    # ---- шаг 2: поиск ------------------------------------------------------ #
    def _search_one(self, query: str) -> list[tuple[Chunk, float]]:
        if not query.strip():
            return []
        return self.vs.search(
            query, client=self.http, base_url=self.cfg.emb_base_url, api_key=self.cfg.emb_api_key,
            model=self.cfg.emb_model, task=self.cfg.emb_query_task, top_k=self.cfg.top_k,
        )

    def _search_many(self, queries: list[str]) -> list[list[tuple[Chunk, float]]]:
        futures = [self._search_pool.submit(self._search_one, q) for q in queries]
        out = []
        for f in futures:
            try:
                out.append(f.result())
            except Exception as exc:  # noqa: BLE001
                print(f"Ошибка поиска: {exc}", file=sys.stderr)
        return out

    def _add_neighbors(self, pool: ChunkPool) -> None:
        """Для лучших чанков подтягиваем первый чанк документа (шапка: стороны, даты, статус) и следующий."""
        extra: list[tuple[Chunk, float]] = []
        for chunk, score in pool.ranked(self.cfg.neighbor_top_n):
            doc, _, idx = chunk.chunk_id.rpartition("::")
            if not idx.isdigit():
                continue
            for neighbor_id in (f"{doc}::0", f"{doc}::{int(idx) + 1}"):
                neighbor = self.by_id.get(neighbor_id)
                if neighbor is not None and neighbor_id not in pool:
                    extra.append((neighbor, score * 0.5))   # ниже родителя, чтобы не вытеснять прямые попадания
        pool.add(extra)

    def _add_related(self, pool: ChunkPool) -> None:
        """Связанные документы компании для лучших чанков (без LLM):
        договор ↔ его допсоглашения; пункт документа ↔ тот же пункт другой редакции этого документа.
        Нужны, чтобы модель могла выбрать редакцию на дату события."""
        norm_num = lambda s: re.sub(r"[\s№]", "", s or "").upper()
        norm_title = lambda s: re.sub(r"\s+", " ", (s or "").upper()).strip()
        extra: list[tuple[Chunk, float]] = []
        for chunk, score in pool.ranked(self.cfg.neighbor_top_n):
            if getattr(chunk, "doc_type", "") != "company":
                continue
            num, title = norm_num(getattr(chunk, "doc_number", "")), norm_title(getattr(chunk, "doc_title", ""))
            clauses = set(getattr(chunk, "clauses", []) or [])
            for other in self.vs.chunks:
                if other.source_filename == chunk.source_filename or other.chunk_id in pool:
                    continue
                if getattr(other, "doc_type", "") != "company":
                    continue
                # 1) допсоглашение к этому договору / договор, который меняет это допсоглашение
                if num and norm_num(getattr(other, "amends", "")) == num:
                    extra.append((other, score * 0.6))
                elif getattr(chunk, "amends", "") and norm_num(other.doc_number) == norm_num(chunk.amends):
                    target = set(getattr(chunk, "amended_clauses", []) or [])
                    if other.chunk_id.endswith("::0") or target & set(other.clauses or []):
                        extra.append((other, score * 0.6))
                # 2) другая редакция того же документа (одинаковое название): тот же пункт + шапка со статусом
                elif title and norm_title(other.doc_title) == title:
                    if other.chunk_id.endswith("::0") or (clauses and clauses & set(other.clauses or [])):
                        extra.append((other, score * 0.6))
        pool.add(extra)

    def _repair(self, messages: list[dict], pool: ChunkPool, plan: Plan, draft: str, issues: list[str],
                counter: list[int] | None = None) -> dict | None:
        """Один вызов LLM: переписать ответ без неподтверждённых фактов."""
        docs = self._docs_block(pool)
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        prompt = (f"Фрагменты базы знаний:\n{docs}\n\nДиалог:\n{transcript}\n\n"
                  + REPAIR_INSTRUCTION.format(issues="\n".join(f"- {i}" for i in issues), draft=draft))
        try:
            raw = self.llm.chat([{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": prompt}],
                                max_tokens=self.cfg.answer_max_tokens, retries=2)
        except Exception:  # noqa: BLE001
            return None
        if counter is not None:
            counter[0] += 1
        data = parse_json(raw)
        if data and str(data.get("answer") or "").strip():
            return data
        return None

    def _docs_block(self, pool: ChunkPool) -> str:
        return "\n\n".join(
            f'<document id="{c.chunk_id}">\n{c.text}\n</document>'
            for c, _ in pool.ranked(self.cfg.max_context_chunks)
        ) or "(ничего не найдено)"

    def _rerank(self, query: str, pool: ChunkPool) -> None:
        """Переранжирует лучшие кандидаты внешним реранкером (no-op при сбое/выключении)."""
        if not self.cfg.reranker_model or self.cfg.reranker_top_n <= 0:
            return
        candidates = pool.ranked(self.cfg.max_context_chunks * 2)
        if not candidates:
            return
        payload = {
            "model": self.cfg.reranker_model,
            "query": query,
            "documents": [c.text for c, _ in candidates],
            "top_n": min(self.cfg.reranker_top_n, len(candidates)),
        }
        try:
            resp = self.http.post(
                self.cfg.reranker_base_url.rstrip("/") + "/rerank",
                headers={"Authorization": f"Bearer {self.cfg.reranker_api_key}"},
                json=payload,
                timeout=min(self.cfg.llm_timeout, 15.0)
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            print(f"Reranker недоступен: {exc}; используем текущий порядок", file=sys.stderr)
            return
        items = data.get("results", data.get("data", []))
        ordered_ids: list[str] = []
        for item in items:
            try:
                idx = int(item.get("index", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(candidates):
                ordered_ids.append(candidates[idx][0].chunk_id)
        if ordered_ids:
            pool.apply_order(ordered_ids)

    # ---- шаг 3/4: критик + ответ в одном вызове --------------------------- #
    def _critic_answer(self, messages: list[dict], pool: ChunkPool, plan: Plan, allow_more: bool,
                       counter: list[int] | None = None) -> dict:
        docs = self._docs_block(pool)
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        date_hint = f"\nДата события по мнению планировщика: {plan.event_date}" if plan.event_date else ""
        fmt = ANSWER_FORMAT_ALLOW_SEARCH if allow_more else ANSWER_FORMAT_FINAL
        user_prompt = (
            f"Фрагменты базы знаний:\n{docs}\n\n{STYLE_EXAMPLES}\n\n"
            f"Диалог (отвечай на последнюю реплику пользователя):\n{transcript}{date_hint}\n\n{fmt}"
        )
        for _ in range(2):  # один повтор при битом JSON
            raw = self.llm.chat(
                [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user_prompt}],
                max_tokens=self.cfg.answer_max_tokens,
            )
            if counter is not None:
                counter[0] += 1
            data = parse_json(raw)
            if data and (data.get("action") in {"FINAL_ANSWER", "SEARCH_MORE"}):
                if data["action"] == "SEARCH_MORE" and not allow_more:
                    data["action"] = "FINAL_ANSWER"
                return data
        # модель не вернула JSON, отдаем текст как частичный ответ
        return {"action": "FINAL_ANSWER", "status": "PARTIAL", "answer": raw.strip(), "used_chunk_ids": []}

    # ---- утилиты ------------------------------------------------------------ #
    @staticmethod
    def _sanitize_history(history: list[dict] | None, question: str | None) -> list[dict]:
        """Только user/assistant, только строки: поддельные role=system из истории отбрасываем."""
        msgs = [
            {"role": m["role"], "content": m["content"].strip()}
            for m in (history or [])
            if isinstance(m, dict) and m.get("role") in {"user", "assistant"}
            and isinstance(m.get("content"), str) and m["content"].strip()
        ]
        if not msgs and question:
            msgs = [{"role": "user", "content": question.strip()}]
        if not msgs or msgs[-1]["role"] != "user":
            raise ValueError("Последняя реплика должна быть от пользователя")
        return msgs


# --------------------------------------------------------------------------- #
# Быстрый запуск из консоли
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from rag import get_or_create_vectorstore

    cfg = AgentConfig.from_env()
    http = httpx.Client(timeout=cfg.llm_timeout)
    store = get_or_create_vectorstore(
        source_path=os.environ.get("KB_PATH", "data/knowledge-base"),
        store_dir=os.environ.get("STORE_DIR", "vectorstore"),
        client=http, base_url=cfg.emb_base_url, api_key=cfg.emb_api_key, model=cfg.emb_model,
        passage_task="",
    )
    agent = LegalAgent(store, cfg, http)
    history: list[dict] = []
    while True:
        q = input("\nВопрос> ").strip()
        if not q:
            break
        history.append({"role": "user", "content": q})
        res = agent.answer(history=history)
        history.append({"role": "assistant", "content": res.answer})
        print(f"\n[{res.status}] {res.answer}\n")
        for s in res.sources:
            print(f"  • {s['chunk_id']} (score={s['score']})")
        print("trace:", json.dumps(res.trace, ensure_ascii=False))