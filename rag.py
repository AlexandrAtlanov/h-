"""Поиск по базе знаний (Роль 1): загрузка документов, разбиение, метаданные, индекс, гибридный поиск.

Публичный API (его использует agent.py, solution.py и веб-чат — менять только по согласию команды):
    Chunk                       — фрагмент с текстом и метаданными
    VectorStore.search(...)     — гибридный поиск BM25 + bge-m3 (RRF) с бустом по номеру договора/контрагенту
    load_documents(path)        — .zip или папка с .txt/.md/.docx
    build_store(...)            — разбивает документы, считает эмбеддинги, возвращает VectorStore
    load_or_build_store(...)    — берёт готовый индекс из kb_index_data.py (если хэш базы совпал), иначе строит
    get_or_create_vectorstore() — то же с кэшем в папке (для веб-чата и локальных скриптов)

Индекс ничего не знает про LLM: только поиск. Если эмбеддер недоступен, поиск деградирует до BM25.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import sys
import threading
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
import numpy as np

# Меняйте при любом изменении разбиения/метаданных: старый готовый индекс станет недействительным.
CHUNKER_VERSION = "v2-clauses"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx"}

LAW_CHUNK_CHARS = 900       # целевой размер фрагмента закона (без шапки)
LAW_OVERLAP_CHARS = 150
COMPANY_SECTION_MAX = 1400  # раздел договора длиннее этого режется по пунктам


# --------------------------------------------------------------------------- #
# Данные
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """Фрагмент документа. Поля до `text` — исторический контракт с agent.py, не переименовывать."""

    chunk_id: str               # "<имя файла>::<номер>", ::0 — шапка документа
    source_filename: str
    header: str                 # шапка документа (название, стороны, дата, статус)
    body: str                   # текст фрагмента без шапки
    text: str                   # то, что видит LLM и эмбеддер: карточка документа + body
    # --- метаданные ---
    doc_type: str = ""          # "law" | "company"
    doc_title: str = ""         # человекочитаемое название документа
    doc_number: str = ""        # "П-17/2024", "К-12", "ЮО-7"...
    doc_date: str = ""          # дата документа ГГГГ-ММ-ДД
    counterparties: list[str] = field(default_factory=list)  # «Альфа-Трейд»...
    edition: str = ""           # редакция ГГГГ-ММ-ДД (для регламентов)
    valid_from: str = ""        # действует с ГГГГ-ММ-ДД
    valid_to: str = ""          # утратил силу с ГГГГ-ММ-ДД
    status: str = ""            # "действует" | "утратил силу"
    amends: str = ""            # для допсоглашений: номер изменяемого договора
    amended_clauses: list[str] = field(default_factory=list)   # какие пункты меняет допсоглашение
    amended_by: list[str] = field(default_factory=list)        # заметки «п. 3.2 изменён ДС № 1 ...»
    clauses: list[str] = field(default_factory=list)           # номера пунктов внутри фрагмента
    locator: str = ""           # «п. 3.1–3.5», «ст. 70» — для ссылок и подсветки в чате


# --------------------------------------------------------------------------- #
# Загрузка
# --------------------------------------------------------------------------- #
def read_docx(raw: bytes) -> str:
    """Текст из .docx без внешних зависимостей."""
    with zipfile.ZipFile(io.BytesIO(raw)) as doc:
        xml = doc.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join("".join(x.text or "" for x in p.iter(ns + "t")) for p in root.iter(ns + "p"))


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ").replace("​", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _decode(name: str, raw: bytes) -> str | None:
    suffix = Path(name).suffix.lower()
    try:
        if suffix == ".docx":
            return read_docx(raw)
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return raw.decode("cp1251")
    except (UnicodeDecodeError, KeyError, ValueError, zipfile.BadZipFile, ET.ParseError):
        print(f"Пропущен нечитаемый документ: {name}", file=sys.stderr)
        return None


def _fix_zip_name(info: zipfile.ZipInfo) -> str:
    """Кириллица в ZIP без флага UTF-8 (архивы из Windows) приходит как cp437."""
    name = info.filename
    if info.flag_bits & 0x800:
        return name
    try:
        return name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            return name.encode("cp437").decode("cp866")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return name


def load_documents(source_path: str | Path) -> list[tuple[str, str]]:
    """[(имя файла без папок, нормализованный текст)] из .zip или папки, отсортировано по имени."""
    path = Path(source_path)
    documents: list[tuple[str, str]] = []
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                name = _fix_zip_name(info)
                if info.is_dir() or name.startswith("__MACOSX/") or Path(name).name.startswith("."):
                    continue
                if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                text = _decode(name, archive.read(info))
                if text and normalize_text(text):
                    documents.append((Path(name).name, normalize_text(text)))
    elif path.is_dir():
        for file_path in path.rglob("*"):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            text = _decode(file_path.name, file_path.read_bytes())
            if text and normalize_text(text):
                documents.append((file_path.name, normalize_text(text)))
    else:
        raise FileNotFoundError(f"База знаний не найдена: {path}")
    if not documents:
        raise ValueError(f"В {path} нет читаемых документов")
    documents.sort(key=lambda d: d[0])
    return documents


# --------------------------------------------------------------------------- #
# Метаданные
# --------------------------------------------------------------------------- #
MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7,
    "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
DATE_NUM_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
DATE_TXT_RE = re.compile(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})", re.IGNORECASE)
DOC_NUMBER_RE = re.compile(r"№\s*([A-Za-zА-Яа-яЁё]{1,4}-?\d+(?:/\d{2,4})?(?:-[А-ЯA-Z])?)")
QUOTED_ORG_RE = re.compile(r"(?:ООО|АО|ПАО|ЗАО|ИП)\s*«([^»]+)»")
CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s")      # 3.1. / 3.1.2.
SECTION_RE = re.compile(r"^(\d+)\.\s+\S")              # 3. Арендная плата
OWN_COMPANY = {"меридиан"}                              # наша сторона, не «контрагент»


def _iso(d: int, m: int, y: int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"


def find_dates(text: str) -> list[tuple[int, str]]:
    """[(позиция, ГГГГ-ММ-ДД)] в порядке появления."""
    found = [(m.start(), _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))) for m in DATE_NUM_RE.finditer(text)]
    found += [(m.start(), _iso(int(m.group(1)), MONTHS[m.group(2).lower()], int(m.group(3))))
              for m in DATE_TXT_RE.finditer(text)]
    return sorted(found)


def _ru_date(iso: str) -> str:
    return f"{iso[8:10]}.{iso[5:7]}.{iso[0:4]}" if len(iso) == 10 else iso


def detect_doc_type(filename: str, text: str) -> str:
    low = filename.lower()
    if low.startswith("законы") or text.startswith("Источник:"):
        return "law"
    return "company"


def extract_metadata(filename: str, lines: list[str]) -> dict:
    """Метаданные документа по шапке. Правила общие, без привязки к конкретным файлам."""
    doc_type = detect_doc_type(filename, "\n".join(lines[:1]))
    meta: dict = {"doc_type": doc_type, "counterparties": [], "amended_clauses": []}
    head = "\n".join(lines[:6])
    full = "\n".join(lines)

    if doc_type == "law":
        src = next((l for l in lines[:3] if l.startswith("Источник:")), lines[0])
        title = src.removeprefix("Источник:").strip()
        title = re.sub(r"\s*\(с изменениями и дополнениями\)\s*$", "", title)
        meta["doc_title"] = title
        art = re.search(r"Статья\s+(\d+(?:\.\d+)?)", title)
        code = next((c for c in ("НК РФ", "ТК РФ", "КоАП РФ", "ГК РФ") if c in title), "")
        if not code:
            fz = re.search(r"N\s*(\d+-ФЗ)", title)
            code = fz.group(1) if fz else ""
        meta["doc_number"] = f"ст. {art.group(1)} {code}".strip() if art else code
        meta["status"] = "действующая редакция"
        return meta

    # --- документы компании ---
    title_lines = [l for l in lines[:4] if not re.match(r"^РЕДАКЦИЯ УТРАТИЛА СИЛУ", l, re.I)]
    title = title_lines[0] if title_lines else lines[0]
    # письма/записки: «ООО «МЕРИДИАН» / Служба / Исх. № ...» — берём строку с типом документа
    for l in lines[:6]:
        if re.match(r"^(СЛУЖЕБНАЯ ЗАПИСКА|ПРИКАЗ|ПИСЬМО|РЕГЛАМЕНТ|ДОГОВОР|ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ|ПОЛОЖЕНИЕ)", l, re.I):
            title = l
            break
    else:
        if re.match(r"^ООО\s*«", title) and len(lines) > 2:
            subj = next((l for l in lines[2:7] if re.match(r"^О\s", l)), "")
            title = " ".join(lines[1:3]) + (f" «{subj}»" if subj else "")
    meta["doc_title"] = title

    num = DOC_NUMBER_RE.search(title) or DOC_NUMBER_RE.search(head)
    meta["doc_number"] = num.group(1) if num else ""

    dates = find_dates(head)
    m_ed = re.search(r"Редакция от (\d{1,2}\.\d{1,2}\.\d{4})", head)
    if m_ed:
        meta["edition"] = find_dates(m_ed.group(1))[0][1]
    m_from = re.search(r"Действует с (\d{1,2}\.\d{1,2}\.\d{4})", head)
    if m_from:
        meta["valid_from"] = find_dates(m_from.group(1))[0][1]
    m_to = re.search(r"УТРАТИЛ[АО]? СИЛУ с (\d{1,2}\.\d{1,2}\.\d{4})", head, re.I)
    if m_to or "утратил" in filename.lower():
        meta["status"] = "утратил силу"
        if m_to:
            meta["valid_to"] = find_dates(m_to.group(1))[0][1]
    else:
        meta["status"] = "действует"
    if dates:
        meta["doc_date"] = meta.get("edition") or dates[0][1]

    orgs = []
    for name in QUOTED_ORG_RE.findall(full[:1500]):
        if name.strip().lower() not in OWN_COMPANY and name not in orgs:
            orgs.append(name.strip())
    meta["counterparties"] = orgs

    # допсоглашение: «к договору поставки № П-17/2024», «Изложить пункт 3.2 ...»
    if re.search(r"ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ", head, re.I):
        target = re.search(r"к договору[^№]{0,40}№\s*([A-Za-zА-Яа-яЁё]{1,4}-?\d+/\d{2,4})", head, re.I)
        if target:
            meta["amends"] = target.group(1)
        meta["amended_clauses"] = sorted(set(re.findall(r"пункт[а-я]*\s+(\d+(?:\.\d+)+)", full, re.I)))
        dnum = re.match(r"ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ\s*№\s*(\S+)", title, re.I)
        meta["doc_number"] = f"ДС № {dnum.group(1)}" if dnum else "ДС"
        # дата допсоглашения — последняя дата в шапке (первая — дата основного договора)
        if dates:
            meta["doc_date"] = dates[-1][1]
    return meta


def doc_card(meta: dict, filename: str) -> str:
    """Короткая «карточка» документа, которая ставится перед каждым фрагментом."""
    parts = [f"Документ: {meta.get('doc_title') or filename}"]
    if meta.get("doc_type") == "law":
        parts.append("Тип: закон (действующая редакция; старых редакций в базе нет)")
    else:
        bits = []
        if meta.get("doc_number"):
            bits.append(f"№ {meta['doc_number']}" if not meta["doc_number"].startswith("ДС") else meta["doc_number"])
        if meta.get("edition"):
            bits.append(f"редакция от {_ru_date(meta['edition'])}")
        elif meta.get("doc_date"):
            bits.append(f"дата {_ru_date(meta['doc_date'])}")
        if meta.get("counterparties"):
            bits.append("контрагент: " + ", ".join(f"ООО «{c}»" for c in meta["counterparties"]))
        if bits:
            parts.append("Реквизиты: " + "; ".join(bits))
        status = meta.get("status", "")
        if status == "утратил силу":
            when = f" с {_ru_date(meta['valid_to'])}" if meta.get("valid_to") else ""
            parts.append(f"Статус: РЕДАКЦИЯ УТРАТИЛА СИЛУ{when}; применяется только к событиям до этой даты")
        elif meta.get("valid_from"):
            parts.append(f"Статус: действует с {_ru_date(meta['valid_from'])}")
        if meta.get("amends"):
            cl = ", ".join(f"п. {c}" for c in meta.get("amended_clauses", []))
            parts.append(f"Изменяет договор № {meta['amends']}" + (f" ({cl})" if cl else ""))
    parts.append(f"Файл: {filename}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Разбиение
# --------------------------------------------------------------------------- #
def _split_long(line: str, limit: int) -> list[str]:
    if len(line) <= limit:
        return [line]
    out, cur = [], ""
    for part in re.split(r"(?<=[.;:])\s+", line):
        while len(part) > limit:
            if cur:
                out.append(cur)
                cur = ""
            out.append(part[:limit])
            part = part[limit:]
        cand = f"{cur} {part}".strip()
        if len(cand) <= limit:
            cur = cand
        else:
            out.append(cur)
            cur = part
    if cur:
        out.append(cur)
    return out


def _locator(clauses: list[str], section_title: str = "") -> str:
    if not clauses:
        return section_title
    if len(clauses) == 1:
        return f"п. {clauses[0]}"
    return f"пп. {clauses[0]}–{clauses[-1]}"


def _chunk_company(lines: list[str]) -> tuple[list[str], list[tuple[str, list[str], str]]]:
    """Шапка (до первого раздела) и список (тело, номера пунктов, локатор) по разделам."""
    first = next((i for i, l in enumerate(lines) if i > 0 and SECTION_RE.match(l) and not CLAUSE_RE.match(l)), None)
    if first is None:                       # письмо/записка/допсоглашение: короткие, целиком
        return lines, []
    header, body = lines[:first], lines[first:]
    sections: list[list[str]] = []
    for l in body:
        if SECTION_RE.match(l) and not CLAUSE_RE.match(l):
            sections.append([l])
        elif sections:
            sections[-1].append(l)
    pieces: list[tuple[str, list[str], str]] = []
    for sec in sections:
        title = sec[0]
        clauses = [m.group(1) for l in sec if (m := CLAUSE_RE.match(l))]
        text = "\n".join(sec)
        if len(text) <= COMPANY_SECTION_MAX:
            pieces.append((text, clauses, _locator(clauses, f"разд. {title}")))
            continue
        cur: list[str] = [title]           # длинный раздел — по пунктам, заголовок раздела в каждом куске
        for l in sec[1:]:
            if len("\n".join(cur + [l])) > COMPANY_SECTION_MAX and len(cur) > 1:
                cl = [m.group(1) for x in cur if (m := CLAUSE_RE.match(x))]
                pieces.append(("\n".join(cur), cl, _locator(cl, title)))
                cur = [title]
            cur.append(l)
        cl = [m.group(1) for x in cur if (m := CLAUSE_RE.match(x))]
        pieces.append(("\n".join(cur), cl, _locator(cl, title)))
    return header, pieces


LAW_NOISE_RE = re.compile(r"^(URL:|Дата выгрузки:|См\. (?:комментарии|справку|Энциклопеди)|Информация об изменениях:)", re.I)
LAW_PART_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s")


def _chunk_law(lines: list[str], meta: dict) -> list[tuple[str, list[str], str]]:
    body = [l for l in lines[1:] if not LAW_NOISE_RE.match(l)]
    split: list[str] = []
    for l in body:
        split.extend(_split_long(l, LAW_CHUNK_CHARS))
    art = meta.get("doc_number", "")
    pieces: list[tuple[str, list[str], str]] = []
    cur: list[str] = []
    for l in split:
        if cur and len("\n".join(cur + [l])) > LAW_CHUNK_CHARS:
            parts = [m.group(1) for x in cur if (m := LAW_PART_RE.match(x))]
            pieces.append(("\n".join(cur), parts, art + (f", п./ч. {parts[0]}–{parts[-1]}" if parts else "")))
            tail: list[str] = []
            for prev in reversed(cur):     # перекрытие хвостом предыдущего куска
                if len("\n".join([prev] + tail)) > LAW_OVERLAP_CHARS:
                    break
                tail.insert(0, prev)
            cur = tail
        cur.append(l)
    if cur:
        parts = [m.group(1) for x in cur if (m := LAW_PART_RE.match(x))]
        pieces.append(("\n".join(cur), parts, art + (f", п./ч. {parts[0]}–{parts[-1]}" if parts else "")))
    return pieces


def chunk_document(filename: str, text: str) -> list[Chunk]:
    lines = normalize_text(text).split("\n")
    meta = extract_metadata(filename, lines)
    card = doc_card(meta, filename)
    base = {k: meta.get(k, "") for k in ("doc_type", "doc_title", "doc_number", "doc_date", "edition",
                                          "valid_from", "valid_to", "status", "amends")}
    base["counterparties"] = list(meta.get("counterparties", []))
    base["amended_clauses"] = list(meta.get("amended_clauses", []))

    if meta["doc_type"] == "law":
        header_text = lines[0]
        pieces = _chunk_law(lines, meta)
    else:
        header_lines, pieces = _chunk_company(lines)
        header_text = "\n".join(header_lines)
        if not pieces:                       # короткий документ целиком одним фрагментом
            header_text = "\n".join(lines[:3])
            pieces = [("\n".join(lines), [], meta.get("doc_number", ""))]
        else:                                # ::0 — шапка: стороны, дата, статус редакции
            pieces.insert(0, (header_text, [], "преамбула"))

    chunks = []
    for i, (body, clauses, loc) in enumerate(pieces):
        chunks.append(Chunk(
            chunk_id=f"{filename}::{i}", source_filename=filename, header=header_text, body=body,
            text=f"{card}\n---\n{body}", clauses=clauses, locator=loc, **base,
        ))
    return chunks


def link_amendments(chunks: list[Chunk]) -> None:
    """Помечает пункты основного договора, изменённые допсоглашением, и дописывает это в text."""
    norm = lambda s: re.sub(r"[\s№]", "", s).upper()
    supplements = [c for c in chunks if c.amends]
    seen: set[str] = set()
    for sup in supplements:
        if sup.source_filename in seen:
            continue
        seen.add(sup.source_filename)
        when = f" от {_ru_date(sup.doc_date)}" if sup.doc_date else ""
        for c in chunks:
            if c.doc_type != "company" or c.amends or norm(c.doc_number) != norm(sup.amends):
                continue
            touched = [cl for cl in sup.amended_clauses if cl in c.clauses] if sup.amended_clauses else []
            if c.chunk_id.endswith("::0"):
                note = f"К договору заключено {sup.doc_number}{when} (изменяет: " + \
                       (", ".join(f"п. {x}" for x in sup.amended_clauses) or "условия") + f"). Файл: {sup.source_filename}"
            elif touched:
                note = (f"ВНИМАНИЕ: {', '.join('п. ' + x for x in touched)} изменён(ы) {sup.doc_number}{when} — "
                        f"см. {sup.source_filename}; проверьте, к каким событиям применяется новая редакция")
            else:
                continue
            if note not in c.amended_by:
                c.amended_by.append(note)
                card, _, body = c.text.partition("\n---\n")
                c.text = f"{card}\n{note}\n---\n{body}"


def chunk_documents(documents: list[tuple[str, str]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for filename, text in documents:
        chunks.extend(chunk_document(filename, text))
    link_amendments(chunks)
    return chunks


# --------------------------------------------------------------------------- #
# BM25 (своя реализация: без лишних зависимостей и с учётом номеров документов)
# --------------------------------------------------------------------------- #
TOKEN_RE = re.compile(r"[0-9a-zа-я]+(?:[./-][0-9a-zа-я]+)*")
STOP = set("и в во не на по с со к ко о об от до за из у а но или ли же бы то что как это для при без над под".split())


def tokenize(text: str) -> list[str]:
    """Нижний регистр, ё→е, грубый стемминг (6 символов), составные номера + их части."""
    out: list[str] = []
    for tok in TOKEN_RE.findall(text.lower().replace("ё", "е")):
        if tok in STOP:
            continue
        if re.search(r"[./-]", tok):
            out.append(tok)
            out.extend(p for p in re.split(r"[./-]", tok) if p and p not in STOP)
            continue
        out.append(tok if tok.isdigit() or len(tok) <= 6 else tok[:6])
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.6) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avg)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            f = np.array([tf.get(t, 0) for tf in self.tf], dtype=np.float32)
            s += idf * f * (self.k1 + 1) / (f + norm)
        return s


# --------------------------------------------------------------------------- #
# Эмбеддинги
# --------------------------------------------------------------------------- #
def get_embeddings(client: httpx.Client, texts: list[str], base_url: str, api_key: str,
                   model: str = "baai/bge-m3", task: str = "", batch_size: int = 32,
                   workers: int = 4, retries: int = 3) -> np.ndarray:
    """L2-нормированные эмбеддинги через OpenAI-совместимый POST /embeddings (батчи параллельно)."""
    if not texts:
        raise ValueError("Список текстов для эмбеддинга пуст")
    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}

    def one(batch: list[str]) -> list[list[float]]:
        payload: dict = {"model": model, "input": batch, "encoding_format": "float"}
        if task and "jina" in model.lower():          # bge-m3 параметр task не принимает
            payload["task"] = task
        last: Exception | None = None
        for attempt in range(retries):
            try:
                r = client.post(url, headers=headers, json=payload, timeout=120.0)
                r.raise_for_status()
                data = r.json()["data"]
                vecs: list[list[float] | None] = [None] * len(batch)
                for pos, item in enumerate(data):
                    vecs[int(item.get("index", pos))] = item["embedding"]
                if any(v is None for v in vecs):
                    raise ValueError("Получены не все embeddings")
                return vecs  # type: ignore[return-value]
            except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
                last = exc
                code = getattr(getattr(exc, "response", None), "status_code", 0)
                if code in (400, 401, 403, 404, 422):
                    break                                   # ретраи не помогут
                if attempt + 1 < retries:
                    threading.Event().wait(1.5 * (attempt + 1))
        raise RuntimeError(f"Эмбеддер недоступен: {type(last).__name__}: {last}")

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(batches)))) as ex:
        results = list(ex.map(one, batches))
    matrix = np.asarray([v for batch in results for v in batch], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# --------------------------------------------------------------------------- #
# Хранилище и поиск
# --------------------------------------------------------------------------- #
def _norm_name(s: str) -> str:
    return s.lower().replace("ё", "е").replace("«", "").replace("»", "").replace('"', "").strip()


def _name_in(name: str, text: str) -> bool:
    """«Альфа-Трейд» не должен совпадать внутри «Альфа-Трейдинг»: проверяем границы слова."""
    pat = r"(?<![0-9a-zа-я])" + re.escape(_norm_name(name)) + r"(?![0-9a-zа-я])"
    return re.search(pat, _norm_name(text)) is not None


def _num_variants(num: str) -> list[str]:
    n = num.lower().replace("ё", "е")
    return list({n, n.replace("/", "-"), n.replace("-", ""), n.replace("/", "")})


class VectorStore:
    """Гибридный индекс. Сигнатура search() совместима со старым FAISS-хранилищем из agent.py."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray | None, meta: dict | None = None) -> None:
        self.chunks = chunks
        self.vectors = vectors              # None → только BM25
        self.meta = meta or {}
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self.by_id = {c.chunk_id: c for c in chunks}
        self._qcache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self.embed_failures = 0

    # --- вспомогательное для агента и чата ---------------------------------
    def documents(self) -> list[str]:
        return sorted({c.source_filename for c in self.chunks})

    def doc_chunks(self, filename: str) -> list[Chunk]:
        return [c for c in self.chunks if c.source_filename == filename]

    def _query_vec(self, query: str, client, base_url, api_key, model, task) -> np.ndarray | None:
        if self.vectors is None or self.embed_failures >= 5:
            return None
        with self._lock:
            if query in self._qcache:
                return self._qcache[query]
        try:
            vec = get_embeddings(client, [query], base_url, api_key, model=model, task=task, batch_size=1)[0]
        except Exception as exc:  # noqa: BLE001 — поиск не должен падать из-за эмбеддера
            with self._lock:
                self.embed_failures += 1
            print(f"Эмбеддинг запроса не получен ({exc}); поиск только по BM25", file=sys.stderr)
            return None
        with self._lock:
            self._qcache[query] = vec
        return vec

    def _entity_boost(self, query: str) -> np.ndarray:
        """Буст фрагментам документа, чей номер или контрагент прямо назван в вопросе."""
        boost = np.zeros(len(self.chunks), dtype=np.float32)
        q = query.lower().replace("ё", "е")
        for i, c in enumerate(self.chunks):
            if c.doc_type != "company" or "Реквизиты" in c.locator:
                continue
            hit = any(_name_in(cp, query) for cp in c.counterparties)
            hit = hit or (c.doc_number and not c.doc_number.startswith("ДС")
                          and any(v in q for v in _num_variants(c.doc_number)))
            hit = hit or (c.amends and any(v in q for v in _num_variants(c.amends)))
            if hit:
                boost[i] = 1.0
        return boost

    def search(self, query: str, client: httpx.Client | None = None, base_url: str = "", api_key: str = "",
               model: str = "baai/bge-m3", task: str = "", top_k: int = 5) -> list[tuple[Chunk, float]]:
        """[(Chunk, score)] по убыванию. score — RRF (+буст), сравним только внутри одного запроса."""
        if not self.chunks or not query.strip():
            return []
        k = min(top_k, len(self.chunks))
        n = len(self.chunks)
        rrf = np.zeros(n, dtype=np.float32)
        C = 60.0

        bm = self.bm25.scores(tokenize(query))
        if bm.max() > 0:
            order = np.argsort(-bm)[:50]
            rrf[order] += 1.0 / (C + np.arange(1, len(order) + 1))

        qv = self._query_vec(query, client, base_url, api_key, model, task) if client is not None else None
        if qv is not None:
            sims = self.vectors @ qv
            order = np.argsort(-sims)[:50]
            rrf[order] += 1.0 / (C + np.arange(1, len(order) + 1))
        elif bm.max() <= 0:
            return []

        # буст ≈ +1 место в одном рейтинге: поднимает нужный договор, но не выталкивает законы целиком
        rrf += self._entity_boost(query) * (1.0 / (C + 1))
        top = np.argsort(-rrf)[:k]
        return [(self.chunks[int(i)], float(rrf[int(i)])) for i in top if rrf[int(i)] > 0]


# --------------------------------------------------------------------------- #
# Построение, сериализация, кэш
# --------------------------------------------------------------------------- #
def corpus_hash(documents: list[tuple[str, str]], model: str) -> str:
    model_key = model.strip().lower().rsplit("/", 1)[-1]      # «BAAI/bge-m3» == «baai/bge-m3»
    h = hashlib.sha256(f"{CHUNKER_VERSION}|{model_key}\n".encode())
    for name, text in sorted(documents):
        h.update(name.encode() + b"\0" + text.encode() + b"\0")
    return h.hexdigest()


def build_store(documents: list[tuple[str, str]], client: httpx.Client | None, base_url: str, api_key: str,
                model: str = "baai/bge-m3", task: str = "", batch_size: int = 32) -> VectorStore:
    """Разбивает и считает эмбеддинги. Без клиента/при сбое эмбеддера — индекс только BM25."""
    chunks = chunk_documents(documents)
    vectors = None
    if client is not None and base_url and api_key:
        try:
            vectors = get_embeddings(client, [c.text for c in chunks], base_url, api_key,
                                     model=model, task=task, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001
            print(f"Не удалось посчитать эмбеддинги базы ({exc}); работаем на BM25", file=sys.stderr)
    meta = {"model": model, "chunker": CHUNKER_VERSION, "num_documents": len(documents),
            "num_chunks": len(chunks), "corpus_hash": corpus_hash(documents, model),
            "has_vectors": vectors is not None}
    return VectorStore(chunks, vectors, meta)


def dump_store(store: VectorStore) -> str:
    """Компактная строка (zlib+base64, векторы float16) — для kb_index_data.py."""
    import zlib
    payload = {
        "meta": store.meta,
        "chunks": [asdict(c) for c in store.chunks],
        "vectors": base64.b64encode(store.vectors.astype(np.float16).tobytes()).decode() if store.vectors is not None else None,
        "dim": int(store.vectors.shape[1]) if store.vectors is not None else 0,
    }
    return base64.b64encode(zlib.compress(json.dumps(payload, ensure_ascii=False).encode(), 9)).decode()


def load_dump(blob: str) -> VectorStore:
    import zlib
    payload = json.loads(zlib.decompress(base64.b64decode(blob)))
    chunks = [Chunk(**c) for c in payload["chunks"]]
    vectors = None
    if payload.get("vectors"):
        raw = np.frombuffer(base64.b64decode(payload["vectors"]), dtype=np.float16)
        vectors = raw.reshape(len(chunks), payload["dim"]).astype(np.float32)
    return VectorStore(chunks, vectors, payload["meta"])


def load_or_build_store(documents: list[tuple[str, str]], client: httpx.Client | None, base_url: str,
                        api_key: str, model: str = "baai/bge-m3") -> VectorStore:
    """Готовый индекс из модуля kb_index_data (если хэш базы и модель совпали), иначе строим заново."""
    expected = corpus_hash(documents, model)
    try:
        import kb_index_data  # type: ignore[import-not-found]
        if getattr(kb_index_data, "CORPUS_HASH", "") == expected:
            store = load_dump(kb_index_data.DATA)
            print(f"Индекс загружен из kb_index_data.py: {len(store.chunks)} фрагментов", file=sys.stderr)
            return store
        print("kb_index_data.py не подходит к этой базе (другой хэш) — строим индекс заново", file=sys.stderr)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"kb_index_data.py не прочитан ({exc}) — строим индекс заново", file=sys.stderr)
    return build_store(documents, client, base_url, api_key, model=model)


def get_or_create_vectorstore(source_path: str | Path, store_dir: str | Path, client: httpx.Client,
                              base_url: str, api_key: str, model: str = "baai/bge-m3",
                              passage_task: str = "", **_: object) -> VectorStore:
    """Для веб-чата и локальных скриптов: кэш в папке store_dir/index.txt (пересборка при смене базы)."""
    documents = load_documents(source_path)
    expected = corpus_hash(documents, model)
    path = Path(store_dir) / "index.txt"
    if path.is_file():
        try:
            store = load_dump(path.read_text(encoding="utf-8"))
            if store.meta.get("corpus_hash") == expected and store.vectors is not None:
                return store
        except Exception as exc:  # noqa: BLE001
            print(f"Кэш индекса повреждён ({exc}), пересобираем", file=sys.stderr)
    store = load_or_build_store(documents, client, base_url, api_key, model=model)
    if store.vectors is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump_store(store), encoding="utf-8")
    return store
