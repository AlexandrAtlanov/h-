"""Детектор выдуманных фактов без LLM, эвристика по базе знаний.

Извлекает из ответа числа, номера пунктов и статей, номера документов
и проверяет их наличие в текстах базы. Числа из вопроса разрешены.
Помечает fabrication_suspect со списком подозрительных фрагментов.

Подозрение детектора не обнуляет вопрос, а только попадает в отчет вместе
с вердиктами судьи. Ограничения. Числительные прописью не проверяются,
число из базы, встретившееся в другом контексте, пропускается, такие случаи
находит судья.
"""

from __future__ import annotations

import re
from pathlib import Path

# Числа вида 450 000, 0,1, 24, 10 000 000
NUMBER_RE = re.compile(r"\d+(?:,\d+)?(?: \d{3})*")
# Ссылки на пункты и статьи (п. 3.1, пп. 6.1 и 6.2, ст. 70, статья 13.11)
CLAUSE_RE = re.compile(r"(?:п|пп|пункт\w*)\.?\s*(\d+(?:\.\d+)*)",
                       re.IGNORECASE)
ARTICLE_RE = re.compile(r"(?:ст|статья\w*)\.?\s*(\d+(?:\.\d+)*)",
                        re.IGNORECASE)
# Номера документов (А-08/2024, ПТ-05/2023, К-12, 2-П, ЮО-7)
DOC_LETTER_NUM_RE = re.compile(r"№\s*([А-ЯЁ]{2,4}-\d{1,4}(?:/\d{4})?)")
DOC_NUM_LETTER_RE = re.compile(r"№\s*(\d{1,3}-[А-ЯЁ])")
# Специальные токены, вырезаются до проверки обычных чисел
DOC_REF_SPAN_RE = re.compile(
    r"№\s*[А-ЯЁ0-9][А-ЯЁ0-9/-]*|"
    r"(?:п|пп|пункт\w*|ст|статья\w*)\.?\s*\d+(?:\.\d+)*|"
    r"\d{2}\.\d{2}\.\d{4}",
    re.IGNORECASE,
)

_TEXT_EXTS = {".txt", ".md"}


class KnowledgeIndex:
    """Данные, которые детектор считает известными базе."""

    def __init__(self, kb_texts: dict[str, str]) -> None:
        self.raw = "\n".join(kb_texts.values())
        self.numbers = extract_numbers(self.raw)
        self.articles = {num for _, num in article_spans(self.raw)}
        doc_numbers: set[str] = set()
        for pattern in (DOC_LETTER_NUM_RE, DOC_NUM_LETTER_RE):
            doc_numbers.update(pattern.findall(self.raw))
        self.doc_numbers = doc_numbers

    @classmethod
    def from_dir(cls, kb_dir: str | Path) -> "KnowledgeIndex":
        """Собирает индекс из папки базы (data/knowledge-base)."""
        kb_dir = Path(kb_dir)
        texts = {
            path.name: path.read_text(encoding="utf-8-sig")
            for path in sorted(kb_dir.iterdir())
            if path.is_file() and path.suffix.lower() in _TEXT_EXTS
        }
        if not texts:
            raise FileNotFoundError(f"в {kb_dir} нет .txt/.md файлов")
        return cls(texts)


def normalize_number(token: str) -> str:
    """Переводит «450 000» в «450000», а «0,1» в «0.1»."""
    return token.replace(" ", "").replace(",", ".")


def extract_numbers(text: str) -> set[str]:
    """Множество нормализованных чисел текста."""
    return {normalize_number(tok) for tok in NUMBER_RE.findall(text)}


def article_spans(text: str) -> list[tuple[str, str]]:
    """Пары (спан, номер) для ссылок на статьи в тексте базы."""
    return [(m.group(0), m.group(1)) for m in ARTICLE_RE.finditer(text)]


def _strip_special_tokens(answer: str) -> str:
    """Убирает ссылки на пункты, статьи, номера и даты перед сбором чисел."""
    return DOC_REF_SPAN_RE.sub(" ", answer)


def check_answer(answer: str, question: str, index: KnowledgeIndex) -> dict:
    """Проверяет ответ и возвращает подозрения по видам."""
    suspects: list[dict] = []
    if not answer.strip():
        return {"fabrication_suspect": False, "suspects": suspects}

    question_numbers = extract_numbers(question)

    # 1. Номера документов: должны быть известны базе или упомянуты в вопросе.
    for pattern, kind in ((DOC_LETTER_NUM_RE, "doc_ref"),
                          (DOC_NUM_LETTER_RE, "doc_ref")):
        for match in pattern.finditer(answer):
            token = match.group(1)
            if token not in index.doc_numbers and token not in question:
                suspects.append({
                    "kind": kind, "fragment": match.group(0),
                    "detail": f"номер «{token}» не найден в базе знаний",
                })

    # 2. Статьи: номер должен встречаться в базе как статья.
    for match in ARTICLE_RE.finditer(answer):
        num = match.group(1)
        if num not in index.articles and num not in question:
            suspects.append({
                "kind": "article", "fragment": match.group(0),
                "detail": f"статья «{num}» не найдена в базе знаний",
            })

    # 3. Пункты: токен «N.M» должен присутствовать в текстах базы.
    for match in CLAUSE_RE.finditer(answer):
        num = match.group(1)
        if num not in index.raw and num not in question:
            suspects.append({
                "kind": "clause", "fragment": match.group(0),
                "detail": f"пункт «{num}» не найден в текстах базы",
            })

    # 4. Числа: должны быть в базе или в вопросе пользователя.
    plain = _strip_special_tokens(answer)
    for token in NUMBER_RE.findall(plain):
        num = normalize_number(token)
        if num not in index.numbers and num not in question_numbers:
            suspects.append({
                "kind": "number", "fragment": token,
                "detail": f"число «{token}» не найдено в базе знаний",
            })

    return {"fabrication_suspect": bool(suspects), "suspects": suspects}


def check_all(records: list[dict], index: KnowledgeIndex) -> list[dict]:
    """Проверяет поле candidate во всех записях."""
    results = []
    for record in records:
        check = check_answer(record.get("candidate") or "",
                             record["question"], index)
        results.append({"id": record["id"], **check})
    return results
