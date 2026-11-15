"""Классификация поведения ответа на answered, refused и clarify.

Сначала работают словари и регулярки, спорные случаи уточняются через LLM.
Нужен для метрики ложных отказов. Ложный отказ это вопрос, где эталон
отвечает, а решение отказалось или переспросило вместо ответа.
"""

from __future__ import annotations

import re

from .common import LLMClient, LLMError, parse_json_object
from .prompts import CLASSIFY_SYSTEM, CLASSIFY_USER_TEMPLATE

LABELS = {"answered", "refused", "clarify"}

REFUSE_MARKERS = (
    "не могу ответить", "не могу", "не смогу", "нет в базе", "в базе нет",
    "в базе знаний нет", "не хватает", "недостаточно информации",
    "нет данных", "нет сведений", "не представлены в базе",
    "отсутствуют в базе", "не удалось найти", "не удалось получить",
    "невозможно ответить", "не располагаю", "обратитесь к юристу",
    "обратитесь в отдел", "вне моей компетенции", "не по профилю",
    "вне компетенции", "ответить по документам не могу",
)
CLARIFY_MARKERS = (
    "уточните", "уточни,", "подскажите номер", "подскажите, какой",
    "какой договор", "о каком договоре", "о каком контрагенте",
    "каком контрагенте", "прошу уточнить", "уточняющий вопрос",
    "не могли бы вы уточнить", "уточните, пожалуйста",
)
CITATION_RE = re.compile(
    r"п\.?\s*\d|пп\.?\s*\d|пункт\w*\s*\d|ст\.?\s*\d|стать\w*\s*\d|№|п\. \d",
)


def classify_by_rules(answer: str) -> tuple[str, list[str], bool]:
    """Словарная эвристика, возвращает метку, маркеры и признак спорности."""
    text = re.sub(r"\s+", " ", answer or "").strip()
    if not text:
        return "refused", ["пустой ответ"], False

    low = " " + text.lower() + " "
    refuse = [m for m in REFUSE_MARKERS if m in low]
    clarify = [m for m in CLARIFY_MARKERS if m in low]
    has_citation = bool(CITATION_RE.search(text))

    if has_citation:
        # Ссылки на пункты и статьи считают ответ содержательным, в том
        # числе отказ с цитатой, когда решение называет недостающие данные.
        return "answered", refuse + clarify, False
    if refuse:
        return "refused", refuse, False
    if clarify:
        return "clarify", clarify, False
    if len(text) >= 60:
        return "answered", [], False
    return "", [], True  # короткий ответ без маркеров, спорный случай


def classify_with_llm(question: str, answer: str,
                      client: LLMClient, model: str) -> str:
    """Уточняет метку через LLM, при сбое вызывает LLMError."""
    user = CLASSIFY_USER_TEMPLATE.format(question=question, answer=answer)
    raw = client.chat(
        [{"role": "system", "content": CLASSIFY_SYSTEM},
         {"role": "user", "content": user}],
        model=model, temperature=0.0, max_tokens=200,
    )
    label = str((parse_json_object(raw) or {}).get("label", "")).lower()
    if label not in LABELS:
        raise LLMError(f"неизвестная метка классификатора: {label!r}")
    return label


def classify_answer(question: str, answer: str, client: LLMClient | None,
                    model: str) -> dict:
    """Полная классификация одного ответа."""
    label, markers, disputed = classify_by_rules(answer)
    confidence = "rules"
    if disputed and client is not None and client.configured:
        try:
            label = classify_with_llm(question, answer, client, model)
            confidence = "llm"
        except LLMError:
            label, confidence = "answered", "llm_fallback"
    elif disputed:
        label, confidence = "answered", "low"
    return {"label": label, "confidence": confidence, "markers": markers}


def classify_all(records: list[dict], client: LLMClient | None,
                 model: str) -> list[dict]:
    """Классифицирует ответы, ошибка одного вопроса не прерывает остальные."""
    results = []
    for record in records:
        try:
            item = classify_answer(record["question"],
                                   record.get("candidate") or "",
                                   client, model)
        except Exception:  # noqa: BLE001
            item = {"label": "answered", "confidence": "error", "markers": []}
        results.append({"id": record["id"], **item})
    return results


def false_refusal_rate(records: list[dict],
                       classifications: list[dict]) -> dict:
    """Доля вопросов, где эталон отвечает, а решение не ответило."""
    by_id = {c["id"]: c["label"] for c in classifications}
    golden_answered = [r for r in records
                       if r.get("golden_status", "answered") == "answered"]
    bad = [r for r in golden_answered if by_id.get(r["id"]) != "answered"]
    refused = [r for r in bad if by_id.get(r["id"]) == "refused"]
    clarify = [r for r in bad if by_id.get(r["id"]) == "clarify"]
    total = len(golden_answered)
    rate = len(bad) / total if total else 0.0
    return {"total": total, "false_refusals": len(refused) + len(clarify),
            "refused_ids": [r["id"] for r in refused],
            "clarify_ids": [r["id"] for r in clarify],
            "rate": round(rate, 4)}
