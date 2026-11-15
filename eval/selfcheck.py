"""Регресс-проверки детекторов eval, без LLM и ключей.

Проверяет fabrication_check на подготовленных кейсах, включая
документированные ограничения эвристики, и classify_answer на словарных
правилах. Запуск из корня репозитория: python -m eval.selfcheck
"""

from __future__ import annotations

from pathlib import Path

from .classify_answer import classify_by_rules
from .common import REPO_ROOT
from .fabrication_check import KnowledgeIndex, check_answer

QUESTION = "Какая арендная плата по договору с ООО «Северный Склад»?"

# (ответ, ожидаем fabrication_suspect)
FABRICATION_CASES = [
    ("Арендная плата составляет 450 000 рублей в месяц (п. 3.1).", False),
    ("Арендная плата составляет 620 000 рублей в месяц (п. 3.9).", True),
    ("Плата 500 000 рублей (договор № А-09/2024, п. 3.1).", True),
    # 90 есть в базе (инъекция в ЮО-7), 18 есть в шкале НДФЛ ст. 224.
    # Это документированные ограничения эвристики, их находит LLM-судья.
    ("Оплата в течение 90 календарных дней.", False),
    ("Пени 0,1% за день (п. 5.1).", False),
    ("Срок хранения 5 лет (записка № ЮО-7).", False),
    ("По ст. 71 ТК РФ срок до шести месяцев.", True),
    ("НДС по ставке 22 процента.", False),
    ("НДС по ставке 18 процентов.", False),
    ("МРОТ с 2026 года 27 093 рубля.", False),
    ("Неустойка 0,5% в день по П-17/2024.", True),
]

# (ответ, ожидаемая метка)
CLASSIFY_CASES = [
    ("Плата 450 000 в месяц, п. 3.1 договора.", "answered"),
    ("В базе нет данных о пособиях, ответить не могу.", "refused"),
    ("Подскажите номер договора, чтобы проверить условия.", "clarify"),
    ("Пени 0,1% (п. 5.1), но документов об увеличении в базе нет.",
     "answered"),
]


def default_kb_dir() -> Path:
    for candidate in (REPO_ROOT / "data" / "knowledge-base",
                      Path.cwd() / "data" / "knowledge-base"):
        if candidate.is_dir():
            return candidate
    raise SystemExit("Не найден data/knowledge-base")


def run_checks() -> int:
    """Возвращает число проваленных проверок, ноль значит все прошли."""
    index = KnowledgeIndex.from_dir(default_kb_dir())
    failures = 0

    for answer, expected in FABRICATION_CASES:
        result = check_answer(answer, QUESTION, index)
        got = result["fabrication_suspect"]
        mark = "OK " if got == expected else "FAIL"
        failures += got != expected
        details = "; ".join(s["detail"] for s in result["suspects"]) or "—"
        print(f"{mark} suspect={got!s:5} | {answer[:58]:58} | {details}")

    for answer, expected in CLASSIFY_CASES:
        label, _, _ = classify_by_rules(answer)
        mark = "OK " if label == expected else "FAIL"
        failures += label != expected
        print(f"{mark} classify={label:8} | {answer[:58]}")

    verdict = ("ВСЕ ПРОВЕРКИ ПРОШЛИ" if not failures
               else f"провалов {failures}")
    print(f"\nИтог: {verdict}")
    return failures


def main() -> None:
    raise SystemExit(1 if run_checks() else 0)


if __name__ == "__main__":
    main()
