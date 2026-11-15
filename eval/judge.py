"""Локальный LLM-судья по разделу 8 технического регламента.

На каждый вопрос передаются golden, ответ baseline и ответ команды. Проводятся
два раунда со сменой порядка предъявления, в обоих оценивается ответ команды
относительно baseline по шкале 0 / 0.25 / 0.5 / 0.75 / 1.0. Балл вопроса
равен среднему двух раундов, выдуманный факт по вердикту судьи обнуляет
вопрос.

Match Rate считается как средний балл, вопросы с технической ошибкой судьи
исключаются. Win Rate равен отношению побед к сумме побед и поражений, ничьи
не учитываются, при полном ничейном исходе берется 0.5. Score равен
0.7 * Match + 0.3 * Win.

Вердикты кэшируются в judge_cache.jsonl. Ключ это хэш версии промпта, модели,
вопроса, обоих ответов и номера раунда.
"""

from __future__ import annotations

import time
from pathlib import Path

from .common import (LLMClient, LLMError, append_jsonl, digest,
                     load_jsonl, parse_json_object)
from .prompts import JUDGE_SYSTEM, JUDGE_USER_TEMPLATE, PROMPT_VERSION

SCORE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


class JudgeCache:
    """Append-only кэш вердиктов по стабильному ключу."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._items: dict[str, dict] = {}
        if path.is_file():
            for rec in load_jsonl(path):
                if rec.get("key"):
                    self._items[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self._items.get(key)

    def put(self, key: str, verdict: dict) -> None:
        self._items[key] = verdict
        append_jsonl({"key": key, **verdict}, self.path)


def snap_score(value: object) -> float | None:
    """Приводит оценку модели к ближайшему значению шкалы."""
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return min(SCORE_GRID, key=lambda grid: abs(grid - num))


def build_messages(golden: str, baseline: str, candidate: str,
                   round_index: int) -> list[dict]:
    """В раунде 1 ответ A это baseline, в раунде 2 наоборот."""
    if round_index == 1:
        answer_a, answer_b = baseline, candidate
        team, other = "B", "A"
    else:
        answer_a, answer_b = candidate, baseline
        team, other = "A", "B"
    user = JUDGE_USER_TEMPLATE.format(
        golden=golden, answer_a=answer_a, answer_b=answer_b,
        team_letter=team, baseline_letter=other,
    )
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user}]


def ask_verdict(client: LLMClient, model: str, messages: list[dict]) -> dict:
    """Вызывает судью и разбирает ответ в score, fabricated и reason."""
    raw = client.chat(messages, model=model, temperature=0.0, max_tokens=400)
    data = parse_json_object(raw) or {}
    score = snap_score(data.get("score"))
    if score is None:
        raise LLMError(f"судья не вернул оценку: {raw[:200]!r}")
    fabricated = str(data.get("fabricated")).strip().lower() \
        in {"true", "да", "1"}
    reason = str(data.get("reason") or "").strip()
    return {"score": score, "fabricated": fabricated, "reason": reason}


def judge_question(record: dict, client: LLMClient, model: str,
                   cache: JudgeCache) -> dict:
    """Возвращает вердикт по одному вопросу из двух раундов."""
    qid = record["id"]
    golden = record["golden"]
    baseline = record.get("baseline") or ""
    candidate = record.get("candidate") or ""

    # Пустые ответы разбираются без LLM.
    if not candidate.strip() and not baseline.strip():
        return {"id": qid, "score": 0.5, "rounds": [], "fabricated": False,
                "reasons": ["оба ответа пустые"], "error": None}
    if not candidate.strip():
        return {"id": qid, "score": 0.0, "rounds": [], "fabricated": False,
                "reasons": ["команда не дала ответ"], "error": None}
    if not baseline.strip():
        return {"id": qid, "score": 1.0, "rounds": [], "fabricated": False,
                "reasons": ["baseline не дал ответа"], "error": None}

    verdicts: list[dict] = []
    reasons: list[str] = []
    for round_index in (1, 2):
        key = digest(PROMPT_VERSION, model, record["question"],
                     baseline, candidate, str(round_index))
        verdict = cache.get(key)
        if verdict is None:
            messages = build_messages(golden, baseline, candidate, round_index)
            verdict = ask_verdict(client, model, messages)
            cache.put(key, verdict)
        verdicts.append(verdict)
        if verdict["reason"]:
            reasons.append(f"р{round_index}: {verdict['reason']}")

    fabricated = any(v["fabricated"] for v in verdicts)
    if fabricated:
        score = 0.0
        reasons.insert(0, "выдуманный факт, вопрос обнулен")
    else:
        score = sum(v["score"] for v in verdicts) / len(verdicts)
    return {"id": qid, "score": score,
            "rounds": [v["score"] for v in verdicts],
            "fabricated": fabricated, "reasons": reasons, "error": None}


def judge_all(records: list[dict], client: LLMClient, model: str,
              cache_path: Path, sleep_on_error: float = 0.5) -> list[dict]:
    """Судит все вопросы, ошибка одного вопроса не прерывает остальные."""
    cache = JudgeCache(cache_path)
    results: list[dict] = []
    for record in records:
        try:
            results.append(judge_question(record, client, model, cache))
        except Exception as exc:  # noqa: BLE001
            results.append({"id": record["id"], "score": None, "rounds": [],
                            "fabricated": False,
                            "reasons": [f"ошибка судьи: {exc}"],
                            "error": str(exc)})
            time.sleep(sleep_on_error)
    return results


def summarize(verdicts: list[dict]) -> dict:
    """Считает Match Rate, Win Rate и Score по методике регламента."""
    judged = [v for v in verdicts if v.get("error") is None]
    errors = len(verdicts) - len(judged)
    scores = [v["score"] for v in judged]
    match_rate = sum(scores) / len(scores) if scores else 0.0
    wins = sum(1 for s in scores if s > 0.5)
    losses = sum(1 for s in scores if s < 0.5)
    draws = sum(1 for s in scores if s == 0.5)
    if wins + losses:
        win_rate = wins / (wins + losses)
    else:
        win_rate = 0.5
    return {"n_questions": len(verdicts), "n_judged": len(judged),
            "n_judge_errors": errors, "match_rate": round(match_rate, 4),
            "wins": wins, "losses": losses, "draws": draws,
            "win_rate": round(win_rate, 4),
            "score": round(0.7 * match_rate + 0.3 * win_rate, 4)}
