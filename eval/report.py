"""Сводный отчет eval/reports/<tag>/report.md.

Собирается из артефактов прогона: answers.jsonl, judge.jsonl,
fabrication.jsonl, classify.jsonl, meta.json и trace.jsonl, если он есть.
Перегенерация без нового прогона: python -m eval.report --tag <tag>.
"""

from __future__ import annotations

import json
from pathlib import Path

from .common import EVAL_DIR, load_jsonl
from .judge import summarize

REPORTS_DIR = EVAL_DIR / "reports"
QUESTION_CUT = 110
ANSWER_CUT = 170


def load_artifacts(tag: str) -> dict:
    """Читает артефакты прогона, отсутствующие файлы дают пустые списки."""
    tag_dir = REPORTS_DIR / tag

    def read(name: str) -> list[dict]:
        path = tag_dir / name
        return load_jsonl(path) if path.is_file() else []

    meta_path = tag_dir / "meta.json"
    meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "records": read("answers.jsonl"),
        "verdicts": read("judge.jsonl"),
        "fabrication": read("fabrication.jsonl"),
        "classify": read("classify.jsonl"),
        "trace": read("trace.jsonl"),
        "meta": meta,
        "dir": tag_dir,
    }


def _table(header: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _cut(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _group_stats(records: list[dict], verdicts: dict, fabrications: dict,
                 classifications: dict, key: str) -> list[list[object]]:
    groups: dict[str, list] = {}
    for rec in records:
        groups.setdefault(str(rec.get(key) or "—"), []).append(rec)
    rows = []
    for name, items in sorted(groups.items()):
        scores = [verdicts[r["id"]]["score"] for r in items
                  if r["id"] in verdicts
                  and verdicts[r["id"]]["score"] is not None]
        wins = sum(1 for s in scores if s > 0.5)
        losses = sum(1 for s in scores if s < 0.5)
        fab = sum(1 for r in items if fabrications.get(r["id"], {})
                  .get("fabrication_suspect"))
        false_ref = sum(
            1 for r in items
            if r.get("golden_status", "answered") == "answered"
            and classifications.get(r["id"], {}).get("label") != "answered")
        avg = f"{sum(scores) / len(scores):.3f}" if scores else "—"
        rows.append([name, len(items), avg, wins, losses, false_ref, fab])
    return rows


def _header(art: dict, summary: dict) -> list[str]:
    meta = art["meta"]
    cand_s = meta.get("durations", {}).get("candidate_s", "?")
    models = meta.get("models", {})
    lines = [
        f"# Отчет оценки {meta.get('tag', '?')}",
        "",
        f"- Набор **{meta.get('set', '?')}**"
        f" ({len(art['records'])} вопросов"
        + (f", limit {meta['limit']}" if meta.get("limit") else "") + ")",
        f"- Дата {meta.get('finished', '?')}",
        f"- Решение `{meta.get('repo', '?')}`"
        f" @ `{meta.get('commit', '?')}`",
        f"- Модели, решение `{models.get('llm', '?')}`,"
        f" судья `{models.get('judge', '?')}`",
        f"- Время прогона решения {cand_s} с"
        f" (лимит {meta.get('time_limit', '?')} с)",
    ]
    if summary:
        lines += [
            "",
            "## Итог по методике лидерборда",
            "",
            f"- **Score = 0.7 × Match + 0.3 × Win = {summary['score']}**",
            f"- Match Rate **{summary['match_rate']}**"
            f" (оценено {summary['n_judged']}/{summary['n_questions']},"
            f" ошибок судьи {summary['n_judge_errors']})",
            f"- Win Rate **{summary['win_rate']}**"
            f" (победы {summary['wins']}, поражения {summary['losses']},"
            f" ничьи {summary['draws']})",
        ]
    else:
        lines += ["", "_Судья не запускалась (нет baseline или LLM)._"]
    return lines


def _category_tables(art: dict, verdicts: dict, fabrications: dict,
                     classifications: dict) -> list[str]:
    header = ["категория", "n", "ср. балл", "W", "L", "ложн. отказы",
              "подозр. выдумок"]
    rows = _group_stats(art["records"], verdicts, fabrications,
                        classifications, "category")
    lines = ["", "## Разбивка по категориям", "", _table(header, rows)]
    kind_rows = _group_stats(art["records"], verdicts, fabrications,
                             classifications, "kind")
    lines += ["", "### По типу вопроса", "",
              _table(["тип"] + header[1:], kind_rows)]
    return lines


def _false_refusals(art: dict, classifications: dict) -> list[str]:
    records = [r for r in art["records"]
               if r.get("golden_status", "answered") == "answered"]
    bad = [r for r in records
           if classifications.get(r["id"], {}).get("label") != "answered"]
    lines = ["", "## Ложные отказы", ""]
    if not records:
        return lines + ["_Нет вопросов с golden_status=answered._"]
    rate = len(bad) / len(records)
    lines.append(f"Доля **{len(bad)} из {len(records)} ({rate:.0%})**. "
                 "Эталон отвечает, а решение отказалось или переспросило.")
    if bad:
        rows = [[r["id"], r["category"],
                 classifications[r["id"]]["label"],
                 _cut(r.get("candidate") or "", ANSWER_CUT)] for r in bad]
        lines += ["", _table(["id", "категория", "поведение", "ответ"], rows)]
    return lines


def _fabrications(art: dict, verdicts: dict, fabrications: dict) -> list[str]:
    judge_fab = {v["id"] for v in art["verdicts"] if v.get("fabricated")}
    rule_fab = {f["id"] for f in art["fabrication"]
                if f.get("fabrication_suspect")}
    union = judge_fab | rule_fab
    total = len(art["records"])
    lines = ["", "## Выдумки", "",
             f"Доля **{len(union)} из {total} ({len(union) / total:.0%})**,"
             " вердикты судьи и подозрения детектора вместе.",
             f"- обнулены судьей, {len(judge_fab)}"
             f" ({', '.join(sorted(judge_fab)) or 'нет'})",
             f"- подозрения детектора, {len(rule_fab)}"
             f" ({', '.join(sorted(rule_fab)) or 'нет'})"]
    suspects = [f for f in art["fabrication"] if f.get("suspects")]
    if suspects:
        rows = []
        for item in suspects[:15]:
            for s in item["suspects"][:3]:
                rows.append([item["id"], s["kind"], s["fragment"],
                             s["detail"]])
        lines += ["", _table(["id", "вид", "фрагмент", "деталь"], rows)]
    return lines


def _worst(art: dict, verdicts: dict, classifications: dict,
           fabrications: dict) -> list[str]:
    scored = [v for v in art["verdicts"] if v.get("score") is not None]
    scored.sort(key=lambda v: v["score"])
    worst = scored[:10]
    lines = ["", "## Топ худших вопросов", ""]
    if not worst:
        return lines + ["_Нет оцененных вопросов._"]
    records = {r["id"]: r for r in art["records"]}
    rows = []
    for v in worst:
        rec = records.get(v["id"], {})
        if v.get("fabricated"):
            reason = "выдуманный факт, вопрос обнулен"
        elif classifications.get(v["id"], {}).get("label") in {"refused",
                                                               "clarify"} \
                and rec.get("golden_status") == "answered":
            reason = f"ложный {classifications[v['id']]['label']}"
        else:
            reason = _cut("; ".join(v.get("reasons", [])), 90) or "—"
        rows.append([v["id"], v["score"], rec.get("category", "—"),
                     _cut(rec.get("question", ""), QUESTION_CUT), reason])
    lines.append(_table(["id", "балл", "категория", "вопрос",
                         "причина"], rows))
    for v in worst[:3]:
        rec = records.get(v["id"], {})
        lines += ["",
                  f"**id {v['id']}** (балл {v['score']})",
                  f"- Эталон {_cut(rec.get('golden', ''), ANSWER_CUT)}",
                  f"- Команда {_cut(rec.get('candidate') or 'нет',
                                    ANSWER_CUT)}",
                  f"- Baseline {_cut(rec.get('baseline') or 'нет',
                                     ANSWER_CUT)}"]
    return lines


def _search_miss(art: dict) -> list[str]:
    lines = ["", "## Промахи поиска", ""]
    if not art["trace"]:
        return lines + [
            "_trace.jsonl не найден (решение не писало трассировку)._"]
    traced = {}
    for item in art["trace"]:
        files = {Path(s.get("filename", "")).name
                 for s in item.get("sources", []) if isinstance(s, dict)}
        traced[item["id"]] = files
    missed = []
    for rec in art["records"]:
        found = traced.get(rec["id"], set())
        diff = [Path(d).name for d in rec.get("source_docs", [])
                if Path(d).name not in found]
        if diff:
            missed.append((rec["id"], rec.get("category", "—"), diff))
    if not missed:
        return lines + ["_Все документы из source_docs попали в контекст._"]
    lines.append(f"Вопросов с промахом **{len(missed)}** "
                 f"из {len(art['records'])}.")
    rows = [[qid, cat, _cut(", ".join(d), 120)] for qid, cat, d in missed]
    lines += ["", _table(["id", "категория", "не найденные документы"], rows)]
    return lines


def _footer(art: dict, summary: dict) -> list[str]:
    meta = art["meta"]
    lines = ["", "## Технические детали", "",
             f"- Команда `{meta.get('command', '?')}`",
             f"- База {meta.get('kb_files', '?')} файлов"
             f" ({meta.get('kb_dir', '?')})",
             f"- Baseline `{meta.get('baseline_dir', 'нет')}`",
             f"- Python `{meta.get('python', '?')}`",
             f"- Артефакты `{art['dir']}`"]
    if summary:
        lines.append(f"- Из Match исключено judge-ошибок "
                     f"{summary['n_judge_errors']}")
    return lines


def build_report(art: dict) -> str:
    """Собирает markdown-отчет из артефактов одного прогона."""
    verdicts = {v["id"]: v for v in art["verdicts"]}
    fabrications = {f["id"]: f for f in art["fabrication"]}
    classifications = {c["id"]: c for c in art["classify"]}
    summary = summarize(art["verdicts"]) if art["verdicts"] else {}

    lines = []
    lines += _header(art, summary)
    lines += _category_tables(art, verdicts, fabrications, classifications)
    lines += _false_refusals(art, classifications)
    lines += _fabrications(art, verdicts, fabrications)
    lines += _worst(art, verdicts, classifications, fabrications)
    lines += _search_miss(art)
    lines += _footer(art, summary)
    return "\n".join(lines) + "\n"


def write_report(tag: str) -> Path:
    """Строит и записывает report.md для тега, возвращает путь."""
    art = load_artifacts(tag)
    if not art["records"]:
        raise SystemExit(f"Нет answers.jsonl для тега «{tag}» "
                         f"в {art['dir']}")
    path = art["dir"] / "report.md"
    path.write_text(build_report(art), encoding="utf-8")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Перегенерировать report.md по артефактам прогона")
    parser.add_argument("--tag", required=True, help="метка прогона")
    args = parser.parse_args()
    path = write_report(args.tag)
    print(f"Отчет записан в {path}")


if __name__ == "__main__":
    main()
