"""Имитация платформы, прогон решения на наборе вопросов и полная оценка.

Одна команда оценивает любую версию решения (каталог, коммит или ветку
через --repo) на любом наборе eval/dataset/<set>.jsonl и выдает отчет,
сопоставимый с методикой лидерборда.

Примеры запуска

    python -m eval.run_eval --repo . --set demo --tag v1
    python -m eval.run_eval --repo ../h- --set main --tag pr-42 --limit 20

Порядок работы
1. собирает DATA_DIR из датасета и базы (questions.csv и knowledge-base.zip)
2. запускает python solution.py с окружением платформы (LLM_BASE_URL,
   LLM_API_KEY, LLM_MODEL, EMB_MODEL, MAX_WORKERS, TOP_K, TIME_LIMIT_SEC
   передаются из текущего окружения)
3. то же для baseline (eval/baseline, точная копия sample_submission
   организаторов, ответы кэшируются)
4. судит ответы (eval.judge), проверяет выдумки (eval.fabrication_check),
   классифицирует поведение (eval.classify_answer)
5. пишет в eval/reports/<tag> файлы answers.jsonl, judge.jsonl,
   fabrication.jsonl, classify.jsonl, trace.jsonl, meta.json и report.md

Ошибка отдельного вопроса или стадии не прерывает прогон, отчет
формируется с пометкой. Ключи API берутся только из окружения, можно
через .env в корне.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from .common import (EVAL_DIR, REPO_ROOT, LLMClient, digest,
                     load_dotenv_silent, load_jsonl, save_jsonl)
from . import classify_answer, fabrication_check, judge, report

DEFAULT_TIME_LIMIT = 1200
GRACE_SECONDS = 60


def load_dataset(set_name: str, limit: int | None) -> list[dict]:
    """Читает eval/dataset/<set>.jsonl и проверяет обязательные поля."""
    path = EVAL_DIR / "dataset" / f"{set_name}.jsonl"
    if not path.is_file():
        raise SystemExit(f"Набор не найден, {path}, см. eval/README.md")
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            missing = {"id", "question", "golden"} - rec.keys()
            if missing:
                raise SystemExit(f"Вопрос {rec.get('id', '?')} "
                                 f"без полей {missing}")
            rec.setdefault("category", "прочее")
            rec.setdefault("kind", "plain")
            rec.setdefault("golden_status", "answered")
            rec.setdefault("source_docs", [])
            records.append(rec)
    if limit:
        records = records[:limit]
    if not records:
        raise SystemExit(f"Набор {set_name} пуст")
    return records


def build_data_dir(records: list[dict], kb_dir: Path,
                   base: Path) -> Path:
    """Пишет questions.csv и knowledge-base.zip во временный каталог."""
    data_dir = base / "data"
    data_dir.mkdir(parents=True)
    with (data_dir / "questions.csv").open("w", encoding="utf-8",
                                           newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "question"])
        for rec in records:
            writer.writerow([rec["id"], rec["question"]])

    kb_files = sorted(p for p in kb_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in {".txt", ".md"})
    if not kb_files:
        raise SystemExit(f"В {kb_dir} нет .txt/.md файлов базы знаний")

    with zipfile.ZipFile(data_dir / "knowledge-base.zip", "w") as zf:
        for path in kb_files:
            zf.write(path, arcname=path.name)
    return data_dir


def read_submission(path: Path) -> dict[str, str]:
    """Читает submission.csv в словарь id -> ответ."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {row["id"].strip(): (row.get("answer") or "").strip()
                for row in csv.DictReader(fh)}


def _as_text(data: bytes | str | None) -> str:
    """Байты или строку из TimeoutExpired переводит в строку."""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data or ""


def run_solution(workdir: Path, data_dir: Path, log_path: Path,
                 time_limit: int, trace_path: Path | None = None
                 ) -> tuple[dict[str, str], float, bool]:
    """Запускает solution.py в workdir с окружением платформы.

    Возвращает ответы, длительность и признак успеха. submission.csv
    и trace.jsonl забираются в отчет и удаляются из workdir.
    """
    env = os.environ.copy()
    env["DATA_DIR"] = str(data_dir)
    env["TIME_LIMIT_SEC"] = str(time_limit)
    if trace_path is not None:
        env["TRACE_FILE"] = str(trace_path)

    started = time.monotonic()
    ok = True
    try:
        proc = subprocess.run(
            [sys.executable, "solution.py"], cwd=workdir, env=env,
            capture_output=True, text=True, timeout=time_limit + GRACE_SECONDS,
        )
        if proc.returncode != 0:
            ok = False
        log_path.write_text(
            f"=== exit {proc.returncode} ===\n{proc.stdout}\n"
            f"=== stderr ===\n{proc.stderr}", encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        ok = False
        out = _as_text(exc.stdout)
        err = _as_text(exc.stderr)
        log_path.write_text(
            f"=== TIMEOUT после {time_limit + GRACE_SECONDS}s ===\n"
            f"{out}\n=== stderr ===\n{err}", encoding="utf-8")
    duration = time.monotonic() - started

    answers = read_submission(workdir / "submission.csv")
    (workdir / "submission.csv").unlink(missing_ok=True)
    return answers, round(duration, 1), ok


def baseline_cache_path(set_name: str, kb_dir: Path,
                        baseline_dir: Path) -> Path:
    key = digest(set_name, str(kb_dir.resolve()), str(baseline_dir.resolve()))
    return EVAL_DIR / "reports" / "_cache" / f"baseline_{key}.jsonl"


def get_baseline_answers(set_name: str, records: list[dict], kb_dir: Path,
                         baseline_dir: Path, base: Path, time_limit: int
                         ) -> tuple[dict[str, str], float, bool]:
    """Берет ответы baseline из кэша либо прогоном на всем сете."""
    full = load_dataset(set_name, None)
    full_ids = [rec["id"] for rec in full]
    cache_path = baseline_cache_path(set_name, kb_dir, baseline_dir)
    cached: dict[str, str] = {}
    if cache_path.is_file():
        cached = {rec["id"]: rec.get("answer") or ""
                  for rec in load_jsonl(cache_path)}
    if all(qid in cached for qid in full_ids):
        return cached, 0.0, True

    answers, duration, ok = run_solution(
        baseline_dir, build_data_dir(full, kb_dir, base / "baseline"),
        base / "baseline.log", time_limit,
    )
    merged = {**cached, **answers}
    save_jsonl([{"id": qid, "answer": merged.get(qid, "")}
                for qid in full_ids], cache_path)
    return merged, duration, ok


def git_commit(repo: Path) -> str:
    try:
        proc = subprocess.run(["git", "-C", str(repo), "rev-parse",
                               "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10)
        return proc.stdout.strip() or "нет git"
    except (OSError, subprocess.SubprocessError):
        return "нет git"


def default_kb_dir(repo: Path) -> Path:
    for candidate in (REPO_ROOT / "data" / "knowledge-base",
                      repo / "data" / "knowledge-base"):
        if candidate.is_dir():
            return candidate
    raise SystemExit("Не найден data/knowledge-base, укажите --kb-dir")


def collect_trace(trace_path: Path, tag_dir: Path) -> None:
    if trace_path.is_file():
        target = tag_dir / "trace.jsonl"
        target.write_text(trace_path.read_text(encoding="utf-8"),
                          encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Полный прогон оценки решения на наборе вопросов")
    parser.add_argument("--repo", default=".",
                        help="каталог проверяемой версии с solution.py")
    parser.add_argument("--set", default="demo",
                        help="имя набора, eval/dataset/<set>.jsonl")
    parser.add_argument("--tag", default=None,
                        help="метка прогона (по умолчанию имя набора)")
    parser.add_argument("--limit", type=int, default=None,
                        help="взять первые N вопросов (для смоук-тестов)")
    parser.add_argument(
        "--baseline", default=str(EVAL_DIR / "baseline"), metavar="DIR",
        help="каталог baseline (по умолчанию eval/baseline)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="пропустить baseline и судью")
    parser.add_argument("--kb-dir", default=None,
                        help="папка базы знаний (.txt)")
    parser.add_argument("--judge-model", default=None,
                        help="модель судьи (JUDGE_MODEL/LLM_MODEL)")
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT,
                        help="лимит времени решения в секундах "
                             "(по умолчанию 1200)")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="не удалять временный DATA_DIR")
    args = parser.parse_args()

    load_dotenv_silent()
    repo = Path(args.repo).resolve()
    if not (repo / "solution.py").is_file():
        raise SystemExit(f"В {repo} нет solution.py")
    tag = args.tag or args.set
    tag_dir = EVAL_DIR / "reports" / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    (tag_dir / "logs").mkdir(exist_ok=True)
    kb_dir = (Path(args.kb_dir).resolve() if args.kb_dir
              else default_kb_dir(repo))

    records = load_dataset(args.set, args.limit)
    print(f"Набор {args.set}, {len(records)} вопросов, решение {repo} "
          f"@ {git_commit(repo)}")

    judge_client = LLMClient()
    judge_model = (args.judge_model
                   or os.getenv("JUDGE_MODEL")
                   or os.getenv("LLM_MODEL")
                   or "deepseek/deepseek-v4.1-flash")

    durations: dict[str, float] = {}
    tmp_holder: list[Path] = []      # каталоги, которые нельзя удалять
    with tempfile.TemporaryDirectory(prefix="evalrun-") as tmp:
        tmp_dir = Path(tmp)
        if args.keep_tmp:
            keep = Path(tempfile.mkdtemp(prefix="evalrun-keep-"))
            tmp_holder.append(keep)
            tmp_dir = keep
            print(f"DATA_DIR сохраняется в {keep}")

        # 1. Кандидат
        data_dir = build_data_dir(records, kb_dir, tmp_dir / "candidate")
        trace_path = tmp_dir / "candidate" / "trace.jsonl"
        candidate, durations["candidate_s"], cand_ok = run_solution(
            repo, data_dir, tag_dir / "logs" / "candidate.log",
            args.time_limit, trace_path)
        collect_trace(trace_path, tag_dir)
        if not cand_ok:
            print("ВНИМАНИЕ, решение завершилось с ошибкой или таймаутом, "
                  "смотрите logs/candidate.log")

        # 2. Baseline, кэшируется
        baseline: dict[str, str] = {}
        baseline_dir = None
        if args.no_baseline:
            print("Baseline пропущен (--no-baseline)")
        else:
            baseline_dir = Path(args.baseline).resolve()
            if not (baseline_dir / "solution.py").is_file():
                print(f"Baseline не найден ({baseline_dir}), судья не "
                      "запустится")
            else:
                baseline, durations["baseline_s"], base_ok = \
                    get_baseline_answers(args.set, records, kb_dir,
                                         baseline_dir, tmp_dir,
                                         args.time_limit)
                if not base_ok:
                    print("ВНИМАНИЕ, baseline упал, судим против его "
                          "пустых ответов")

    # 3. answers.jsonl
    for rec in records:
        rec["candidate"] = candidate.get(rec["id"], "")
        rec["baseline"] = baseline.get(rec["id"], "")
    save_jsonl(records, tag_dir / "answers.jsonl")

    # 4. Детектор выдумок без LLM
    try:
        index = fabrication_check.KnowledgeIndex.from_dir(kb_dir)
        fabrications = fabrication_check.check_all(records, index)
    except Exception as exc:  # noqa: BLE001
        print(f"fabrication_check упал, {exc}")
        fabrications = []
    save_jsonl(fabrications, tag_dir / "fabrication.jsonl")

    # 5. Классификатор поведения, словари плюс LLM для спорных
    try:
        classifications = classify_answer.classify_all(
            records, judge_client, judge_model)
    except Exception as exc:  # noqa: BLE001
        print(f"classify_answer упал, {exc}")
        classifications = []
    save_jsonl(classifications, tag_dir / "classify.jsonl")

    # 6. Судья, нужны baseline и LLM
    verdicts: list[dict] = []
    if baseline and judge_client.configured:
        verdicts = judge.judge_all(records, judge_client, judge_model,
                                   tag_dir / "judge_cache.jsonl")
    elif not baseline:
        print("Судья пропущена, нет ответов baseline")
    else:
        print("Судья пропущена, нет LLM_BASE_URL / LLM_API_KEY")
    save_jsonl(verdicts, tag_dir / "judge.jsonl")

    # 7. meta.json и отчет
    meta = {
        "tag": tag, "set": args.set, "limit": args.limit,
        "repo": str(repo), "commit": git_commit(repo),
        "kb_dir": str(kb_dir), "kb_files": len(list(kb_dir.glob('*.txt'))),
        "baseline_dir": str(baseline_dir) if baseline_dir else None,
        "time_limit": args.time_limit,
        "models": {"llm": os.getenv("LLM_MODEL", ""),
                   "judge": judge_model},
        "durations": durations,
        "python": sys.version.split()[0],
        "command": "python -m eval.run_eval " + " ".join(sys.argv[1:]),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (tag_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = report.write_report(tag)
    if verdicts:
        summary = judge.summarize(verdicts)
        print(f"Score={summary['score']} Match={summary['match_rate']} "
              f"Win={summary['win_rate']} "
              f"(W {summary['wins']} / L {summary['losses']} / "
              f"D {summary['draws']})")
    print(f"Отчет записан в {report_path}")


if __name__ == "__main__":
    main()
