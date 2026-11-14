"""Сборка ZIP для сабмита (Роль 1) с проверками перед отправкой.

    python make_zip.py                     → dist/submission_<дата>_<коммит>.zip (конвейер agent)
    python make_zip.py --pipeline simple   → то же, но solution.py работает упрощённым путём (1 вызов LLM)

Что делает:
  * кладёт в КОРЕНЬ архива все .py из корня репозитория, кроме служебных (EXCLUDE), и requirements.txt;
  * requirements.txt для сабмита формируется из SUBMISSION_PACKAGES с версиями из разрешённого списка;
  * проверяет: синтаксис, импорты в чистой папке, отсутствие ключей, размер ≤ 128 МБ,
    актуальность kb_index_data.py (совпадение хэша базы);
  * в архив НЕ попадают .env, данные, вопросы, папки app/, eval/, data/.
"""

from __future__ import annotations

import os
import py_compile
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"

# Разрешённые организаторами версии (requirements.txt из папки с данными)
ALLOWED = {
    "fastapi": "0.143.2", "uvicorn": "0.54.0", "pydantic": "2.14.2", "python-multipart": "0.0.20",
    "python-dotenv": "1.1.1", "openai": "3.45.0", "httpx": "0.28.1", "requests": "2.32.5",
    "langchain": "1.4.1", "langchain-openai": "1.7.4", "langchain-text-splitters": "1.1.3",
    "numpy": "2.2.6", "faiss-cpu": "1.15.1", "chromadb": "1.5.9", "rank-bm25": "0.2.2",
    "bm25s": "0.3.16", "scikit-learn": "1.9.1", "rapidfuzz": "3.14.1", "pandas": "2.3.3",
    "pypdf": "6.21.4", "python-docx": "1.2.0", "beautifulsoup4": "4.14.2", "lxml": "6.0.2",
    "PyYAML": "6.0.3", "SQLAlchemy": "2.0.44", "tiktoken": "0.12.0",
}
# Что реально нужно сабмиту. Добавили новую библиотеку в agent.py/rag.py — допишите сюда.
SUBMISSION_PACKAGES = ["httpx", "numpy", "python-dotenv"]

# Служебные скрипты, которые не нужны на платформе
EXCLUDE = {"make_zip.py", "build_index.py", "test.py", "mock_api.py", "conftest.py", "run_config.py"}
REQUIRED = {"solution.py", "agent.py", "rag.py"}
MAX_MB = 128

# импортируемое имя → пакет из ALLOWED (для проверки, что всё нужное есть в requirements)
IMPORT_TO_PACKAGE = {"httpx": "httpx", "numpy": "numpy", "dotenv": "python-dotenv", "openai": "openai",
                     "requests": "requests", "faiss": "faiss-cpu", "rank_bm25": "rank-bm25", "bm25s": "bm25s",
                     "sklearn": "scikit-learn", "rapidfuzz": "rapidfuzz", "pandas": "pandas", "yaml": "PyYAML",
                     "tiktoken": "tiktoken", "pydantic": "pydantic", "langchain": "langchain",
                     "langchain_openai": "langchain-openai", "langchain_text_splitters": "langchain-text-splitters",
                     "chromadb": "chromadb", "pypdf": "pypdf", "docx": "python-docx", "bs4": "beautifulsoup4",
                     "lxml": "lxml", "sqlalchemy": "SQLAlchemy"}

errors: list[str] = []
warnings: list[str] = []


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return ""


def secret_values() -> list[str]:
    """Значения ключей из .env и окружения — их не должно быть в коде."""
    vals = []
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line and re.search(r"KEY|TOKEN|SECRET|PASSWORD", line.split("=")[0], re.I):
                v = line.split("=", 1)[1].strip().strip("'\"")
                if len(v) >= 8:
                    vals.append(v)
    for k, v in os.environ.items():
        if re.search(r"API_KEY|TOKEN|SECRET", k) and len(v) >= 8:
            vals.append(v)
    return vals


def main() -> None:
    pipeline = "agent"
    if "--pipeline" in sys.argv:
        pipeline = sys.argv[sys.argv.index("--pipeline") + 1]
        if pipeline not in {"agent", "simple"}:
            sys.exit("--pipeline: agent или simple")
    files = sorted(p for p in ROOT.glob("*.py") if p.name not in EXCLUDE)
    names = {p.name for p in files}
    for req in REQUIRED - names:
        errors.append(f"нет обязательного файла {req}")

    # 1. синтаксис и сторонние импорты
    stdlib = set(sys.stdlib_module_names)
    local = {p.stem for p in files} | {"run_config", "kb_index_data"}
    used_packages: set[str] = set()
    for p in files:
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"синтаксическая ошибка в {p.name}: {exc.msg}")
        if p.name == "kb_index_data.py":
            continue
        for mod in re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][\w]*)", p.read_text(encoding="utf-8"), re.M):
            if mod in stdlib or mod in local or mod == "__future__":
                continue
            pkg = IMPORT_TO_PACKAGE.get(mod)
            if pkg is None:
                errors.append(f"{p.name}: импорт «{mod}» не из разрешённого списка")
            else:
                used_packages.add(pkg)
    for pkg in sorted(used_packages - set(SUBMISSION_PACKAGES)):
        errors.append(f"пакет {pkg} используется в коде, но не добавлен в SUBMISSION_PACKAGES")
    for pkg in SUBMISSION_PACKAGES:
        if pkg not in ALLOWED:
            errors.append(f"пакет {pkg} не разрешён организаторами")
    requirements = "".join(f"{pkg}=={ALLOWED[pkg]}\n" for pkg in SUBMISSION_PACKAGES if pkg in ALLOWED)

    # 2. секреты
    secrets = secret_values()
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        if any(s in text for s in secrets):
            errors.append(f"в {p.name} найдено значение ключа из .env/окружения")
        if p.name != "kb_index_data.py" and re.search(r"\b(sk-[A-Za-z0-9_\-]{20,}|eyJ[A-Za-z0-9_\-]{30,})", text):
            errors.append(f"в {p.name} похоже на API-ключ")

    # 3. готовый индекс
    if "kb_index_data.py" not in names:
        warnings.append("нет kb_index_data.py — индекс будет строиться на платформе (≈ +1 мин). "
                        "Запустите python build_index.py")
    else:
        try:
            sys.path.insert(0, str(ROOT))
            from rag import corpus_hash, load_documents
            import kb_index_data  # type: ignore[import-not-found]
            kb = next((ROOT / "data" / n for n in ("kb.zip", "knowledge-base.zip", "knowledge-base")
                       if (ROOT / "data" / n).exists()), None)
            if kb is None:
                warnings.append("не найдена локальная база в data/ — не могу проверить хэш kb_index_data.py")
            elif kb_index_data.CORPUS_HASH != corpus_hash(load_documents(kb), getattr(kb_index_data, "MODEL", "bge-m3")):
                warnings.append("kb_index_data.py устарел (база или разбиение поменялись) — "
                                "пересоберите: python build_index.py. Иначе индекс построится на платформе.")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"не удалось проверить kb_index_data.py: {exc}")

    if errors:
        print("ОШИБКИ — архив не собран:")
        for e in errors:
            print("  ✗", e)
        sys.exit(1)

    # 4. сборка
    DIST.mkdir(exist_ok=True)
    sha = git("rev-parse", "--short", "HEAD") or "nogit"
    dirty = "-dirty" if git("status", "--porcelain", "--", *sorted(names)) else ""
    out = DIST / f"submission_{time.strftime('%m%d_%H%M')}_{sha}{dirty}_{pipeline}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            z.write(p, arcname=p.name)              # строго в корень архива
        z.writestr("requirements.txt", requirements)
        z.writestr("run_config.py", f'PIPELINE = "{pipeline}"\n')
    size_mb = out.stat().st_size / 1e6
    if size_mb > MAX_MB:
        out.unlink()
        sys.exit(f"ОШИБКА: архив {size_mb:.1f} МБ > {MAX_MB} МБ")

    # 5. импорт в чистой папке (как на платформе: только содержимое архива)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(out) as z:
            z.extractall(tmp)
        check = subprocess.run(
            [sys.executable, "-c", "import rag, agent, solution; print('ok')"],
            cwd=tmp, capture_output=True, text=True,
            env={k: v for k, v in os.environ.items() if not re.search(r"KEY|TOKEN|SECRET", k)} | {"DATA_DIR": "data"},
        )
        if check.returncode != 0 or "ok" not in check.stdout:
            print("ВНИМАНИЕ: модули не импортируются из чистой папки:\n" + check.stderr[-1500:])
            warnings.append("импорт в чистой папке не прошёл (если это ModuleNotFoundError httpx/numpy — "
                            "просто не установлены локально)")

    print(f"Архив: {out.relative_to(ROOT)} ({size_mb:.2f} МБ)")
    print("Состав:", ", ".join(sorted(names | {"requirements.txt", "run_config.py"})), f"| конвейер: {pipeline}")
    print("requirements.txt:", requirements.strip().replace("\n", ", "))
    if dirty:
        warnings.append("есть незакоммиченные изменения — закоммитьте и поставьте тег sub-ДД-N перед отправкой")
    for w in warnings:
        print("  ⚠", w)
    print("\nДальше: тег  git tag sub-$(date +%d)-N && git push --tags  и строка в SUBMISSIONS.md")


if __name__ == "__main__":
    main()
