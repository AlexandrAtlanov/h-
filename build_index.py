"""Предрасчёт индекса базы знаний в kb_index_data.py (Роль 1).

Зачем: на платформе копируются только .py файлы из корня ZIP, поэтому готовый индекс
кладём внутрь .py-модуля. solution.py берёт его, если хэш базы совпал; иначе строит индекс сам.

Запуск (ключи в .env или окружении: LLM_BASE_URL, LLM_API_KEY, при необходимости EMB_MODEL):
    python build_index.py                    # база из data/knowledge-base (или data/kb.zip)
    python build_index.py путь/к/kb.zip      # другая база
Пересобирать после любого изменения rag.py, влияющего на разбиение (CHUNKER_VERSION), или базы.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from rag import build_store, corpus_hash, dump_store, load_documents

ROOT = Path(__file__).resolve().parent


def pick_kb() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    data = Path(os.getenv("DATA_DIR", "data"))
    for name in ("kb.zip", "knowledge-base.zip", "knowledge_base.zip", "knowledge-base"):
        if (data / name).exists():
            return data / name
    raise SystemExit(f"Не найдена база в {data}/ (kb.zip или папка knowledge-base)")


def main() -> None:
    base = (os.getenv("EMB_BASE_URL") or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")).rstrip("/")
    key = os.getenv("EMB_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("EMB_MODEL", "baai/bge-m3")
    if not base or not key:
        raise SystemExit("Нужны LLM_BASE_URL и LLM_API_KEY (в .env или окружении)")

    kb = pick_kb()
    docs = load_documents(kb)
    t = time.time()
    with httpx.Client(timeout=120.0) as client:
        store = build_store(docs, client, base, key, model=model)
        if store.vectors is None:
            raise SystemExit("Эмбеддинги не посчитаны — проверьте ключ, адрес и EMB_MODEL. Файл не записан.")

        out = ROOT / "kb_index_data.py"
        out.write_text(
            '"""Сгенерировано build_index.py — не редактировать вручную."""\n'
            f"CORPUS_HASH = {corpus_hash(docs, model)!r}\n"
            f"MODEL = {model!r}\n"
            f"DATA = {dump_store(store)!r}\n",
            encoding="utf-8",
        )
        print(f"Готово за {time.time() - t:.0f}s: {len(docs)} документов, {len(store.chunks)} фрагментов, "
              f"размерность {store.vectors.shape[1]}, файл {out.name} {out.stat().st_size / 1e6:.1f} МБ")

        # Быстрая проверка поиска на демо-вопросах: глазами видно, находится ли нужный документ
        qfile = Path(os.getenv("DATA_DIR", "data")) / "questions.csv"
        if qfile.is_file():
            print("\nТоп-3 фрагмента по вопросам из", qfile)
            with qfile.open(encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    q = row["question"].replace("\n", " ")
                    hits = store.search(q[-600:], client=client, base_url=base, api_key=key, model=model, top_k=3)
                    print(f"\n[{row['id']}] {q[:110]}")
                    for c, s in hits:
                        print(f"   {s:.4f}  {c.source_filename.split('__')[-1][:50]}  {c.locator}")


if __name__ == "__main__":
    main()
