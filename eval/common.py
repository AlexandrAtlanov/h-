"""Общие утилиты eval. Работа с jsonl, хэши, окружение и LLM-клиент.

Ключи API читаются только из переменных окружения и не попадают в логи,
отчеты и исключения.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import httpx

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent


def load_jsonl(path: str | Path) -> list[dict]:
    """Читает jsonl-файл, пустые строки пропускает."""
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: Iterable[dict], path: str | Path) -> None:
    """Атомарно записывает записи в jsonl-файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(record: dict, path: str | Path) -> None:
    """Дописывает одну запись в конец jsonl-файла."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def digest(*parts: str) -> str:
    """Короткий детерминированный хэш от набора строк."""
    joined = "\x00".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def env(name: str, default: str = "") -> str:
    """Значение переменной окружения, пустое считается отсутствующим."""
    value = os.getenv(name, "").strip()
    return value if value else default


def load_dotenv_silent() -> None:
    """Тихо загружает .env из текущего каталога, если есть dotenv."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def parse_json_object(text: str) -> dict | None:
    """Достает первый JSON-объект из ответа модели или None."""
    text = (text or "").strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


class LLMError(RuntimeError):
    """LLM недоступна или вернула нечитаемый ответ."""


class LLMClient:
    """Тонкий клиент OpenAI-совместимого POST /chat/completions.

    Адрес и ключ берутся из окружения (LLM_BASE_URL и LLM_API_KEY либо
    OPENAI_*). Значение ключа нигде не сохраняется.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 120.0) -> None:
        self.base_url = (base_url or env("LLM_BASE_URL",
                                         env("OPENAI_BASE_URL"))).rstrip("/")
        self.api_key = api_key or env("LLM_API_KEY", env("OPENAI_API_KEY"))
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def chat(self, messages: list[dict], model: str, temperature: float = 0.0,
             max_tokens: int = 1024, retries: int = 2) -> str:
        """Один запрос генерации, пустой ответ считается ошибкой."""
        if not self.configured:
            raise LLMError("нет LLM_BASE_URL / LLM_API_KEY в окружении")
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = self.base_url + "/chat/completions"
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = httpx.post(url, headers=headers, json=payload,
                                  timeout=self.timeout)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"].get("content")
                if isinstance(content, str) and content.strip():
                    return content
                raise LLMError("пустой ответ модели")
            except (httpx.HTTPError, KeyError, IndexError, ValueError,
                    LLMError) as exc:
                last = exc
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"LLM недоступна после {retries + 1} попыток: "
                       f"{type(last).__name__}")
