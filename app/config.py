"""Конфигурация веб-приложения из переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
APP_NAME = "Юридический ассистент"
APP_VERSION = "0.1.0"


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(os.getenv(name, "") or default)))
    except ValueError:
        return default


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(os.getenv(name, "") or default)))
    except ValueError:
        return default


def _resolve(value: str, default: str) -> Path:
    path = Path(value or default)
    if path.is_absolute():
        return path
    local = ROOT / path
    if local.exists() or not (Path.cwd() / path).exists():
        return local
    return Path.cwd() / path


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = field(default_factory=lambda: _resolve(os.getenv("DATA_DIR", ""), "data"))
    vectorstore_dir: Path = field(
        default_factory=lambda: _resolve(
            os.getenv("VECTORSTORE_DIR", "") or os.getenv("STORE_DIR", ""), "data/vectorstore"
        )
    )
    max_concurrent_answers: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT_ANSWERS", 4, 1, 16)
    )
    chat_timeout_sec: float = field(
        default_factory=lambda: _env_float("CHAT_TIMEOUT_SEC", 170.0, 30.0, 600.0)
    )
    max_history_messages: int = field(
        default_factory=lambda: _env_int("MAX_HISTORY_MESSAGES", 20, 2, 100)
    )
    max_history_chars: int = field(
        default_factory=lambda: _env_int("MAX_HISTORY_CHARS", 16_000, 2_000, 200_000)
    )
    max_message_chars: int = field(
        default_factory=lambda: _env_int("MAX_MESSAGE_CHARS", 4_000, 200, 20_000)
    )


KB_NAMES = ("kb.zip", "knowledge-base.zip", "knowledge_base.zip", "knowledge-base", "kb")


def find_kb(data_dir: Path) -> Path:
    """Архив или папка базы знаний: те же имена, что у solution.py."""
    if data_dir.is_file() and data_dir.suffix.lower() == ".zip":
        return data_dir
    for name in KB_NAMES:
        path = data_dir / name
        if path.is_file():
            return path
        if path.is_dir() and any(path.iterdir()):
            return path
    raise FileNotFoundError(
        f"База знаний не найдена в {data_dir}. Ожидается один из: {', '.join(KB_NAMES)}"
    )
