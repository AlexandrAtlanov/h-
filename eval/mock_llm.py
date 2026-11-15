"""Мок OpenAI-совместимого API для смоук-тестов eval-конвейера.

Поднимает HTTP-сервер с тремя эндпоинтами. POST /chat/completions различает
вызовы по маркерам в system-промпте: судья ([EVAL-JUDGE) получает вердикт,
классификатор ([EVAL-CLASSIFY) получает метку, планировщик и ответ
ассистента (маркеры agent.py) получают заготовки под их JSON-протокол,
прочие запросы получают нейтральный текст. POST /embeddings возвращает
детерминированные векторы (sha256 от текста), POST /rerank сохраняет
порядок документов, GET /health проверяет готовность.

Ключи и внешние вызовы не нужны, все ответы детерминированы.
Запуск: python -m eval.mock_llm [--port 8765]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MOCK_ANSWER = (
    "По договору поставки № П-17/2024 с ООО «Альфа-Трейд» оплата "
    "производится в течение 30 календарных дней с даты подписания "
    "универсального передаточного документа (п. 4.2)."
)
EMBED_DIM = 32


def mock_vector(text: str) -> list[float]:
    """Детерминированный нормированный вектор из хэша текста."""
    raw = hashlib.sha256(text.encode("utf-8")).digest()
    values = [b / 255.0 - 0.5 for b in (raw * (EMBED_DIM // len(raw) + 1))]
    vec = values[:EMBED_DIM]
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [round(v / norm, 6) for v in vec]


def chat_content(payload: dict) -> str:
    """Формирует ответ по виду запроса: судья, классификатор, агент, прочее."""
    messages = payload.get("messages") or []
    system = next((m.get("content", "") for m in messages
                   if m.get("role") == "system"), "")
    if system.startswith("[EVAL-JUDGE"):
        return json.dumps({"score": 0.75, "fabricated": False,
                           "reason": "мок-вердикт: ответы сопоставимы"})
    if system.startswith("[EVAL-CLASSIFY"):
        return json.dumps({"label": "answered", "reason": "мок-классификация"})
    if "модуль планирования" in system:
        # agent.py подставит исходный вопрос, если sub_queries пуст.
        return json.dumps({"is_legal_domain": True, "refusal_message": "",
                           "event_date": None, "sub_queries": []})
    if "юридический ассистент банка" in system:
        user = next((m.get("content", "") for m in messages
                     if m.get("role") == "user"), "")
        chunk_ids = re.findall(r'<document id="([^"]+)">', user)
        return json.dumps({"action": "FINAL_ANSWER", "status": "ANSWERED",
                           "answer": MOCK_ANSWER,
                           "used_chunk_ids": chunk_ids[:2],
                           "missing_info": None})
    return MOCK_ANSWER


class Handler(BaseHTTPRequestHandler):
    """Роутинг мок-сервера, вся диагностика идет в stderr."""

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[mock] {self.command} {self.path}", file=sys.stderr)

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return

        if self.path == "/chat/completions":
            self._json(200, {
                "choices": [{"message": {"role": "assistant",
                                         "content": chat_content(payload)}}],
            })
        elif self.path == "/embeddings":
            inputs = payload.get("input") or []
            if isinstance(inputs, str):
                inputs = [inputs]
            self._json(200, {
                "data": [{"index": i, "embedding": mock_vector(text)}
                         for i, text in enumerate(inputs)],
            })
        elif self.path == "/rerank":
            docs = payload.get("documents") or []
            top_n = int(payload.get("top_n") or len(docs))
            self._json(200, {"results": [
                {"index": i, "relevance_score": 1.0 - i * 0.01}
                for i in range(min(top_n, len(docs)))
            ]})
        else:
            self._json(404, {"error": "not found"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Мок OpenAI-совместимого API")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[mock] слушаю http://127.0.0.1:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
