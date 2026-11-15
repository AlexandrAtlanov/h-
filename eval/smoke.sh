#!/usr/bin/env bash
# Смоук-тест eval-конвейера на мок-LLM, без ключей и внешних API.
# Запуск из корня репозитория командой bash eval/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${MOCK_PORT:-8765}"
TAG="${SMOKE_TAG:-smoke}"
REPO="${SMOKE_REPO:-.}"

python -m eval.mock_llm --port "$PORT" &
MOCK_PID=$!
trap 'kill "$MOCK_PID" 2>/dev/null || true' EXIT

python - "$PORT" <<'PY'
import sys, time, urllib.request
url = f"http://127.0.0.1:{sys.argv[1]}/health"
for _ in range(50):
    try:
        urllib.request.urlopen(url, timeout=1)
        sys.exit(0)
    except Exception:
        time.sleep(0.2)
sys.exit("мок-сервер не запустился")
PY

LLM_BASE_URL="http://127.0.0.1:$PORT" \
LLM_API_KEY="mock-key" \
LLM_MODEL="mock/model" \
JUDGE_MODEL="mock/judge" \
python -m eval.run_eval --repo "$REPO" --set demo --tag "$TAG" --limit 5 \
    --time-limit 180

echo
echo "SMOKE OK, отчет лежит в eval/reports/$TAG/report.md"
