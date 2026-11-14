# Юридический ассистент

Веб-чат по предоставленной базе юридических документов. Ответы строятся только по найденным фрагментам базы и сопровождаются ссылками на документ и пункт.

## Быстрый запуск

Нужны Docker и Docker Compose.

```powershell
Copy-Item .env.example .env
docker compose up --build
```

После запуска откройте <http://localhost:8000>.

Остановка:

```powershell
docker compose down
```

Индекс документов строится при первом запуске и сохраняется в volume `index-cache`. При изменении базы знаний volume можно удалить и построить индекс заново:

```powershell
docker compose down -v
docker compose up --build
```

## Настройка

`.env` используется только локально и не должен попадать в git.

```env
LLM_BASE_URL=https://your-proxy.example/v1
LLM_API_KEY=
LLM_MODEL=deepseek/deepseek-v4.1-flash

EMB_BASE_URL=https://your-proxy.example/v1
EMB_API_KEY=
EMB_MODEL=baai/bge-m3

RERANKER_BASE_URL=https://your-proxy.example/v1
RERANKER_API_KEY=
RERANKER_MODEL=
RERANKER_TOP_N=8
```

Имена моделей и адрес прокси должны соответствовать настройкам организаторов. Официальный endpoint `api.deepseek.com` поддерживает чат, но не `/embeddings`; без отдельного endpoint для `bge-m3` приложение использует BM25 как резервный поиск.

## Что умеет интерфейс

- чат с историей текущего диалога;
- статусы ответа: подтверждено документами, неполный ответ, нет подтверждения, уточнение;
- список подтверждающих фрагментов;
- открытие полного документа по клику на источник;
- подсветка найденного пункта в документе;
- каталог документов с поиском по названию, номеру и контрагенту;
- адаптивная раскладка для desktop и mobile;
- явные состояния подготовки индекса, ошибки и недоступности модели.

## Архитектура

```text
браузер
  -> FastAPI (app/)
      -> LegalService
          -> LegalAgent (agent.py)
              -> гибридный поиск rag.py
                  -> BM25 + bge-m3 + опциональный reranker
          -> API документов и фрагментов
```

`solution.py`, `agent.py` и `rag.py` остаются основным контуром решения. Веб-приложение использует тот же индекс и адаптирует результат агента для интерфейса.

## API

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/healthz` | состояние индекса и моделей |
| `GET` | `/api/meta` | версия и примеры вопросов |
| `GET` | `/api/documents` | каталог документов |
| `GET` | `/api/documents/{id}` | полный документ и его фрагменты |
| `GET` | `/api/chunks/{id}` | отдельный фрагмент |
| `POST` | `/api/chat` | ответ на вопрос и источники |

Пример запроса:

```json
{
  "message": "Какая арендная плата по договору?",
  "history": []
}
```

## Локальная проверка

Проверка синтаксиса и тесты запускаются из корня проекта:

```powershell
node --check app/static/app.js
python -m compileall -q app tests
python -m unittest discover -s tests -v
```

Для проверки интерфейса без API можно запустить локальную заглушку:

```powershell
python tests/mock_llm.py 9000
```

Затем в другом окне:

```powershell
$env:LLM_BASE_URL = "http://127.0.0.1:9000/v1"
$env:LLM_API_KEY = "test"
$env:EMB_BASE_URL = "http://127.0.0.1:9000/v1"
$env:EMB_API_KEY = "test"
uvicorn app.main:app --reload
```

Заглушка нужна только для локальной проверки и не входит в production Docker-образ.

## Ограничения

- ассистент не использует внешние источники для юридических выводов;
- при отсутствии подтверждения в базе ответ не подменяется догадкой;
- без доступного embedding endpoint поиск работает в режиме BM25;
- API-ключи хранятся только в `.env` или секретах окружения.
