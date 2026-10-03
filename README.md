# Parcer — сбор и семантическая аналитика сообщений из Telegram

Пайплайн, который собирает сообщения из Telegram (посты каналов, комментарии обсуждений, сообщения чатов), складывает их в PostgreSQL, строит по ним векторные индексы FAISS и отвечает на вопросы естественным языком: считает релевантные сообщения, показывает примеры и строит тематические сводки через LLM. Сверху — Telegram-бот с доступом по паролю, подписками на темы и рассылкой дайджестов.

Проект делался как внутренний аналитический инструмент: сотрудники задают вопросы текстом в Telegram и получают ответ + выгрузку подвыборки в Excel.

---

## Задача

Поток сообщений из десятков Telegram-источников (каналы, обсуждения, чаты) нужно было превратить в быстрый источник ответов на регулярные вопросы:

- «сколько было сообщений по определённой теме за 2024 год?»
- «покажи 5 примеров жалоб на турникеты»
- «на что жалуются по оплате проезда?»
- «сводка за последнюю неделю»

Ключевые требования: ответы строго по факту данных (без выдумок), работа со склейками сообщений («микротреды»), опора на векторный поиск по смыслу и работа как с локальной LLM, так и с внешним API.

## Как это работает

```
Telegram (каналы / обсуждения / чаты)
      │  Telethon, курсоры по msg_id, прокси, обработка FloodWait
      ▼
ingest_telegram.py ──► PostgreSQL
      │                 tg_sources, tg_source_state
      │                 tg_posts, tg_comments, tg_thread_state
      │                 tg_chat_messages (+ staging-таблицы)
      ▼
sql/thread_docs_views.sql ── MATERIALIZED VIEW
      │   thread_docs_channel   (пост + комментарии)
      │   thread_docs_chat_reply (корень + ответы, recursive parent chain)
      ▼
build_faiss_threads.py ── индекс по «тредам»
build_faiss_micro.py   ── индекс по «микротредам» (склейки из чатов)
      │   index.faiss + meta.json + mapping.csv + doc_id_map.npy
      ▼
Вопрос в Telegram
      │
      ▼
python_telegram_bot.py ── авторизация по паролю, белый список, подписки, rate-limit
      │
      ▼
analysis_tg_pg.py ── разбор интента: greeting │ count │ examples │ summary
      │
      ├── СКОУП: линии/станции/период/аспекты (алиасы, режимы)
      │
      ├── СЕМАНТИЧЕСКОЕ ЯДРО (semantic_faiss.py)
      │     эмбеддинги: intfloat/multilingual-e5-base (локально)
      │                 или GigaChat Embeddings-2 (gigachat_embeddings.py)
      │     FAISS threads + micro, пороги близости
      │     + лексический приоритет по токенам (all → any → остальные)
      │
      ├── count    → количество + выгрузка подвыборки (XLSX/CSV)
      ├── examples → K примеров + полная подвыборка
      └── summary  → summary_llm.py
                        ├── (опц.) semantic top-N по запросу
                        ├── LLM тегирует каждый отзыв (1–3 тега)
                        ├── частотные теги → «тег: пояснение»
                        └── анти-галлюцинационный пост-фильтр
                            (пункт остаётся, только если опирается
                             на исходные сообщения; иначе — rule-based fallback)
```

Генерация сводок уходит в **GigaChat** или локальный **Ollama** (`qwen2.5:7b-instruct` по умолчанию) — провайдер выбирается переменной `SUMMARY_PROVIDER`.

## Файлы

| Файл | Назначение |
|---|---|
| `ingest_telegram.py` | Сбор из Telegram через Telethon: посты, комментарии обсуждений, сообщения чатов; курсоры, прокси, обработка `FloodWait` |
| `load_tg_csv_to_pg.py` | Импорт выгрузок CSV в staging-таблицы и upsert в основные таблицы |
| `rebuild_after_csv_import.py` | Пересборка производных (materialized views / индексов) после CSV-импорта |
| `run_pipeline.py` | Оркестратор: lock-файл, ingest → refresh matviews → атомарная пересборка FAISS-индексов в окне сборки |
| `refresh_mviews.py` | Пересборка materialized views `thread_docs_channel` / `thread_docs_chat_reply` |
| `build_faiss_threads.py` | Оффлайн-построение FAISS-индекса по «тредам» (пост + комментарии) |
| `build_faiss_micro.py` | Оффлайн-построение FAISS-индекса по «микротредам» (сессии из чатов) |
| `semantic_faiss.py` | Загрузка индексов, эмбеддинги, семантический поиск, кэш модели, перезагрузка при изменении |
| `gigachat_embeddings.py` | Эмбеддинги через GigaChat API (OAuth-токен, батчи, ретраи) |
| `analysis_tg_pg.py` | Ядро аналитики: интенты, скоуп (линии/станции/период/аспекты), семантика, счёт, примеры, экспорт |
| `summary_llm.py` | Сводки через теги: top-N, тегирование LLM, анти-галлюцинационный фильтр, rule-based fallback |
| `python_telegram_bot.py` | Telegram-бот: авторизация, белый список, подписки, дайджесты, rate-limit, self-test |
| `sql/schema.sql` | Схема таблиц источников, постов, комментариев, сообщений чатов и курсоров |
| `sql/staging_tg_tables.sql` | Staging-таблицы для CSV-импорта |
| `sql/thread_docs_views.sql` | Materialized views «тредов» для индексации |
| `sql/after_ingest_refresh.sql`, `sql/drop_dupes.sql` | Обслуживание: обновление представлений, дедупликация |
| `ingest_build/Parcerrun_ingest.cmd` | Windows-обёртка запуска ингеста по расписанию |
| `ingest_build/ingest_telegram.txt` | Зависимости сборщика (Telethon, psycopg) |
| `.env.example` | Все переменные окружения с безопасными значениями по умолчанию |

## Стек

Python 3.11 · Telethon · PostgreSQL + SQLAlchemy 2 / psycopg · pandas / numpy · sentence-transformers (`intfloat/multilingual-e5-base`) · GigaChat Embeddings-2 · FAISS · GigaChat / Ollama (LLM) · python-telegram-bot 21 · openpyxl / xlsxwriter

## Что показывает проект с инженерной стороны

- **Гибридный поиск**: семантика (эмбеддинги FAISS + пороги) комбинируется с доменными правилами и лексическим приоритетом — вместо «чистого» RAG, который на коротких запросах даёт мусор.
- **Два уровня индексации**: индекс по «тредам» (пост + комментарии) и по «микротредам» — сессии из чатов, где сообщения склеиваются по временным паузам и кластеризуются по эмбеддингам. Это позволяет искать смысл там, где нет явной структуры треда.
- **Разделение скоупа и смысла**: сначала отсекаем по линиям/станциям/периоду/аспектам, только потом считаем близость — экономит вычисления и убирает ложные попадания.
- **Борьба с галлюцинациями**: теги и формулировки LLM проверяются по исходным сообщениям, неподтверждённые отбрасываются; при недоступности LLM — детерминированный rule-based fallback.
- **Инкрементальный сбор**: курсоры по `msg_id` на источник и на тред, staging + upsert, дедупликация — повторный запуск не дублирует данные.
- **Отказоустойчивый пайплайн**: lock-файл против параллельных запусков, окно сборки индексов, атомарная пересборка через временный каталог (`atomic_build_with_tmp`) — при сбое сборки старый индекс сохраняется.
- **Продовая эксплуатация**: rate-limit рассылок, защита от дублей сообщений, кэш датасета с бастом по `.data_updated`, команда `/selftest` с контрольными запросами.
- **Безопасность**: все секреты и параметры — только через переменные окружения; данные сообщений и служебные артефакты индексов в репозитории не хранятся.

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # заполнить доступы
# PostgreSQL: применить sql/schema.sql, затем sql/thread_docs_views.sql

python ingest_telegram.py --with-comments --comments-backfill=200 --batch=300
python build_faiss_threads.py
python build_faiss_micro.py
python python_telegram_bot.py
```

Полный цикл (сбор → представления → индексы) оркеструет `run_pipeline.py`. Периодический запуск ингеста на Windows — через `ingest_build/Parcerrun_ingest.cmd`.

## Переменные окружения

Основные: `BOT_TOKEN`, `BOT_ACCESS_PASSWORD`, `BOT_ADMINS`, `BOT_ALLOWED_USERS`, `PG_DSN` (или `PG_USER`/`PG_PASSWORD`/`PG_HOST`/`PG_PORT`/`PG_DB`), `TG_API_ID`, `TG_API_HASH`, `TG_SESSION_FILE`, `TG_CHANNELS`/`TG_GROUPS`/`TG_DISCUSSIONS`/`TG_CHATS`, `TG_VIEW_RAW`, `TG_VIEW_THREADS`, `EMBED_PROVIDER`, `EMBEDDER`, `EMBED_DEVICE`, `FAISS_THREADS_DIR`, `FAISS_MICRO_DIR`, `FAISS_TOP_K`, `SUMMARY_PROVIDER`, `GIGACHAT_AUTH_KEY`, `OLLAMA_HOST`, `OLLAMA_MODEL`, `STATIONS_CSV`. Полный список с дефолтами — в `.env.example`.

## Ограничения и что дальше

- Правила скоупа (алиасы станций, аспекты) завязаны на предметную область — для нового домена нужен свой словарь.
- Индексы FAISS пересобираются оффлайн целыми файлами; инкрементального обновления векторов нет (только пропуск по state-файлам, если данные не менялись).
- Схема БД и представления рассчитаны на ручное применение `sql/*.sql` и refresh materialized views в окне сборки.
- Направления развития: инкрементальная досборка индексов, оценка качества сводок на ручном gold-set, вынос конфигурации правил в YAML.
