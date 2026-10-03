# -*- coding: utf-8 -*-
import os
import re
import json
import asyncio
import logging
import html
import datetime as dt
import pathlib
import time
from collections import OrderedDict
from logging.handlers import RotatingFileHandler
from typing import Set, List, Optional, Tuple, Dict, Any
from zoneinfo import ZoneInfo
import pandas as pd
from telegram.error import BadRequest

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram.request import HTTPXRequest

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).with_name(".env"))
load_dotenv(override=False)

from sqlalchemy import create_engine, text

import analysis_tg_pg as apg  



# ===================== НАСТРОЙКИ =====================
BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()


def _parse_ids(raw: str) -> Set[int]:
    """'123, 456' -> {123, 456}. Пустая строка -> пустое множество."""
    out: Set[int] = set()
    for chunk in re.split(r"[,\s;]+", raw or ""):
        chunk = chunk.strip().lstrip("-")
        if chunk.isdigit():
            out.add(int(chunk))
    return out


ADMINS: Set[int] = _parse_ids(os.getenv("BOT_ADMINS", ""))
ALLOWED_USERS: Set[int] = _parse_ids(os.getenv("BOT_ALLOWED_USERS", ""))
TABLE_USERS = os.getenv("BOT_USERS_TABLE", "bot_users")

GLOBAL_RATE_PER_SEC = int(os.getenv("BROADCAST_RPS", "25"))
SLEEP_BETWEEN_SENDS = max(0.04, 1.0 / GLOBAL_RATE_PER_SEC)

BOT_PASSWORD = (os.getenv("BOT_ACCESS_PASSWORD") or "").strip()

# Рассылка (дайджест)
DIGEST_TZ = os.getenv("DIGEST_TZ", os.getenv("BOT_TZ", "Europe/Moscow"))
DIGEST_TICK_SEC = int(os.getenv("DIGEST_TICK_SEC", "60"))  # как часто проверять HH:MM
DIGEST_MAX_ROWS = int(os.getenv("DIGEST_MAX_ROWS", "5000"))  # ограничение df перед LLM
DIGEST_TOP_N = int(os.getenv("DIGEST_TOP_N", "60"))  # top-N для summary_llm (если доступен)
DIGEST_MAX_TOPICS = int(os.getenv("DIGEST_MAX_TOPICS", "5"))
DIGEST_MAX_BULLETS_PER_TOPIC = int(os.getenv("DIGEST_MAX_BULLETS_PER_TOPIC", "4"))
DIGEST_MAX_HTML_CHARS = int(os.getenv("DIGEST_MAX_HTML_CHARS", "3500"))

# Отправлять XLSX с исходными комментариями по каждой теме дайджеста
DIGEST_SEND_TOPIC_EXPORTS = int(os.getenv("DIGEST_SEND_TOPIC_EXPORTS", "1"))
DIGEST_TOPIC_EXPORT_MAX_ROWS = int(os.getenv("DIGEST_TOPIC_EXPORT_MAX_ROWS", "5000"))

WELCOME_TEXT = (
    "👋 Привет! Я анализирую сообщения из Telegram (посты + комментарии + чаты).\n\n"
    "Спроси, например:\n"
    "• «сводка по МЦД-3»\n"
    "• «жалобы на турникеты»\n"
    "• «3 примера про Беговую»\n"
    "• «сколько отзывов за 2024»\n\n"
    "Или используй меню ниже 👇"
)

RESTORED_DEFAULT = (
    "✅ Связь с сервером восстановлена.\n\n"
    "Бот снова отвечает на запросы.\n\n"
    "Попробуйте:\n"
    "• «сводка по МЦД-3»\n"
    "• «3 примера по D4»\n"
    "• «сколько отзывов за 2024»\n\n"
)

BULLET = "🔹"


# ===================== ЛОГИ =====================
def _setup_logger() -> logging.Logger:
    logger = logging.getLogger()
    if getattr(logger, "_inited", False):
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.getenv("LOG_DIR", os.path.join(base_dir, "logs"))
    os.makedirs(log_dir, exist_ok=True)

    fh = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    for noisy in (
        "httpx",
        "telegram",
        "telegram.ext",
        "apscheduler",
        "apscheduler.scheduler",
        "apscheduler.executors.default",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


    logger._inited = True
    return logger


_setup_logger()
log = logging.getLogger("main")


# ===================== DB =====================
def _sqlalchemy_uri_from_env() -> str:
    pg_dsn = (os.getenv("PG_DSN") or "").strip()
    if pg_dsn:
        if pg_dsn.startswith("postgresql://"):
            return "postgresql+psycopg2://" + pg_dsn[len("postgresql://") :]
        return pg_dsn

    user = os.getenv("PG_USER", "postgres")
    pwd = os.getenv("PG_PASSWORD", "")
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DB", "postgres")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_sqlalchemy_uri_from_env(), pool_pre_ping=True)
    return _engine


def ensure_users_table():
    sql_create = f"""
    CREATE TABLE IF NOT EXISTS "{TABLE_USERS}" (
        chat_id    BIGINT PRIMARY KEY,
        user_id    BIGINT,
        username   TEXT,
        first_name TEXT,
        last_name  TEXT,
        subscribed BOOLEAN NOT NULL DEFAULT TRUE,
        authorized BOOLEAN NOT NULL DEFAULT FALSE,
        joined_at  TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
    );
    """
    sql_alter_auth = f'ALTER TABLE "{TABLE_USERS}" ADD COLUMN IF NOT EXISTS authorized BOOLEAN NOT NULL DEFAULT FALSE;'
    sql_alter_topics = f'ALTER TABLE "{TABLE_USERS}" ADD COLUMN IF NOT EXISTS topics TEXT NOT NULL DEFAULT \'[]\';'
    sql_alter_days = f'ALTER TABLE "{TABLE_USERS}" ADD COLUMN IF NOT EXISTS digest_days INTEGER NOT NULL DEFAULT 7;'

    # новое: время и отметка последней отправки
    sql_alter_time = f'ALTER TABLE "{TABLE_USERS}" ADD COLUMN IF NOT EXISTS digest_time TEXT NOT NULL DEFAULT \'09:00\';'
    sql_alter_last = f'ALTER TABLE "{TABLE_USERS}" ADD COLUMN IF NOT EXISTS last_digest_sent DATE;'

    with get_engine().begin() as conn:
        conn.execute(text(sql_create))
        conn.execute(text(sql_alter_auth))
        conn.execute(text(sql_alter_topics))
        conn.execute(text(sql_alter_days))
        conn.execute(text(sql_alter_time))
        conn.execute(text(sql_alter_last))


def upsert_user_from_update(update: Update, subscribed: Optional[bool] = None):
    """
    subscribed:
      None  -> НЕ трогаем текущий флаг subscribed (если пользователь уже есть)
      True/False -> явно выставляем
    """
    u = update.effective_user
    ch = update.effective_chat
    if not u or not ch:
        return

    insert_sub = True if subscribed is None else bool(subscribed)
    update_sub = subscribed  # может быть None

    sql = f"""
    INSERT INTO "{TABLE_USERS}" (chat_id, user_id, username, first_name, last_name, subscribed, joined_at, updated_at)
    VALUES (:chat_id, :user_id, :username, :first_name, :last_name, :insert_sub, NOW(), NOW())
    ON CONFLICT (chat_id) DO UPDATE SET
        user_id    = EXCLUDED.user_id,
        username   = EXCLUDED.username,
        first_name = EXCLUDED.first_name,
        last_name  = EXCLUDED.last_name,
        subscribed = CASE
                       WHEN :update_sub IS NULL THEN "{TABLE_USERS}".subscribed
                       ELSE :update_sub
                     END,
        updated_at = NOW();
    """
    with get_engine().begin() as conn:
        conn.execute(
            text(sql),
            {
                "chat_id": ch.id,
                "user_id": u.id,
                "username": u.username,
                "first_name": u.first_name,
                "last_name": u.last_name,
                "insert_sub": insert_sub,
                "update_sub": update_sub,
            },
        )


def set_subscribed(chat_id: int, subscribed: bool):
    sql = f'UPDATE "{TABLE_USERS}" SET subscribed=:sub, updated_at=NOW() WHERE chat_id=:cid'
    with get_engine().begin() as conn:
        conn.execute(text(sql), {"sub": subscribed, "cid": chat_id})


def set_authorized(chat_id: int, authorized: bool):
    sql = f'UPDATE "{TABLE_USERS}" SET authorized=:auth, updated_at=NOW() WHERE chat_id=:cid'
    with get_engine().begin() as conn:
        conn.execute(text(sql), {"auth": authorized, "cid": chat_id})


def is_authorized(chat_id: int) -> bool:
    sql = f'SELECT authorized FROM "{TABLE_USERS}" WHERE chat_id=:cid'
    with get_engine().begin() as conn:
        row = conn.execute(text(sql), {"cid": chat_id}).fetchone()
        return bool(row[0]) if row else False


def get_all_subscribed_chat_ids() -> List[int]:
    sql = f'SELECT chat_id FROM "{TABLE_USERS}" WHERE subscribed = TRUE'
    with get_engine().begin() as conn:
        rows = conn.execute(text(sql)).fetchall()
        return [r[0] for r in rows]


def _get_user_row(chat_id: int):
    sql = f'SELECT subscribed, authorized, topics, digest_days, digest_time, last_digest_sent FROM "{TABLE_USERS}" WHERE chat_id=:cid'
    with get_engine().begin() as conn:
        return conn.execute(text(sql), {"cid": chat_id}).fetchone()


def is_subscribed(chat_id: int) -> bool:
    row = _get_user_row(chat_id)
    return bool(row[0]) if row else True


def get_topics(chat_id: int) -> List[str]:
    row = _get_user_row(chat_id)
    if not row:
        return []
    raw = row[2] if len(row) >= 3 else "[]"
    try:
        arr = json.loads(raw or "[]")
        out = []
        for x in arr:
            t = str(x).strip()
            if t == "turnstiles":  # миграция старого ключа
                t = "infra"
            if t:
                out.append(t)
        # дедуп
        seen = set()
        res = []
        for t in out:
            if t not in seen:
                seen.add(t)
                res.append(t)
        return res
    except Exception:
        return []


def set_topics(chat_id: int, topics: List[str]):
    topics = [("infra" if t == "turnstiles" else t) for t in topics]
    topics = [t for t in topics if t]
    raw = json.dumps(topics, ensure_ascii=False)

    # тоже сбрасываем last_digest_sent
    sql = f'UPDATE "{TABLE_USERS}" SET topics=:t, last_digest_sent=NULL, updated_at=NOW() WHERE chat_id=:cid'
    with get_engine().begin() as conn:
        conn.execute(text(sql), {"t": raw, "cid": chat_id})



def get_digest_days(chat_id: int) -> int:
    row = _get_user_row(chat_id)
    try:
        return int(row[3]) if row and row[3] is not None else 7
    except Exception:
        return 7


def set_digest_days(chat_id: int, days: int):
    days = int(days)
    sql = f'UPDATE "{TABLE_USERS}" SET digest_days=:d, last_digest_sent=NULL, updated_at=NOW() WHERE chat_id=:cid'
    with get_engine().begin() as conn:
        conn.execute(text(sql), {"d": days, "cid": chat_id})



def get_digest_time(chat_id: int) -> str:
    row = _get_user_row(chat_id)
    try:
        t = str(row[4]) if row and row[4] else "09:00"
    except Exception:
        t = "09:00"
    t = (t or "").strip()
    if not re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", t):
        t = "09:00"
    return t


def set_digest_time(chat_id: int, hhmm: str):
    hhmm = (hhmm or "").strip()
    if not re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", hhmm):
        raise ValueError("bad hhmm")

    # важно: сбрасываем last_digest_sent, чтобы можно было отправить повторно сегодня
    sql = f'UPDATE "{TABLE_USERS}" SET digest_time=:t, last_digest_sent=NULL, updated_at=NOW() WHERE chat_id=:cid'
    with get_engine().begin() as conn:
        conn.execute(text(sql), {"t": hhmm, "cid": chat_id})



def get_last_digest_sent(chat_id: int) -> Optional[dt.date]:
    row = _get_user_row(chat_id)
    try:
        return row[5] if row and len(row) >= 6 else None
    except Exception:
        return None


def set_last_digest_sent(chat_id: int, d: dt.date):
    sql = f'UPDATE "{TABLE_USERS}" SET last_digest_sent=:d, updated_at=NOW() WHERE chat_id=:cid'
    with get_engine().begin() as conn:
        conn.execute(text(sql), {"d": d, "cid": chat_id})

def reset_last_digest_sent(chat_id: int):
    sql = f'UPDATE "{TABLE_USERS}" SET last_digest_sent=NULL, updated_at=NOW() WHERE chat_id=:cid'
    with get_engine().begin() as conn:
        conn.execute(text(sql), {"cid": chat_id})


def get_subscribed_users_settings() -> List[Dict[str, Any]]:
    """
    
    """
    where = "subscribed=TRUE"
    if BOT_PASSWORD:
        # whitelist всегда получает рассылку даже без authorized
        ids = sorted({*ADMINS, *ALLOWED_USERS})
        if ids:
            where += f" AND (authorized=TRUE OR user_id = ANY(ARRAY[{','.join(map(str, ids))}]::bigint[]))"
        else:
            where += " AND authorized=TRUE"

    sql = f"""
    SELECT chat_id, subscribed, authorized, topics, digest_days, digest_time, last_digest_sent
      FROM "{TABLE_USERS}"
     WHERE {where}
    """
    out: List[Dict[str, Any]] = []
    with get_engine().begin() as conn:
        rows = conn.execute(text(sql)).fetchall()
        for r in rows:
            chat_id = int(r[0])
            topics_raw = r[3] or "[]"
            try:
                topics = json.loads(topics_raw) if isinstance(topics_raw, str) else (topics_raw or [])
            except Exception:
                topics = []
            topics = [("infra" if t == "turnstiles" else str(t).strip()) for t in topics if str(t).strip()]
            # дедуп
            seen = set()
            topics2 = []
            for t in topics:
                if t not in seen:
                    seen.add(t)
                    topics2.append(t)

            dd = int(r[4] or 7)
            tt = str(r[5] or "09:00").strip()
            if not re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", tt):
                tt = "09:00"

            out.append(
                {
                    "chat_id": chat_id,
                    "digest_days": dd,
                    "digest_time": tt,
                    "topics": topics2,
                    "last_digest_sent": r[6],
                }
            )
    return out


# ===================== UI / SUBSCRIPTIONS =====================
TOPIC_CHOICES = [
    ("infra", "Пассажирская инфраструктура"),
    ("delays", "Задержки/сбои"),
    ("payment", "Оплата/QR"),
    ("safety", "Безопасность"),
    ("comfort", "Комфорт/чистота"),
]

# --- фильтр по темам для дайджеста ---
TOPIC_PATTERNS = {
    "infra": (
        r"("
        r"турникет\w*|валидатор\w*|"
        r"подземн\w*\s+переход|надземн\w*\s+переход|"
        r"переход\s+закрыт|переход\s+открыт|"
        r"выход\s+закрыт|выход\s+открыт|"
        r"вход\s+закрыт|вход\s+открыт|"
        r"лифт\s+не\s+работа\w*|"
        r"эскалатор\s+не\s+работа\w*|"
        r"лестниц\w*|"
        r"платформ\w*|перрон\w*|"
        r"касс\w*|"
        r"навигац\w*|указател\w*|"
        r"двер\w+\s+не\s+открыва\w*|"
        r"двер\w+\s+не\s+закрыва\w*|"
        r"благоустрой\w*|тпу|"
        r"освещен\w*|освещение|"
        r"пандус\w*"
        r")"
    ),

    "delays": (
        r"("
        r"задерж\w+|опоздан\w+|"
        r"отмен\w+\s+поезд|отмен\w+\s+электричк\w*|"
        r"сбой\s+движени\w*|сбои\s+движени\w*|"
        r"увеличен\w+\s+интервал\w*|"
        r"интервалы?\s+увеличен\w*|"
        r"поезд\s+не\s+пришел|"
        r"не\s+пришла|не\s+пришел|"
        r"сокращенн\w+\s+маршрут\w*|"
        r"движени\w+\s+остановлен\w*|"
        r"нет\s+движени\w*|"
        r"не\s+ходит|"
        r"по\s+расписанию\s+не\s+ид[её]т|"
        r"отмен[её]н\w*"
        r")"
    ),

    "payment": (
        r"("
        r"не\s+проходит\s+оплата|"
        r"не\s+работает\s+оплата|"
        r"оплата\s+не\s+проходит|"
        r"оплата\s+не\s+работает|"
        r"qr|сбп|"
        r"тройк\w*|карта\s+тройк\w*|виртуальн\w+\s+тройк\w*|"
        r"валидатор\w*|"
        r"турникет\w+\s+не\s+пуска\w*|"
        r"не\s+считыва\w+\s+карт\w*|"
        r"не\s+чита\w+\s+карт\w*|"
        r"списан\w*|"
        r"двойн\w+\s+списан\w*|"
        r"пополн\w+\s+карт\w*|"
        r"терминал\s+оплаты|"
        r"стоп[\-\s]?лист|"
        r"проездн\w*\s+билет|"
        r"купить\s+билет|"
        r"билет\s+не\s+считыва\w*|"
        r"билет\s+не\s+работа\w*|"
        r"контрол[её]р\w*\s+заблокир\w*"
        r")"
    ),

    "safety": (
        r"("
        r"зацепер\w*|зацепинг\w*|"
        r"травм\w*|"
        r"упал\w*\s+на\s+пут\w*|"
        r"погиб\w*|гибел\w*|"
        r"драк\w*|"
        r"агресси\w*|"
        r"хулиган\w*|"
        r"вандал\w*|"
        r"полици\w*|"
        r"напал\w*|"
        r"угроза\s+безопасности|"
        r"подозрительн\w+\s+лиц\w*|"
        r"краж\w*|украл\w*|"
        r"нож\w*|"
        r"опасн\w+\s+поведени\w*"
        r")"
    ),

    "comfort": (
        r"("
        r"гряз\w+\s+вагон\w*|"
        r"грязн\w+\s+вагон\w*|"
        r"грязн\w+\s+платформ\w*|"
        r"запах\w*|гар\w*|вон\w*|"
        r"жарк\w+\s+в\s+вагон\w*|"
        r"холод\w+\s+в\s+вагон\w*|"
        r"кондиционер\w*\s+не\s+работа\w*|"
        r"печк\w*\s+не\s+работа\w*|"
        r"душно|"
        r"теснот\w*|"
        r"давк\w*|"
        r"толп\w*|"
        r"неудобн\w+\s+сиден\w*|"
        r"грязно|"
        r"антисанитар\w*|"
        r"мусор\w*"
        r")"
    ),
}

INFRA_STRICT_RE = re.compile(
    r"(турникет\w*|валидатор\w*|"
    r"подземн\w*\s+переход|надземн\w*\s+переход|"
    r"переход\s+закрыт|переход\s+открыт|"
    r"выход\s+закрыт|выход\s+открыт|"
    r"вход\s+закрыт|вход\s+открыт|"
    r"лифт\s+не\s+работа\w*|"
    r"эскалатор\s+не\s+работа\w*|"
    r"лестниц\w*|"
    r"платформ\w*|перрон\w*|"
    r"касс\w*|"
    r"навигац\w*|указател\w*|"
    r"двер\w+\s+не\s+открыва\w*|"
    r"двер\w+\s+не\s+закрыва\w*|"
    r"благоустрой\w*|тпу|"
    r"освещен\w*|освещение|"
    r"пандус\w*)",
    re.IGNORECASE
)

INFRA_NEGATIVE_RE = re.compile(
    r"(военкомат|военн\w+\s+билет|"
    r"паспорт\w*|"
    r"найден\w*|потеря\w*|утеря\w*|"
    r"медкомисси\w*|"
    r"юмор\w*|шутк\w*|мем\w*|"
    r"реклам\w*)",
    re.IGNORECASE
)

DELAYS_STRICT_RE = re.compile(
    r"(задерж\w+|опоздан\w+|"
    r"отмен\w+\s+поезд|отмен\w+\s+электричк\w*|"
    r"сбой\s+движени\w*|сбои\s+движени\w*|"
    r"увеличен\w+\s+интервал\w*|"
    r"интервалы?\s+увеличен\w*|"
    r"поезд\s+не\s+пришел|не\s+пришла|не\s+пришел|"
    r"сокращенн\w+\s+маршрут\w*|"
    r"движени\w+\s+остановлен\w*|"
    r"нет\s+движени\w*|"
    r"не\s+ходит|"
    r"по\s+расписанию\s+не\s+ид[её]т|"
    r"отмен[её]н\w*)",
    re.IGNORECASE
)

DELAYS_NEGATIVE_RE = re.compile(
    r"(задержан\w+\s+полици\w*|"
    r"задержан\w+\s+человек\w*|"
    r"оплата|валидатор|тройк\w*|"
    r"гряз\w*|вон\w*|гар\w*|"
    r"военн\w+\s+билет|паспорт\w*)",
    re.IGNORECASE
)

PAYMENT_STRICT_RE = re.compile(
    r"(не\s+проходит\s+оплата|"
    r"не\s+работает\s+оплата|"
    r"оплата\s+не\s+проходит|"
    r"оплата\s+не\s+работает|"
    r"валидатор\w*|"
    r"турникет\w+\s+не\s+пуска\w*|"
    r"qr|сбп|"
    r"тройк\w*|карта\s+тройк\w*|виртуальн\w+\s+тройк\w*|"
    r"не\s+считыва\w+\s+карт\w*|"
    r"не\s+чита\w+\s+карт\w*|"
    r"списан\w*|"
    r"двойн\w+\s+списан\w*|"
    r"пополн\w+\s+карт\w*|"
    r"терминал\s+оплаты|"
    r"стоп[\-\s]?лист|"
    r"проездн\w*\s+билет|"
    r"купить\s+билет|"
    r"билет\s+не\s+считыва\w*|"
    r"билет\s+не\s+работа\w*|"
    r"контрол[её]р\w*\s+заблокир\w*)",
    re.IGNORECASE
)

PAYMENT_NEGATIVE_RE = re.compile(
    r"(военн\w+\s+билет|"
    r"найден\w*|потеря\w*|утеря\w*|"
    r"расписан\w*|интервал\w*|"
    r"отмен\w+\s+поезд|задержк\w+\s+поезд|"
    r"травм\w*|зацепер\w*|пожар\w*|"
    r"вандал\w*|погиб\w*|"
    r"юмор\w*|шутк\w*)",
    re.IGNORECASE
)

SAFETY_STRICT_RE = re.compile(
    r"(зацепер\w*|зацепинг\w*|"
    r"травм\w*|"
    r"упал\w*\s+на\s+пут\w*|"
    r"погиб\w*|гибел\w*|"
    r"драк\w*|"
    r"агресси\w*|"
    r"хулиган\w*|"
    r"вандал\w*|"
    r"полици\w*|"
    r"напал\w*|"
    r"угроза\s+безопасности|"
    r"подозрительн\w+\s+лиц\w*|"
    r"краж\w*|украл\w*|"
    r"нож\w*|"
    r"опасн\w+\s+поведени\w*)",
    re.IGNORECASE
)

SAFETY_NEGATIVE_RE = re.compile(
    r"(оплата|валидатор|тройк\w*|"
    r"гряз\w*|гар\w*|вон\w*|"
    r"задержк\w+\s+поезд|отмен\w+\s+поезд|"
    r"военн\w+\s+билет|паспорт\w*|"
    r"реклам\w*)",
    re.IGNORECASE
)

COMFORT_STRICT_RE = re.compile(
    r"(гряз\w+\s+вагон\w*|"
    r"грязн\w+\s+вагон\w*|"
    r"грязн\w+\s+платформ\w*|"
    r"запах\w*|гар\w*|вон\w*|"
    r"жарк\w+\s+в\s+вагон\w*|"
    r"холод\w+\s+в\s+вагон\w*|"
    r"кондиционер\w*\s+не\s+работа\w*|"
    r"печк\w*\s+не\s+работа\w*|"
    r"душно|"
    r"теснот\w*|"
    r"давк\w*|"
    r"толп\w*|"
    r"неудобн\w+\s+сиден\w*|"
    r"грязно|"
    r"антисанитар\w*|"
    r"мусор\w*)",
    re.IGNORECASE
)

COMFORT_NEGATIVE_RE = re.compile(
    r"(оплата|валидатор|тройк\w*|"
    r"военн\w+\s+билет|паспорт\w*|"
    r"полици\w*|краж\w*|зацепер\w*|"
    r"отмен\w+\s+поезд|задержк\w+\s+поезд)",
    re.IGNORECASE
)

def _topic_title(key: str) -> str:
    m = dict(TOPIC_CHOICES)
    return m.get(key, key)

def _safe_topic_basename(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\-_а-яА-ЯёЁ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "topic"

def _topic_doc_recheck(topic_key: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Повторная проверка темы уже по агрегированному документу.
    Нужна, чтобы после сборки thread/doc убрать шумные попадания.
    """
    if df is None or df.empty:
        return df

    work = df.copy()

    text_series = (
        work.get("full_text", "").astype(str).fillna("") + "\n" +
        work.get("body_text", "").astype(str).fillna("") + "\n" +
        work.get("root_text", "").astype(str).fillna("")
    )

    rules = {
        "infra": (INFRA_STRICT_RE, INFRA_NEGATIVE_RE),
        "delays": (DELAYS_STRICT_RE, DELAYS_NEGATIVE_RE),
        "payment": (PAYMENT_STRICT_RE, PAYMENT_NEGATIVE_RE),
        "safety": (SAFETY_STRICT_RE, SAFETY_NEGATIVE_RE),
        "comfort": (COMFORT_STRICT_RE, COMFORT_NEGATIVE_RE),
    }

    pair = rules.get(topic_key)
    if not pair:
        return work

    strict_re, negative_re = pair

    strong_mask = text_series.str.contains(strict_re, na=False, regex=True)
    negative_mask = text_series.str.contains(negative_re, na=False, regex=True)

    mask = strong_mask & ~negative_mask
    filtered = work[mask].copy()

    # fallback: если всё отфильтровалось, лучше вернуть исходное,
    # чем получить пустую тему
    return filtered if not filtered.empty else work

def _build_digest_html(chat_id: int) -> tuple[str, list[tuple[str, str]]]:
    """
    Возвращает:
    - HTML дайджеста
    - список файлов [(topic_key, export_path), ...]
    """
    days = get_digest_days(chat_id)
    topics = get_topics(chat_id)
    if not topics:
        topics = [k for k, _ in TOPIC_CHOICES]

    df = apg.load_data()
    if df is None or df.empty:
        return "<b>🗞 Дайджест</b>\n\nНет данных.", []

    now_utc = pd.Timestamp.now(tz="UTC")
    cutoff = now_utc - pd.Timedelta(days=int(days))

    if "msg_date" in df.columns:
        df = df[df["msg_date"] >= cutoff].copy()

    if df.empty:
        return f"<b>🗞 Дайджест за последние {days} дн.</b>\n\nНет свежих сообщений.", []

    if "_concat_txt" in df.columns:
        txt = df["_concat_txt"]
    else:
        txt = (df.get("full_text", "").astype(str) + " " + df.get("parent_text", "").astype(str)).str.lower()

    blocks = [f"<b>🗞 Дайджест за последние {days} дн.</b>"]
    export_files: list[tuple[str, str]] = []

    for t in topics:
        pat = TOPIC_PATTERNS.get(t)
        if not pat:
            continue

        sub = df[txt.str.contains(pat, na=False, regex=True)].copy()
        if sub.empty:
            continue

        if len(sub) > DIGEST_MAX_ROWS:
            sub = sub.head(DIGEST_MAX_ROWS).copy()

        title = _topic_title(t)

        # Вместо сырого набора строк собираем компактные docs:
        # один корень + агрегированное обсуждение.
        try:
            digest_docs = apg.build_digest_topic_docs(sub, topic_key=t, topic_title=title)
        except Exception as e:
            log.warning("[DIGEST_DOCS] failed topic=%s: %s", t, e)
            digest_docs = sub.copy()
            digest_docs["digest_topic_key"] = t
            digest_docs["digest_topic_title"] = title

        try:
            digest_docs = _topic_doc_recheck(t, digest_docs)
        except Exception as e:
            log.warning("[DIGEST_RECHECK] failed topic=%s: %s", t, e)

        # Export именно compact docs, а не raw rows
        if DIGEST_SEND_TOPIC_EXPORTS:
            try:
                exp = digest_docs.head(DIGEST_TOPIC_EXPORT_MAX_ROWS).copy()
                basename = f"digest_{days}d_{_safe_topic_basename(t)}_{_safe_topic_basename(title)}"
                export_path = apg._export_df(exp, basename)
                if export_path:
                    export_files.append((t, export_path))
                    log.info("[DIGEST_EXPORT] topic=%s rows=%d file=%s", t, len(exp), export_path)
            except Exception as e:
                log.warning("[DIGEST_EXPORT] failed topic=%s: %s", t, e)

        # Summary тоже строим по compact docs
        try:
            sub2 = apg._make_summary_text(digest_docs)
            text_col = "summary_text"
        except Exception:
            sub2 = digest_docs
            text_col = "full_text"

        if getattr(apg, "summarize_topn", None) is not None:
            summ = apg.summarize_topn(
                df=sub2,
                query_text=f"дайджест: {title}",
                text_col=text_col,
                datetime_col="msg_date",
                top_n=DIGEST_TOP_N,
            )

            try:
                summ = apg.enrich_summary_with_geo(summ, sub2)
            except Exception as e:
                log.warning("[DIGEST_GEO] enrich failed topic=%s: %s", t, e)
        else:
            summ = f"{BULLET} сообщений: {len(digest_docs)}"

        html_bullets = _to_html_bullets(summ)
        html_bullets = _limit_html_bullets(html_bullets, DIGEST_MAX_BULLETS_PER_TOPIC)

        if html_bullets.strip():
            blocks.append(f"\n<b>{html.escape(title)}</b>\n{html_bullets}")

    html_text = "\n".join(blocks).strip()

    
    if len(html_text) > DIGEST_MAX_HTML_CHARS:
        safe_blocks = []
        current_len = 0

        for block in blocks:
            candidate = (("\n".join(safe_blocks + [block])).strip())
            if len(candidate) > DIGEST_MAX_HTML_CHARS:
                break
            safe_blocks.append(block)
            current_len = len(candidate)

        if safe_blocks:
            html_text = "\n".join(safe_blocks).strip()
            if len(safe_blocks) < len(blocks):
                html_text += "\n\n…"
        else:
            html_text = "<b>🗞 Дайджест</b>\n\nСлишком большой объём текста, сокращён."

    return html_text, export_files
    



DIGEST_DAY_CHOICES = [
    (1, "За 1 день"),
    (3, "За 3 дня"),
    (7, "За 7 дней"),
    (30, "За 30 дней"),
]


def _main_menu_kb(chat_id: int) -> InlineKeyboardMarkup:
    sub = is_subscribed(chat_id)
    sub_label = "🔔 Подписка: Вкл" if sub else "🔕 Подписка: Выкл"

    kb = [
        [
            InlineKeyboardButton("🧾 Статус подписки", callback_data="menu:status"),
            InlineKeyboardButton(sub_label, callback_data="menu:toggle_sub"),
        ],
        [
            InlineKeyboardButton("📌 Вывод сводки по МЦД", callback_data="menu:mcd_summary"),
            InlineKeyboardButton("⏱ Период и время рассылки", callback_data="menu:schedule"),
        ],
        [
            InlineKeyboardButton("📝 Ввести свободный запрос", callback_data="menu:free_query"),
            InlineKeyboardButton("⚙️ Темы подписки", callback_data="menu:topics"),
        ],
    ]
    return InlineKeyboardMarkup(kb)


def _schedule_kb() -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton("⏱ Период рассылки", callback_data="menu:period")],
        [InlineKeyboardButton("🕘 Время рассылки", callback_data="menu:time")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="menu:back")],
    ]
    return InlineKeyboardMarkup(kb)


def _mcd_summary_kb() -> InlineKeyboardMarkup:
    kb = [
        [InlineKeyboardButton("🚆 Все МЦД (1–4)", callback_data="q:summary:мцд_all")],
        [
            InlineKeyboardButton("МЦД-1", callback_data="q:summary:мцд-1"),
            InlineKeyboardButton("МЦД-2", callback_data="q:summary:мцд-2"),
        ],
        [
            InlineKeyboardButton("МЦД-3", callback_data="q:summary:мцд-3"),
            InlineKeyboardButton("МЦД-4", callback_data="q:summary:мцд-4"),
        ],
        [InlineKeyboardButton("⬅️ Назад", callback_data="menu:back")],
    ]
    return InlineKeyboardMarkup(kb)


def _topics_kb(chat_id: int) -> InlineKeyboardMarkup:
    picked = set(get_topics(chat_id))

    all_keys = [k for k, _ in TOPIC_CHOICES]
    all_on = all(k in picked for k in all_keys)
    toggle_all_label = "🟢 Снять всё" if all_on else "🟣 Выбрать всё"

    rows = []
    rows.append([InlineKeyboardButton(toggle_all_label, callback_data="topic:toggle_all")])

    for key, title in TOPIC_CHOICES:
        mark = "✅ " if key in picked else "☑️ "
        rows.append([InlineKeyboardButton(mark + title, callback_data=f"topic:toggle:{key}")])

    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu:back")])
    return InlineKeyboardMarkup(rows)


def _period_kb(chat_id: int) -> InlineKeyboardMarkup:
    cur = get_digest_days(chat_id)
    rows = []
    for d, label in DIGEST_DAY_CHOICES:
        mark = "✅ " if d == cur else "☑️ "
        rows.append([InlineKeyboardButton(mark + label, callback_data=f"period:set:{d}")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu:schedule")])
    return InlineKeyboardMarkup(rows)


def _time_kb(chat_id: int) -> InlineKeyboardMarkup:
    cur = get_digest_time(chat_id)
    options = ["08:00", "09:00", "10:00", "12:00", "18:00", "21:00"]
    rows = []
    for t in options:
        mark = "✅ " if t == cur else "☑️ "
        rows.append([InlineKeyboardButton(mark + t, callback_data=f"time:set:{t}")])
    rows.append([InlineKeyboardButton("✍️ Ввести вручную (HH:MM)", callback_data="time:manual")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu:schedule")])
    return InlineKeyboardMarkup(rows)


def _render_status(chat_id: int) -> str:
    sub = is_subscribed(chat_id)
    days = get_digest_days(chat_id)
    hhmm = get_digest_time(chat_id)
    topics = get_topics(chat_id)

    m = dict(TOPIC_CHOICES)
    tnames = [m.get(t, t) for t in topics]
    topics_txt = ", ".join(tnames) if tnames else "не выбраны"

    return (
        "<b>🧾 Статус подписки</b>\n\n"
        f"🔹 Подписка: <b>{'ВКЛ' if sub else 'ВЫКЛ'}</b>\n"
        f"🔹 Период дайджеста: <b>последние {days} дн.</b>\n"
        f"🔹 Время дайджеста: <b>{html.escape(hhmm)}</b> (TZ: {html.escape(DIGEST_TZ)})\n"
        f"🔹 Темы: <b>{html.escape(topics_txt)}</b>\n\n"
        "Используй кнопки ниже для настройки."
    )


# ===================== ДОСТУП / УТИЛИТЫ =====================
def _is_admin(update: Update) -> bool:
    try:
        uid = update.effective_user.id if update.effective_user else None
        return uid in ADMINS
    except Exception:
        return False


def _extract_payload(update: Update, command_names: List[str]) -> str:
    txt = update.message.text or ""
    for name in command_names:
        m = re.match(rf"^/{name}(?:@\w+)?\s*(.*)$", txt, flags=re.S)
        if m:
            return (m.group(1) or "")
    return ""


def _allowed(update: Update) -> bool:
    """Разрешён ли пользователю доступ к аналитике."""
    try:
        uid = update.effective_user.id if update.effective_user else None
        if uid in ADMINS or uid in ALLOWED_USERS:
            return True

        if not BOT_PASSWORD:
            return False

        ch = update.effective_chat
        if not ch:
            return False

        ensure_users_table()
        return is_authorized(ch.id)
    except Exception:
        return False


class _SeenUpdates(OrderedDict):
    def __init__(self, maxlen=8192, ttl=60):
        super().__init__()
        self.maxlen = maxlen
        self.ttl = ttl

    def push(self, key) -> bool:
        now = time.time()
        for k in list(self.keys()):
            if now - self[k] > self.ttl:
                del self[k]
        if key in self:
            return False
        self[key] = now
        if len(self) > self.maxlen:
            self.popitem(last=False)
        return True


_SEEN_UPDATES = _SeenUpdates(maxlen=8192, ttl=60)


async def _typing_indicator(context: ContextTypes.DEFAULT_TYPE, chat_id: int, stop_event: asyncio.Event):
    """Периодически шлёт 'печатает...' пока не будет установлен stop_event."""
    try:
        while not stop_event.is_set():
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(6.0)
    except Exception as e:
        logging.warning("typing indicator failed: %s", e)


def _normalize_abbreviations(s: str) -> str:
    if not s:
        return s
    s = re.sub(r"\bWi\s*:\s*Fi\b", "Wi-Fi", s, flags=re.I)
    s = re.sub(r"\bQR\s*:\s*код(ами|ов|ы|у|ом)?\b", r"QR-код\1", s, flags=re.I)
    s = re.sub(r"\bQR\s*:\s*кодов\b", "QR-кодов", s, flags=re.I)
    s = re.sub(r"\bQR\s*:\s*", "QR-", s, flags=re.I)
    s = re.sub(r"\bМЦД\s*:\s*([1-4])\b", r"МЦД-\1", s, flags=re.I)
    s = re.sub(r"\bD\s*:\s*([1-4])\b", r"D\1", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def _looks_like_time_around(s: str, idx: int) -> bool:
    left = s[max(0, idx - 2) : idx]
    right = s[idx + 1 : idx + 3]
    return left.strip().isdigit() and right.strip().isdigit()


def _is_bad_title_fragment(t: str) -> bool:
    t_norm = re.sub(r"\s+", " ", (t or "").strip().lower())
    if t_norm in {"текст", "случай", "пример", "заголовок"}:
        return True
    if re.fullmatch(r"[a-z]{1,3}\d?", t_norm):
        return True
    if re.fullmatch(r"d[1-4]", t_norm):
        return True
    if re.fullmatch(r"мцд\-?[1-4]", t_norm):
        return False
    if len(t_norm) <= 2:
        return True
    return False


def _split_title_body(line_no_bullet: str) -> Optional[Tuple[str, str]]:
    s = _normalize_abbreviations((line_no_bullet or "").strip())
    idx = s.find(":")
    if idx == -1:
        parts = re.split(r"\s[—\-]\s", s, maxsplit=1)
        if len(parts) == 2:
            s = f"{parts[0]}: {parts[1]}"
            idx = s.find(":")
        else:
            return None
    if _looks_like_time_around(s, idx):
        return None
    title, body = s[:idx].strip(), s[idx + 1 :].strip()
    if _is_bad_title_fragment(title):
        return None
    return title, body


def _to_html_bullets(text_out: str) -> str:
    lines = text_out.splitlines()
    html_lines = []
    for ln in lines:
        raw = ln.strip()
        raw = re.sub(r"^(?:[🔹•\-\–\—\*]\s*)+", "", raw)
        if not raw:
            html_lines.append("")
            continue
        raw = _normalize_abbreviations(raw)
        sb = _split_title_body(raw)
        if sb:
            title, body = sb
            title = html.escape(title)
            body = html.escape(body)
            html_lines.append(f"{BULLET} <b>{title}</b>: {body}")
        else:
            html_lines.append(f"{BULLET} {html.escape(raw)}")
    return "\n".join(html_lines).strip()

def _limit_html_bullets(html_text: str, max_bullets: int = 4) -> str:
    """
    Оставляет только первые N непустых bullet-строк.
    """
    if not html_text:
        return html_text

    lines = [ln for ln in html_text.splitlines() if ln.strip()]
    out = []
    bullets = 0

    for ln in lines:
        s = ln.strip()
        if s.startswith(BULLET):
            if bullets >= max_bullets:
                continue
            bullets += 1
        out.append(ln)

    return "\n".join(out).strip()


def _trim_digest_html(text: str, limit: int = 3500) -> str:
    """
    Грубая, но безопасная подрезка HTML дайджеста до лимита Telegram.
    """
    if not text:
        return text
    if len(text) <= limit:
        return text

    cut = text[:limit].rstrip()

    # пытаемся не оборвать на середине HTML-тега
    last_lt = cut.rfind("<")
    last_gt = cut.rfind(">")
    if last_lt > last_gt:
        cut = cut[:last_lt].rstrip()

    return cut + "\n\n…"

def _build_final_html(result_text: str, header: str = "✅ Ответ") -> str:
    lines = (result_text or "").splitlines()
    body = _to_html_bullets("\n".join(lines))
    full = f"<b>{html.escape(header)}</b>\n\n{body}".strip()
    if len(full) > 3900:
        parts = full.splitlines()
        cur = ""
        for p in parts:
            if len(cur) + len(p) + 1 <= 3800:
                cur += (("\n" if cur else "") + p)
            else:
                break
        full = cur + "\n\n…(урезано для Telegram)"
    return full

async def _safe_edit_message_text(q, text: str, **kwargs):
    try:
        await q.edit_message_text(text, **kwargs)
    except BadRequest as e:
        if "Message is not modified" in str(e):
            return
        raise


async def _send_file_if_any(context: ContextTypes.DEFAULT_TYPE, chat_id: int, path: Optional[str], caption: Optional[str] = None):
    if not path:
        return
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_DOCUMENT)
        with open(path, "rb") as f:
            await context.bot.send_document(
                chat_id=chat_id,
                document=f,
                filename=pathlib.Path(path).name,
                caption=caption or "",
            )
    except Exception as e:
        logging.warning("Не удалось отправить файл: %s", e)


# ===================== ДАЙДЖЕСТ (РАССЫЛКА) =====================


_TOPIC_TITLES = dict(TOPIC_CHOICES)


def _now_in_tz() -> dt.datetime:
    try:
        return dt.datetime.now(ZoneInfo(DIGEST_TZ))
    except Exception:
        return dt.datetime.now()




async def _digest_tick(context: ContextTypes.DEFAULT_TYPE):
    """
    Раз в минуту проверяем: если у пользователя сейчас его HH:MM и сегодня ещё не отправляли — отправляем.
    """
    try:
        users = get_subscribed_users_settings()
        if not users:
            return

        now_local = _now_in_tz()
        hhmm_now = now_local.strftime("%H:%M")
        today = now_local.date()

        for u in users:
            chat_id = int(u["chat_id"])
            hhmm = str(u.get("digest_time") or "09:00").strip()

            if hhmm != hhmm_now:
                continue

            last_sent = u.get("last_digest_sent")
            days = int(u.get("digest_days") or 7)

            if isinstance(last_sent, dt.date):
                # если уже отправляли недавно — ждём N дней
                if (today - last_sent).days < max(1, days):
                    continue


            # чтобы было видно, что бот “жив”
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception:
                pass
            log.info(
                "[DIGEST] try_send chat_id=%s hhmm=%s now=%s days=%s topics=%s",
                chat_id, hhmm, hhmm_now, u.get("digest_days"), u.get("topics")
            )

            try:
                digest_html, export_files = _build_digest_html(chat_id)
                _log_digest_text(digest_html)

                await context.bot.send_message(
                    chat_id=chat_id,
                    text=digest_html,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )

                for topic_key, export_path in export_files:
                    await _send_file_if_any(
                        context,
                        chat_id,
                        export_path,
                        caption=f"📎 {html.unescape(_topic_title(topic_key))} — комментарии, попавшие в дайджест"
                    )

                set_last_digest_sent(chat_id, today)

                await context.bot.send_message(
                    chat_id=chat_id,
                    text="✅ Меню",
                    reply_markup=_main_menu_kb(chat_id),
                )

                log.info("[DIGEST] sent_ok chat_id=%s date=%s", chat_id, today)
            except Exception as e:
                logging.warning("Digest send failed to %s: %s", chat_id, e)

            await asyncio.sleep(SLEEP_BETWEEN_SENDS)

    except Exception as e:
        logging.warning("digest tick error: %s", e)


def _log_digest_text(text: str, limit: int = 4000) -> None:
    try:
        s = (text or "").strip()
        if not s:
            log.info("[DIGEST_TEXT] <empty>")
            return
        if len(s) > limit:
            log.info("[DIGEST_TEXT] %s ... [truncated %d chars]", s[:limit], len(s) - limit)
        else:
            log.info("[DIGEST_TEXT] %s", s)
    except Exception as e:
        log.warning("[DIGEST_TEXT] log failed: %s", e)

# ===================== HANDLERS =====================
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_users_table()
    if not update.message:
        return

    text_msg = (update.message.text or "").strip()
    chat = update.effective_chat

    # анти-дубль
    key = (chat.id if chat else 0, update.effective_message.message_id or 0, text_msg)
    if not _SEEN_UPDATES.push(key):
        return

    # пароль как сообщение
    if BOT_PASSWORD and not _allowed(update) and text_msg == BOT_PASSWORD:
        if chat:
            upsert_user_from_update(update)
            set_authorized(chat.id, True)
        await update.message.reply_text("✅ Пароль принят. Доступ к боту открыт.\n\n" + WELCOME_TEXT, reply_markup=_main_menu_kb(chat.id))
        return

    # проверка доступа
    if not _allowed(update):
        if BOT_PASSWORD:
            await update.message.reply_text(
                "🔐 Этот бот доступен по паролю.\nПожалуйста введите пароль.",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.message.reply_text("Доступ запрещён.")
        return

    # обновим инфу о пользователе
    upsert_user_from_update(update, subscribed=True)

    # --- режим ручного ввода времени рассылки ---
    if context.user_data.get("awaiting_digest_time"):
        m = re.fullmatch(r"\s*([01]\d|2[0-3]):([0-5]\d)\s*", text_msg)
        if not m:
            await update.message.reply_text("❌ Неверный формат. Введи HH:MM, например 09:30.")
            return
        hhmm = f"{m.group(1)}:{m.group(2)}"
        try:
            set_digest_time(chat.id, hhmm)
        except Exception:
            await update.message.reply_text("❌ Не удалось сохранить время. Попробуй ещё раз.")
            return
        context.user_data["awaiting_digest_time"] = False
        await update.message.reply_text(f"✅ Время рассылки установлено: <b>{hhmm}</b>", parse_mode=ParseMode.HTML, reply_markup=_main_menu_kb(chat.id))
        return

    q = text_msg
    log.info("[QUERY] %s (by %s)", q, update.effective_user.id if update.effective_user else "?")

    progress_msg = await update.message.reply_text("⏳ Обрабатываю запрос…")
    chat_id = chat.id if chat else None

    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(_typing_indicator(context, chat_id, stop_event)) if chat_id else None

    try:
        text_result, export_path = await asyncio.to_thread(apg.process_query, q)
        html_text = _build_final_html(text_result or "⚠️ Ничего не найдено.")

        try:
            await progress_msg.edit_text(
                html_text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=_main_menu_kb(chat_id),
            )
        except Exception:
            await update.message.reply_text(
                html_text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_markup=_main_menu_kb(chat_id),
            )

        if export_path and chat_id is not None:
            await _send_file_if_any(context, chat_id, export_path, caption="📎 Отфильтрованные строки")

    except Exception as e:
        logging.exception("Ошибка в обработчике: %s", e)
        try:
            await progress_msg.edit_text("⚠️ Ошибка при обработке запроса.")
        except Exception:
            await update.message.reply_text("⚠️ Ошибка при обработке запроса.")
    finally:
        if typing_task is not None:
            stop_event.set()
            try:
                await typing_task
            except Exception:
                pass


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return
    

    chat = update.effective_chat
    if not chat:
        return

    ensure_users_table()
    upsert_user_from_update(update)

    if BOT_PASSWORD and not _allowed(update):
        await q.edit_message_text("🔐 Доступ по паролю. Введите пароль сообщением или /login <пароль>")
        return

    data = q.data or ""

    # -------- меню --------
    if data == "menu:back":
        await q.edit_message_text("✅ Меню", reply_markup=_main_menu_kb(chat.id))
        return

    if data == "menu:help":
        await q.edit_message_text(WELCOME_TEXT, reply_markup=_main_menu_kb(chat.id))
        return

    if data == "menu:status":
        await q.edit_message_text(_render_status(chat.id), parse_mode=ParseMode.HTML, reply_markup=_main_menu_kb(chat.id))
        return

    if data == "menu:toggle_sub":
        new_val = not is_subscribed(chat.id)
        set_subscribed(chat.id, new_val)
        await q.answer("Подписка включена ✅" if new_val else "Подписка отключена 🔕", show_alert=False)
        await q.edit_message_text(_render_status(chat.id), parse_mode=ParseMode.HTML, reply_markup=_main_menu_kb(chat.id))
        return

    if data == "menu:topics":
        await q.edit_message_text("⚙️ Темы подписки — выбери, что включать в дайджест:", reply_markup=_topics_kb(chat.id))
        return

    if data == "menu:schedule":
        await q.edit_message_text("⏱ Настройка рассылки:", reply_markup=_schedule_kb())
        return

    if data == "menu:period":
        await q.edit_message_text("⏱ Период дайджеста — за сколько дней собирать сводку:", reply_markup=_period_kb(chat.id))
        return

    if data == "menu:time":
        await q.edit_message_text("🕘 Время дайджеста — выбери или введи вручную:", reply_markup=_time_kb(chat.id))
        return

    if data == "menu:mcd_summary":
        await q.edit_message_text("📌 Сводка по МЦД — выбери вариант:", reply_markup=_mcd_summary_kb())
        return

    if data == "menu:free_query":
        txt = (
            "<b>📝 Свободный запрос</b>\n\n"
            "Просто напиши сообщение текстом — бот сам определит режим:\n"
            f"{BULLET} <b>Сводка</b>: «сводка по мцд-3 за январь 2026»\n"
            f"{BULLET} <b>Примеры</b>: «покажи примеры про турникеты на мцд-4»\n"
            f"{BULLET} <b>Сколько</b>: «сколько сообщений про оплату за 2025»\n\n"
            "Можно упоминать станции и месяцы."
        )
        await q.edit_message_text(txt, parse_mode=ParseMode.HTML, reply_markup=_main_menu_kb(chat.id))
        return

    # -------- темы --------
    if data == "topic:toggle_all":
        cur = set(get_topics(chat.id))
        all_keys = [k for k, _ in TOPIC_CHOICES]
        all_on = all(k in cur for k in all_keys)
        if all_on:
            set_topics(chat.id, [])
            await q.answer("Снял все темы")
        else:
            set_topics(chat.id, all_keys)
            await q.answer("Выбрал все темы")
        await q.edit_message_reply_markup(reply_markup=_topics_kb(chat.id))
        return
    
    if data.startswith("topic:toggle:"):
        key = data.split(":", 2)[2].strip()
        all_keys = [k for k, _ in TOPIC_CHOICES]
        if key not in all_keys:
            await q.answer("Неизвестная тема", show_alert=False)
            return

        cur = set(get_topics(chat.id))
        if key in cur:
            cur.remove(key)
            await q.answer("Тема выключена")
        else:
            cur.add(key)
            await q.answer("Тема включена")

        set_topics(chat.id, sorted(cur))
        await q.edit_message_reply_markup(reply_markup=_topics_kb(chat.id))
        return

    # -------- быстрые запросы --------
    if data.startswith("q:"):
        _, intent, payload = data.split(":", 2)

        if intent == "summary":
            if payload == "мцд_all":
                user_q = "сводка по мцд-1 мцд-2 мцд-3 мцд-4"
            else:
                user_q = f"сводка по {payload}"
        elif intent == "count":
            user_q = f"сколько сообщений про {payload}"
        elif intent == "examples":
            user_q = f"покажи 3 примера про {payload}"
        else:
            user_q = payload

        await q.answer("⏳ Обрабатываю…", show_alert=False)

        try:
            text_result, export_path = await asyncio.to_thread(apg.process_query, user_q)
            html_text = _build_final_html(text_result or "⚠️ Ничего не найдено.")

            # 1) ответ — отдельным сообщением (не редактируем меню)
            await context.bot.send_message(
                chat_id=chat.id,
                text=html_text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )

            # 2) файл — следом
            if export_path:
                await _send_file_if_any(context, chat.id, export_path, caption="📎 Отфильтрованные строки")

            # 3) старое меню удаляем (если получится)
            try:
                if q.message:
                    await q.message.delete()
            except Exception:
                pass

            # 4) и шлём свежее меню вниз
            await context.bot.send_message(chat_id=chat.id, text="✅ Меню", reply_markup=_main_menu_kb(chat.id))

        except Exception as e:
            logging.exception("Ошибка кнопки: %s", e)
            await context.bot.send_message(chat_id=chat.id, text="⚠️ Ошибка при обработке.", reply_markup=_main_menu_kb(chat.id))
        return


    # -------- период --------
    if data.startswith("period:set:"):
        d = int(data[len("period:set:"):])
        set_digest_days(chat.id, d)
        await q.answer(f"Период: последние {d} дн.")
        await q.edit_message_reply_markup(reply_markup=_period_kb(chat.id))
        return


    # -------- время --------
    if data.startswith("time:set:"):
        hhmm = data[len("time:set:"):]   # <- будет "09:00"
        try:
            set_digest_time(chat.id, hhmm)
        except Exception:
            await q.answer("Не удалось сохранить время", show_alert=True)
            return
        await q.answer(f"Время: {hhmm}")
        await q.edit_message_reply_markup(reply_markup=_time_kb(chat.id))
        return


    if data == "time:manual":
        context.user_data["awaiting_digest_time"] = True
        await q.answer()
        await context.bot.send_message(chat.id, "✍️ Введи время рассылки в формате <b>HH:MM</b> (например 09:30).", parse_mode=ParseMode.HTML)
        return

    


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_users_table()
    if update.effective_chat:
        upsert_user_from_update(update)

    if BOT_PASSWORD and not _allowed(update):
        await update.message.reply_text(
            "🔐 Этот бот работает по паролю.\n"
            "Отправьте команду:\n"
            "<code>/login ваш_пароль</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    await update.message.reply_text(WELCOME_TEXT, reply_markup=_main_menu_kb(update.effective_chat.id))


async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        await update.message.reply_text("Доступ запрещён.")
        return
    ensure_users_table()
    ch = update.effective_chat
    if ch:
        set_subscribed(ch.id, False)
    await update.message.reply_text("🔕 Уведомления отключены. Вернуться — /start")


async def login_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_users_table()
    if not BOT_PASSWORD:
        await update.message.reply_text("Пароль для бота не настроен. Обратитесь к администратору.")
        return
    payload = _extract_payload(update, ["login"]).strip()
    if not payload:
        await update.message.reply_text("Использование: /login <пароль>")
        return
    if payload == BOT_PASSWORD:
        if update.effective_chat:
            upsert_user_from_update(update, subscribed=True)
            set_authorized(update.effective_chat.id, True)
        await update.message.reply_text("✅ Пароль принят. Доступ к боту открыт.\n\n" + WELCOME_TEXT, reply_markup=_main_menu_kb(update.effective_chat.id))
    else:
        await update.message.reply_text("❌ Неверный пароль.")


async def greet_all_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        await update.message.reply_text("Доступ запрещён.")
        return
    ensure_users_table()
    payload = _extract_payload(update, ["greet_all"])
    greeting = payload.strip() if payload.strip() else WELCOME_TEXT
    ids = get_all_subscribed_chat_ids()
    if not ids:
        await update.message.reply_text("Нет подписчиков для рассылки.")
        return
    await update.message.reply_text(f"Начинаю рассылку по {len(ids)} чатам…")
    sent, failed = await _broadcast_to_all(context, greeting)
    await update.message.reply_text(f"Готово. Отправлено: {sent}. Ошибок: {failed}.")


async def _broadcast_to_all(context: ContextTypes.DEFAULT_TYPE, text_msg: str) -> Tuple[int, int]:
    ids = get_all_subscribed_chat_ids()
    sent = 0
    failed = 0
    for cid in ids:
        try:
            await context.bot.send_chat_action(chat_id=cid, action=ChatAction.TYPING)
            await context.bot.send_message(chat_id=cid, text=text_msg, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            sent += 1
        except Exception as e:
            failed += 1
            logging.warning("Send failed to %s: %s", cid, e)
        await asyncio.sleep(SLEEP_BETWEEN_SENDS)
    return sent, failed


async def restored_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        await update.message.reply_text("Доступ запрещён.")
        return
    ensure_users_table()
    payload = _extract_payload(update, ["restored"])
    message = payload.strip() if payload.strip() else RESTORED_DEFAULT
    ids = get_all_subscribed_chat_ids()
    if not ids:
        await update.message.reply_text("Нет подписчиков для рассылки.")
        return
    await update.message.reply_text(f"Рассылаю «восстановлено» по {len(ids)} чатам…")
    sent, failed = await _broadcast_to_all(context, message)
    await update.message.reply_text(f"Готово. Отправлено: {sent}. Ошибок: {failed}.")


async def schedule_greeting_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update):
        await update.message.reply_text("Доступ запрещён.")
        return
    if not context.args:
        await update.message.reply_text("Использование: /schedule_greeting HH:MM [Текст]")
        return
    try:
        hh, mm = context.args[0].split(":")
        hour, minute = int(hh), int(mm)
        payload = _extract_payload(update, ["schedule_greeting"])
        parts = payload.split(None, 1)
        msg_text = parts[1].strip() if len(parts) == 2 else WELCOME_TEXT
    except Exception:
        await update.message.reply_text("Неверное время. Пример: /schedule_greeting 09:00 Текст")
        return

    async def _job_send(_context: ContextTypes.DEFAULT_TYPE):
        ids = get_all_subscribed_chat_ids()
        sent = 0
        for cid in ids:
            try:
                await _context.bot.send_message(cid, msg_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                sent += 1
            except Exception as e:
                logging.warning("Scheduled send failed to %s: %s", cid, e)
            await asyncio.sleep(SLEEP_BETWEEN_SENDS)
        logging.info("Scheduled greeting sent to %d chats", sent)

    context.application.job_queue.run_daily(_job_send, time=dt.time(hour=hour, minute=minute))
    await update.message.reply_text(f"OK. Ежедневная рассылка в {hour:02d}:{minute:02d} запланирована.")


# ===================== MAIN =====================
def main():
    if not BOT_TOKEN or len(BOT_TOKEN) < 30:
        raise RuntimeError("BOT_TOKEN не задан или выглядит неверно")

    ensure_users_table()

    try:
        apg._setup_logger("tg_analytics", "tg_analytics.log")
        apg._setup_logger("main", "analysis.log")
    except Exception:
        pass

    TG_PROXY = (os.getenv("TG_PROXY") or "").strip() or None

    request = HTTPXRequest(
        proxy=TG_PROXY,
        connection_pool_size=32,
        pool_timeout=60,
        connect_timeout=30,
        read_timeout=60,
        write_timeout=60,
    )

    get_updates_request = HTTPXRequest(
        proxy=TG_PROXY,
        connection_pool_size=16,
        pool_timeout=60,
        connect_timeout=30,
        read_timeout=60,
        write_timeout=60,
    )

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(request)
        .get_updates_request(get_updates_request)
        .build()
    )

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("login", login_cmd))
    app.add_handler(CommandHandler("greet_all", greet_all_cmd))
    app.add_handler(CommandHandler("restored", restored_cmd))
    app.add_handler(CommandHandler("schedule_greeting", schedule_greeting_cmd))

    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))

    app.job_queue.run_repeating(_digest_tick, interval=DIGEST_TICK_SEC, first=10)

    logging.info("🤖 Бот запущен. Ожидаем сообщения...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
