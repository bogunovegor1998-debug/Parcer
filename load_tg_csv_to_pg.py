# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import csv
import json
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH, override=False)
load_dotenv(override=False)

INCOMING_DIR = Path(os.getenv("CSV_IMPORT_DIR", str(BASE_DIR / "incoming_csv")))
LOG_PATH = BASE_DIR / "load_tg_csv_to_pg.log"


def logp(*args):
    s = " ".join(str(a) for a in args)
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {s}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _sqlalchemy_uri_from_env() -> str:
    pg_dsn = (os.getenv("PG_DSN") or "").strip()
    if pg_dsn:
        if pg_dsn.startswith("postgresql://"):
            return "postgresql+psycopg2://" + pg_dsn[len("postgresql://"):]
        return pg_dsn

    user = os.getenv("PG_USER", "postgres")
    pwd = os.getenv("PG_PASSWORD", "")
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DB", "postgres")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


def get_engine():
    return create_engine(_sqlalchemy_uri_from_env(), pool_pre_ping=True)


def _csv_value(v: str):
    if v is None:
        return None
    x = str(v).strip()
    if x == "":
        return None
    return x


def truncate_staging(engine):
    sqls = [
        "TRUNCATE TABLE public.stg_tg_posts;",
        "TRUNCATE TABLE public.stg_tg_comments;",
        "TRUNCATE TABLE public.stg_tg_chat_messages;",
    ]
    with engine.begin() as conn:
        for sql in sqls:
            conn.execute(text(sql))


def load_posts_csv(engine, csv_path: Path) -> int:
    if not csv_path.exists():
        logp(f"[SKIP] no file: {csv_path}")
        return 0

    sql = text("""
        INSERT INTO public.stg_tg_posts (
            source_id, chat_id, msg_id, msg_date, sender_id, text,
            views, forwards, replies_count, permalink, raw_json, doc_id
        )
        VALUES (
            :source_id, :chat_id, :msg_id, :msg_date, :sender_id, :text,
            :views, :forwards, :replies_count, :permalink, CAST(:raw_json AS jsonb), :doc_id
        )
    """)

    cnt = 0
    with engine.begin() as conn, open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            conn.execute(sql, {
                "source_id": _csv_value(row.get("source_id")),
                "chat_id": _csv_value(row.get("chat_id")),
                "msg_id": _csv_value(row.get("msg_id")),
                "msg_date": _csv_value(row.get("msg_date")),
                "sender_id": _csv_value(row.get("sender_id")),
                "text": row.get("text"),
                "views": _csv_value(row.get("views")),
                "forwards": _csv_value(row.get("forwards")),
                "replies_count": _csv_value(row.get("replies_count")),
                "permalink": row.get("permalink"),
                "raw_json": row.get("raw_json"),
                "doc_id": row.get("doc_id"),
            })
            cnt += 1
    logp(f"[LOAD] posts csv rows={cnt}")
    return cnt


def load_comments_csv(engine, csv_path: Path) -> int:
    if not csv_path.exists():
        logp(f"[SKIP] no file: {csv_path}")
        return 0

    sql = text("""
        INSERT INTO public.stg_tg_comments (
            source_id, root_chat_id, root_msg_id, chat_id, msg_id, msg_date,
            sender_id, text, permalink, raw_json, reply_to_msg_id, doc_id
        )
        VALUES (
            :source_id, :root_chat_id, :root_msg_id, :chat_id, :msg_id, :msg_date,
            :sender_id, :text, :permalink, CAST(:raw_json AS jsonb), :reply_to_msg_id, :doc_id
        )
    """)

    cnt = 0
    with engine.begin() as conn, open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            conn.execute(sql, {
                "source_id": _csv_value(row.get("source_id")),
                "root_chat_id": _csv_value(row.get("root_chat_id")),
                "root_msg_id": _csv_value(row.get("root_msg_id")),
                "chat_id": _csv_value(row.get("chat_id")),
                "msg_id": _csv_value(row.get("msg_id")),
                "msg_date": _csv_value(row.get("msg_date")),
                "sender_id": _csv_value(row.get("sender_id")),
                "text": row.get("text"),
                "permalink": row.get("permalink"),
                "raw_json": row.get("raw_json"),
                "reply_to_msg_id": _csv_value(row.get("reply_to_msg_id")),
                "doc_id": row.get("doc_id"),
            })
            cnt += 1
    logp(f"[LOAD] comments csv rows={cnt}")
    return cnt


def load_chat_csv(engine, csv_path: Path) -> int:
    if not csv_path.exists():
        logp(f"[SKIP] no file: {csv_path}")
        return 0

    sql = text("""
        INSERT INTO public.stg_tg_chat_messages (
            source_id, chat_id, msg_id, msg_date, sender_id,
            text, permalink, raw_json, parent_msg_id
        )
        VALUES (
            :source_id, :chat_id, :msg_id, :msg_date, :sender_id,
            :text, :permalink, CAST(:raw_json AS jsonb), :parent_msg_id
        )
    """)

    cnt = 0
    with engine.begin() as conn, open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            conn.execute(sql, {
                "source_id": _csv_value(row.get("source_id")),
                "chat_id": _csv_value(row.get("chat_id")),
                "msg_id": _csv_value(row.get("msg_id")),
                "msg_date": _csv_value(row.get("msg_date")),
                "sender_id": _csv_value(row.get("sender_id")),
                "text": row.get("text"),
                "permalink": row.get("permalink"),
                "raw_json": row.get("raw_json"),
                "parent_msg_id": _csv_value(row.get("parent_msg_id")),
            })
            cnt += 1
    logp(f"[LOAD] chat csv rows={cnt}")
    return cnt


def merge_staging(engine):
    sql_posts = """
    INSERT INTO public.tg_posts (
        source_id, chat_id, msg_id, msg_date, sender_id, text,
        views, forwards, replies_count, permalink, raw_json, doc_id
    )
    SELECT
        s.source_id, s.chat_id, s.msg_id, s.msg_date, s.sender_id, s.text,
        s.views, s.forwards, s.replies_count, s.permalink, s.raw_json, s.doc_id
    FROM public.stg_tg_posts s
    ON CONFLICT (chat_id, msg_id) DO UPDATE
    SET
        source_id = EXCLUDED.source_id,
        msg_date = EXCLUDED.msg_date,
        sender_id = EXCLUDED.sender_id,
        text = EXCLUDED.text,
        views = EXCLUDED.views,
        forwards = EXCLUDED.forwards,
        replies_count = EXCLUDED.replies_count,
        permalink = EXCLUDED.permalink,
        raw_json = EXCLUDED.raw_json,
        doc_id = EXCLUDED.doc_id;
    """

    sql_comments = """
    INSERT INTO public.tg_comments (
        source_id, root_chat_id, root_msg_id, chat_id, msg_id, msg_date,
        sender_id, text, permalink, raw_json, reply_to_msg_id, doc_id
    )
    SELECT
        s.source_id, s.root_chat_id, s.root_msg_id, s.chat_id, s.msg_id, s.msg_date,
        s.sender_id, s.text, s.permalink, s.raw_json, s.reply_to_msg_id, s.doc_id
    FROM public.stg_tg_comments s
    ON CONFLICT (chat_id, msg_id) DO UPDATE
    SET
        source_id = EXCLUDED.source_id,
        root_chat_id = EXCLUDED.root_chat_id,
        root_msg_id = EXCLUDED.root_msg_id,
        msg_date = EXCLUDED.msg_date,
        sender_id = EXCLUDED.sender_id,
        text = EXCLUDED.text,
        permalink = EXCLUDED.permalink,
        raw_json = EXCLUDED.raw_json,
        reply_to_msg_id = EXCLUDED.reply_to_msg_id,
        doc_id = EXCLUDED.doc_id;
    """

    sql_chat = """
    INSERT INTO public.tg_chat_messages (
        source_id, chat_id, msg_id, msg_date, sender_id,
        text, permalink, raw_json, parent_msg_id
    )
    SELECT
        s.source_id, s.chat_id, s.msg_id, s.msg_date, s.sender_id,
        s.text, s.permalink, s.raw_json, s.parent_msg_id
    FROM public.stg_tg_chat_messages s
    ON CONFLICT (chat_id, msg_id) DO UPDATE
    SET
        source_id = EXCLUDED.source_id,
        msg_date = EXCLUDED.msg_date,
        sender_id = EXCLUDED.sender_id,
        text = EXCLUDED.text,
        permalink = EXCLUDED.permalink,
        raw_json = EXCLUDED.raw_json,
        parent_msg_id = EXCLUDED.parent_msg_id;
    """

    with engine.begin() as conn:
        conn.execute(text(sql_posts))
        conn.execute(text(sql_comments))
        conn.execute(text(sql_chat))

    logp("[MERGE] completed")


def main():
    engine = get_engine()

    posts_csv = INCOMING_DIR / "tg_posts.csv"
    comments_csv = INCOMING_DIR / "tg_comments.csv"
    chat_csv = INCOMING_DIR / "tg_chat_messages.csv"

    logp(f"[START] incoming_dir={INCOMING_DIR}")

    truncate_staging(engine)
    load_posts_csv(engine, posts_csv)
    load_comments_csv(engine, comments_csv)
    load_chat_csv(engine, chat_csv)
    merge_staging(engine)

    logp("[DONE] csv import completed")


if __name__ == "__main__":
    main()