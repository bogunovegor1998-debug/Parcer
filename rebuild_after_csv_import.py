# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import subprocess
import datetime as dt
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=False)
load_dotenv(override=False)

LOG_PATH = BASE_DIR / "rebuild_after_csv_import.log"


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


def refresh_matviews(engine):
    with engine.begin() as conn:
        logp("[MV] refreshing public.thread_docs_channel ...")
        conn.execute(text("REFRESH MATERIALIZED VIEW public.thread_docs_channel;"))
        logp("[MV] OK public.thread_docs_channel")

        logp("[MV] refreshing public.thread_docs_chat_reply ...")
        conn.execute(text("REFRESH MATERIALIZED VIEW public.thread_docs_chat_reply;"))
        logp("[MV] OK public.thread_docs_chat_reply")


def run_cmd(cmd):
    logp("[CMD]", " ".join(cmd))
    p = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip("\n")
        if line:
            logp("  " + line)

    rc = p.wait()
    logp("[RC]", rc)
    if rc != 0:
        raise RuntimeError(f"Command failed rc={rc}: {' '.join(cmd)}")


def main():
    engine = get_engine()

    refresh_matviews(engine)

    run_cmd([sys.executable, "build_faiss_threads.py"])
    run_cmd([sys.executable, "build_faiss_micro.py"])

    logp("[DONE] rebuild completed")


if __name__ == "__main__":
    main()