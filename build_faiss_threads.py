# -*- coding: utf-8 -*-
"""
build_faiss_threads.py — сборка FAISS индекса по "thread docs":
A) thread_docs_channel (пост + комменты)
B) thread_docs_chat_reply (корень + ответы)

Выход (FAISS_THREADS_DIR):
  - index.faiss
  - meta.json
  - mapping.csv
  - doc_id_map.npy

ENV:
  PG_DSN=postgresql://...
  FAISS_THREADS_DIR=./faiss_threads_store
  EMBEDDER=intfloat/multilingual-e5-base
  EMBED_DEVICE=cuda|cpu|mps
  EMB_BATCH=256
  EMB_TRUNC=6000
"""

from __future__ import annotations

import os
import json
import time
import logging

import numpy as np
import pandas as pd
import torch
import faiss

from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text


from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).with_name(".env"), override=False)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - faiss-threads - %(levelname)s - %(message)s",
)
log = logging.getLogger("faiss-threads")


FAISS_DIR = os.getenv("FAISS_THREADS_DIR", os.path.join(os.path.dirname(__file__), "faiss_threads_store"))
EMBED_MODEL = os.getenv("EMBEDDER", "intfloat/multilingual-e5-base")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")
EMB_BATCH = int(os.getenv("EMB_BATCH", "256"))
EMB_TRUNC = int(os.getenv("EMB_TRUNC", "6000"))

# маленькая защита от гигантских тредов (для эмбеддера)
MAX_BODY_CHARS = int(os.getenv("THREAD_BODY_MAX_CHARS", "30000"))


import os

def _sqlalchemy_uri_from_env() -> str:
    pg_dsn = (os.getenv("PG_DSN") or "").strip()
    if pg_dsn:
        # psycopg_pool обычно юзает postgresql://...
        if pg_dsn.startswith("postgresql://"):
            return "postgresql+psycopg2://" + pg_dsn[len("postgresql://"):]
        return pg_dsn

    # fallback если PG_DSN не задан
    user = os.getenv("PG_USER", "postgres")
    pwd  = os.getenv("PG_PASSWORD", "")
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db   = os.getenv("PG_DB", "postgres")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

from sqlalchemy import create_engine

engine = create_engine(_sqlalchemy_uri_from_env(), pool_pre_ping=True)


def _device_pick() -> str:
    try:
        if EMBED_DEVICE == "cuda" and torch.cuda.is_available():
            return "cuda"
        if EMBED_DEVICE == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def _norm_ws(s: str) -> str:
    return " ".join((s or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _clean_text(s: str) -> str:
    s = _norm_ws(str(s or ""))
    if not s:
        return ""
    if EMB_TRUNC and len(s) > EMB_TRUNC:
        s = s[:EMB_TRUNC]
    return s


def load_thread_docs(engine):
    sql = """
    SELECT
      c.thread_id,
      'channel'::text        AS doc_kind,
      c.source_username,
      c.source_title,
      c.root_chat_id         AS chat_id,
      c.root_msg_id,
      c.start_dt,
      c.last_dt,
      c.root_text,
      c.comments_text        AS body_text,
      c.n_comments           AS n_items,
      c.root_permalink
    FROM public.thread_docs_channel c

    UNION ALL

    SELECT
      r.thread_id,
      'chat_reply'::text     AS doc_kind,
      r.source_username,
      NULL::text             AS source_title,
      r.chat_id,
      r.root_msg_id,
      r.start_dt,
      r.last_dt,
      r.root_text,
      r.replies_text         AS body_text,
      r.n_messages           AS n_items,
      r.root_permalink
    FROM public.thread_docs_chat_reply r;
    """
    return pd.read_sql(text(sql), engine)


def build_doc_text(row: pd.Series) -> str:
    root = _clean_text(row.get("root_text", ""))
    body = row.get("body_text", "")
    body = str(body or "")
    if MAX_BODY_CHARS and len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS]
    body = _clean_text(body)

    kind = str(row.get("doc_kind") or "").strip()
    if kind == "chan":
        if body:
            return _clean_text(f"{root}\n\nКомментарии:\n{body}")
        return root
    else:
        if body:
            return _clean_text(f"{root}\n\nОтветы:\n{body}")
        return root


def main():
    os.makedirs(FAISS_DIR, exist_ok=True)

    device = _device_pick()
    log.info("device=%s torch=%s cuda=%s", device, torch.__version__, torch.version.cuda)

    engine = create_engine(_sqlalchemy_uri_from_env(), pool_pre_ping=True)
    with engine.connect() as c:
        print(c.execute(text("""
            SELECT current_database(), current_user,
                to_regclass('public.thread_docs_channel'),
                to_regclass('public.thread_docs_chat_reply')
        """)).fetchone())



    df = load_thread_docs(engine)
    with engine.connect() as c:
        print(c.execute(text("select current_database(), current_user, to_regclass('public.thread_docs_all')")).fetchone())

    if df is None or df.empty:
        raise SystemExit("thread_docs_all пустой. Сначала создай/обнови materialized views.")

    # соберём doc_text
    df["doc_text"] = df.apply(build_doc_text, axis=1)

    # уберём пустое
    df = df[df["doc_text"].astype(str).str.strip().ne("")].copy()
    df.reset_index(drop=True, inplace=True)

    log.info("rows_for_index=%d", len(df))
    if df.empty:
        raise SystemExit("После очистки doc_text не осталось документов для индекса.")

    # FAISS ids
    faiss_ids = np.arange(len(df), dtype=np.int64)
    doc_ids = df["thread_id"].astype(str).to_numpy()

    # embedder
    log.info("load embedder: %s (device=%s)", EMBED_MODEL, device)
    model = SentenceTransformer(EMBED_MODEL, device=device)

    # E5: documents = passage:
    inp = [f"passage: {t}" for t in df["doc_text"].tolist()]

    t0 = time.time()
    vecs = model.encode(
        inp,
        batch_size=EMB_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    vecs = np.ascontiguousarray(vecs)
    dim = int(vecs.shape[1])
    log.info("embeddings shape=%s dt=%.2fs", vecs.shape, time.time() - t0)

    # faiss (cosine через IP при normalize_embeddings=True)
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(vecs, faiss_ids)

    # save
    idx_path = os.path.join(FAISS_DIR, "index.faiss")
    meta_path = os.path.join(FAISS_DIR, "meta.json")
    map_path = os.path.join(FAISS_DIR, "mapping.csv")
    npy_path = os.path.join(FAISS_DIR, "doc_id_map.npy")

    faiss.write_index(index, idx_path)

    meta = {
        "model": EMBED_MODEL,
        "dim": dim,
        "rows": int(vecs.shape[0]),
        "built_on_unix": time.time(),
        "embed_device": device,
        "embed_batch": EMB_BATCH,
        "embed_trunc": EMB_TRUNC,
        "body_max_chars": MAX_BODY_CHARS,
        "source": "thread_docs_all (A=thread_docs_channel, B=thread_docs_chat_reply)",
        "id_scheme": "faiss_id:int64 -> doc_id_map.npy -> thread_id:text",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # mapping.csv (для дебага/ручной проверки)
    out = df.copy()
    out.insert(0, "faiss_id", faiss_ids)

    # даты — в строку
    for c in ("start_dt", "last_dt"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce", utc=True).astype(str)

    cols = [
        "faiss_id",
        "thread_id",
        "doc_kind",
        "source_username",
        "source_title",
        "chat_id",
        "root_msg_id",
        "start_dt",
        "last_dt",
        "n_items",
        "root_permalink",
    ]
    cols = [c for c in cols if c in out.columns]
    out[cols].to_csv(map_path, index=False, encoding="utf-8-sig", sep=";")

    np.save(npy_path, doc_ids)

    log.info("saved: %s | %s | %s | %s", idx_path, meta_path, map_path, npy_path)
    log.info("DONE")


if __name__ == "__main__":
    main()
