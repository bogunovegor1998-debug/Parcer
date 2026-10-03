# -*- coding: utf-8 -*-
"""
analysis_tg_pg.py — аналитика для python_telegram_bot (Postgres VIEW/MATVIEW)

Функции:
- greeting
- count
- examples
- summary (опционально по тредам: thread_docs_all)
Опционально:
- summary через FAISS (threads store) + LLM теги (summary_llm.py)
- + microthreads FAISS (микротреды из "каши" чатов) — участвуют в summary pool

ENV:
  PG_DSN (или PG_USER/PG_PASSWORD/PG_HOST/PG_PORT/PG_DB)
  TG_VIEW_RAW=feedback_raw_v3
  TG_VIEW_THREADS=thread_docs_all
  SUMMARY_USE_THREADS=1

  SEM_USE_FAISS=1
  FAISS_THREADS_DIR=.../faiss_threads_store
  FAISS_MODEL=intfloat/multilingual-e5-base
  FAISS_DEVICE=cuda|cpu|mps
  FAISS_TOP_K=120

  SEM_USE_MICRO=1
  FAISS_MICRO_DIR=.../faiss_micro
  FAISS_MICRO_TOP_K=120

  FAISS_RELOAD_ON_CHANGE=1

  DATA_TTL_SEC=600
  CACHE_BUST_FILE=.../.data_updated   (touch после ingest/refresh/FAISS swap)
"""

from __future__ import annotations

import os
import re
import time
import csv
import warnings
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Optional, Tuple, Set, Dict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


# ---- optional LLM summary ----
import traceback
_tmp_log = logging.getLogger("tg_analytics")

try:
    from summary_llm import summarize_topn, answer_with_context
except Exception as e:
    summarize_topn = None
    answer_with_context = None
    _tmp_log.error("summary_llm import FAILED: %s", e)
    _tmp_log.error(traceback.format_exc())


# ===================== ЛОГИ =====================
def _setup_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
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
        os.path.join(log_dir, filename),
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False

    for noisy in (
        "httpx", "requests", "urllib3",
        "huggingface_hub", "sentence_transformers", "transformers", "faiss",
        "telegram", "telegram.ext", "telethon",
        "sqlalchemy.engine",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logger._inited = True
    return logger


log = _setup_logger("tg_analytics", "tg_analytics.log")
main_log = _setup_logger("main", "analysis.log")


# ===================== КОНФИГ =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

DATA_TTL_SEC = int(os.getenv("DATA_TTL_SEC", "600"))

RAW_VIEW_NAME = os.getenv("TG_VIEW_RAW", os.getenv("TG_VIEW", "feedback_raw_v3"))
THREADS_VIEW_NAME = os.getenv("TG_VIEW_THREADS", "thread_docs_all")

SUMMARY_USE_THREADS = os.getenv("SUMMARY_USE_THREADS", "1").strip().lower() not in ("0", "false", "no")

# FAISS (threads store)
SEM_USE_FAISS = os.getenv("SEM_USE_FAISS", "1").strip().lower() not in ("0", "false", "no")
FAISS_THREADS_DIR = os.getenv("FAISS_THREADS_DIR", os.path.join(BASE_DIR, "faiss_threads_store"))
FAISS_MODEL = os.getenv("FAISS_MODEL", "intfloat/multilingual-e5-base")
FAISS_DEVICE = os.getenv("FAISS_DEVICE", os.getenv("EMBED_DEVICE", "cpu"))
FAISS_TOP_K = int(os.getenv("FAISS_TOP_K", "120"))

COUNT_FAISS_TOP_K = int(os.getenv("COUNT_FAISS_TOP_K", "3000"))
COUNT_MICRO_TOP_K = int(os.getenv("COUNT_MICRO_TOP_K", "3000"))
COUNT_MIN_SCORE_THREADS = float(os.getenv("COUNT_MIN_SCORE_THREADS", "0.78"))
COUNT_MIN_SCORE_MICRO = float(os.getenv("COUNT_MIN_SCORE_MICRO", "0.78"))

# FAISS (micro)
SEM_USE_MICRO = os.getenv("SEM_USE_MICRO", "1").strip().lower() not in ("0", "false", "no")
FAISS_MICRO_DIR = os.getenv("FAISS_MICRO_DIR", os.path.join(BASE_DIR, "faiss_micro"))
FAISS_MICRO_TOP_K = int(os.getenv("FAISS_MICRO_TOP_K", os.getenv("FAISS_TOP_K", "120")))

FAISS_RELOAD_ON_CHANGE = os.getenv("FAISS_RELOAD_ON_CHANGE", "1").strip().lower() not in ("0", "false", "no")

FAISS_MIN_SCORE_THREADS = float(os.getenv("FAISS_MIN_SCORE_THREADS", "0.42"))
FAISS_MIN_SCORE_MICRO = float(os.getenv("FAISS_MIN_SCORE_MICRO", "0.45"))
FAISS_STRICT_TOKENS_MAX = int(os.getenv("FAISS_STRICT_TOKENS_MAX", "2"))

LEX_AND_RATIO = float(os.getenv("LEX_AND_RATIO", "8.0"))
LEX_AND_MIN = int(os.getenv("LEX_AND_MIN", "1"))

SUMMARY_MIN_DOCS_FOR_LLM = int(os.getenv("SUMMARY_MIN_DOCS_FOR_LLM", "5"))

QA_DF_CAP = int(os.getenv("QA_DF_CAP", "40"))
QA_CONTEXT_TOP_K = int(os.getenv("QA_CONTEXT_TOP_K", "12"))
QA_MIN_AND_HITS = int(os.getenv("QA_MIN_AND_HITS", "2"))
QA_AND_RATIO = float(os.getenv("QA_AND_RATIO", "6.0"))
QA_PREVIEW_EXAMPLES = int(os.getenv("QA_PREVIEW_EXAMPLES", "3"))

# DB
PG_DSN = (os.getenv("PG_DSN") or "").strip()
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "postgres")

XLSX_ROWS_MAX = int(os.getenv("XLSX_ROWS_MAX", "30000"))
CSV_ROWS_MAX = int(os.getenv("CSV_ROWS_MAX", "100000"))
COUNT_EXPORT_MAX = int(os.getenv("COUNT_EXPORT_MAX", "100000"))
EXAMPLES_EXPORT_MAX = int(os.getenv("EXAMPLES_EXPORT_MAX", "2000"))
SUMMARY_DF_CAP = int(os.getenv("SUMMARY_DF_CAP", "12000"))  
SUMMARY_TOPN = int(os.getenv("SUMMARY_TOPN", "120"))         
SUMMARY_GEO_MAX_LINES = int(os.getenv("SUMMARY_GEO_MAX_LINES", "2"))
SUMMARY_GEO_MAX_STATIONS = int(os.getenv("SUMMARY_GEO_MAX_STATIONS", "3"))
SUMMARY_GEO_MIN_MENTION_COUNT = int(os.getenv("SUMMARY_GEO_MIN_MENTION_COUNT", "2"))

EXPORT_THREAD_BODY_MAX_CHARS = int(os.getenv("EXPORT_THREAD_BODY_MAX_CHARS", "3500"))
EXPORT_THREAD_HEAD_LINES = int(os.getenv("EXPORT_THREAD_HEAD_LINES", "4"))
EXPORT_THREAD_TAIL_LINES = int(os.getenv("EXPORT_THREAD_TAIL_LINES", "2"))
EXPORT_THREAD_KEY_LINES = int(os.getenv("EXPORT_THREAD_KEY_LINES", "8"))

MAX_EXAMPLE_CHARS = int(os.getenv("MAX_EXAMPLE_CHARS", "1100"))
BULLET = "🔹"
XLSX_WIDE_COL = int(os.getenv("XLSX_WIDE_COL", "110"))

STATIONS_CSV = os.getenv("STATIONS_CSV", os.path.join(BASE_DIR, "stations.csv"))
_STATIONS_DF: Optional[pd.DataFrame] = None

_engine: Optional[Engine] = None
_DATA_CACHE_BY_VIEW: Dict[str, Dict] = {}  # view_name -> {"ts":..., "df":...}

FILTER_NOISE = os.getenv("FILTER_NOISE", "1").strip().lower() not in ("0", "false", "no")
NOISE_MAX_LEN = int(os.getenv("NOISE_MAX_LEN", "60"))  # короткие “привет” и т.п.

# ===================== HOT-RELOAD (CACHE BUST) =====================

CACHE_BUST_FILE = os.getenv("CACHE_BUST_FILE", os.path.join(BASE_DIR, ".data_updated")).strip()
CACHE_BUST_CHECK_EVERY_SEC = int(os.getenv("CACHE_BUST_CHECK_EVERY_SEC", "3"))

_last_bust_check_ts = 0.0
_last_bust_mtime = 0.0

def _bust_file_mtime() -> float:
    """mtime файла-флажка. 0 если нет/ошибка."""
    if not CACHE_BUST_FILE:
        return 0.0
    try:
        return Path(CACHE_BUST_FILE).stat().st_mtime
    except Exception:
        return 0.0

def invalidate_caches(reason: str = ""):
    global _DATA_CACHE_BY_VIEW
    global _FAISS_THREADS_STORE, _FAISS_THREADS_FP, _FAISS_THREADS_DIR_USED
    global _FAISS_MICRO_STORE, _FAISS_MICRO_FP, _FAISS_MICRO_DIR_USED

    _DATA_CACHE_BY_VIEW = {}  # <-- важно: сбрасываем все витрины

    _FAISS_THREADS_STORE = None
    _FAISS_THREADS_FP = 0.0
    _FAISS_THREADS_DIR_USED = None

    _FAISS_MICRO_STORE = None
    _FAISS_MICRO_FP = 0.0
    _FAISS_MICRO_DIR_USED = None

    log.info("[CACHE] invalidated (%s)", reason or "no-reason")

def maybe_reload_on_bust_file():
    """
    Дёргаем часто, но реальную stat делаем раз в N секунд.
    При изменении bust-файла — сбрасываем кэши.
    """
    global _last_bust_check_ts, _last_bust_mtime
    now = time.time()
    if now - _last_bust_check_ts < CACHE_BUST_CHECK_EVERY_SEC:
        return
    _last_bust_check_ts = now

    mt = _bust_file_mtime()
    if mt and mt != _last_bust_mtime:
        _last_bust_mtime = mt
        invalidate_caches(reason=f"bust_file_mtime={mt}")

# ===================== ТЕКСТ =====================
TOKEN_RE = re.compile(r"[a-zа-я0-9\-]+", re.I)

def _lower(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("ё", "е").lower()).strip()

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-\.\sа-яА-ЯёЁ]+", "_", s)
    return re.sub(r"\s+", " ", s).strip()

_NOISE_PAT = re.compile(
    r"^\s*(?:"
    r"(?:привет(?:ствую)?|здравствуй(?:те)?|доброе\s+утро|добрый\s+день|добрый\s+вечер|доброй\s+ночи|welcome)"
    r"(?:[\s!,.:\-\)\(]*"
    r"(?:тебя|вас|в\s+чате|в\s+канале|в\s+группе|в\s+сообществе).*)?"
    r"|спасибо"
    r")\s*$",
    re.I,
)

def _drop_noise_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Убираем короткие “привет/здравствуйте/…” из каналов/чат-ботов и т.п.
    Работает по тексту (без завязки на sender_is_bot, которого может не быть).
    """
    if df is None or df.empty or not FILTER_NOISE:
        return df
    if "full_text" not in df.columns:
        return df

    ft = df["full_text"].fillna("").astype(str).map(_normalize_ws)
    short = ft.str.len().fillna(0) <= NOISE_MAX_LEN
    only_greet = ft.map(lambda s: bool(_NOISE_PAT.match(_lower(s))))
    mask = short & only_greet

    dropped = int(mask.sum())
    if dropped > 0:
        main_log.info("[NOISE] drop=%d", dropped)
        return df.loc[~mask].copy()
    return df


# ===================== STATIONS =====================
def load_stations() -> pd.DataFrame:
    global _STATIONS_DF
    if _STATIONS_DF is not None:
        return _STATIONS_DF

    if not os.path.isfile(STATIONS_CSV):
        log.warning("[STATIONS] stations.csv not found: %s (station scope disabled)", STATIONS_CSV)
        _STATIONS_DF = pd.DataFrame()
        return _STATIONS_DF

    try:
        df = pd.read_csv(STATIONS_CSV, sep=";", encoding="cp1251")
    except Exception:
        df = pd.read_csv(STATIONS_CSV, sep=";", encoding="utf-8")

    if "name_norm" not in df.columns:
        if "name_ru" in df.columns:
            df["name_norm"] = df["name_ru"]
        else:
            _STATIONS_DF = pd.DataFrame()
            return _STATIONS_DF

    df["name_norm"] = df["name_norm"].astype(str).map(_lower)
    _STATIONS_DF = df
    log.info("[STATIONS] loaded=%d", len(df))
    return df

def _stations_from_query(q: str) -> List[str]:
    st = load_stations()
    if st.empty:
        return []
    ql = _lower(q)
    hits: Set[str] = set()
    for name in st["name_norm"].dropna().unique():
        name = str(name).strip()
        if name and name in ql:
            hits.add(name)
    return sorted(hits)

def _station_pat(st_name: str) -> Optional[re.Pattern]:
    s = _lower(st_name).replace("-", " ")
    toks = re.findall(r"[0-9a-zа-я]+", s, flags=re.I)
    if not toks:
        return None
    body = r"[\s\-]+".join(map(re.escape, toks))
    return re.compile(r"\b" + body + r"\b", re.I)


# ===================== DB / LOAD =====================
def _sqlalchemy_uri() -> str:
    if PG_DSN:
        if PG_DSN.startswith("postgresql://"):
            return "postgresql+psycopg://" + PG_DSN[len("postgresql://"):]
        return PG_DSN
    return f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(_sqlalchemy_uri(), pool_pre_ping=True)
    return _engine

def _ensure_doc_id(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "doc_id" in df.columns:
        return df
    if "chat_id" not in df.columns or "msg_id" not in df.columns:
        if "row_id" in df.columns:
            df["doc_id"] = df["row_id"].astype(str)
            return df
        raise RuntimeError("Нельзя собрать doc_id: нет chat_id/msg_id (и нет row_id).")

    kind = df["kind"].astype(str).str.lower() if "kind" in df.columns else pd.Series(["msg"] * len(df), index=df.index)
    prefix = np.where(
        kind.str.contains("post", na=False),
        "post:",
        np.where(kind.str.contains("comment", na=False), "comment:", kind + ":")
    )
    df["doc_id"] = prefix + df["chat_id"].astype(str) + ":" + df["msg_id"].astype(str)
    return df

def _ensure_parent_text(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "parent_text" in df.columns:
        return df
    if "parent_msg_id" not in df.columns:
        df["parent_msg_id"] = pd.NA
        df["parent_text"] = ""
        return df

    parent = df[["chat_id", "msg_id", "full_text"]].copy()
    parent = parent.rename(columns={"msg_id": "parent_msg_id", "full_text": "parent_text"})
    out = df.merge(parent, on=["chat_id", "parent_msg_id"], how="left")
    out["parent_text"] = out["parent_text"].fillna("").astype(str)
    return out

def _normalize_df_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализуем разные витрины к общим колонкам."""
    if df is None or df.empty:
        return df

    cols = set(df.columns)

    # THREADS VIEW: thread_docs_all
    if "thread_id" in cols and ("body_text" in cols or "comments_text" in cols or "replies_text" in cols):
        out = df.copy()

        out["kind"] = out.get("doc_kind", "thread").astype(str)
        out["msg_date"] = pd.to_datetime(out.get("last_dt"), errors="coerce", utc=True)

        # thread_docs_all обычно уже имеет root_text + body_text
        out["root_text"] = out.get("root_text", "").astype(str).map(_normalize_ws)
        body = out.get("body_text", None)
        if body is None and "comments_text" in out.columns:
            body = out["comments_text"]
        if body is None and "replies_text" in out.columns:
            body = out["replies_text"]
        if body is None:
            body = ""

        out["full_text"] = (out["root_text"].astype(str) + "\n" + pd.Series(body, index=out.index).astype(str)).map(_normalize_ws)

        out["parent_msg_id"] = pd.NA
        out["parent_text"] = ""

        out["doc_id"] = out["thread_id"].astype(str)

        out["permalink"] = out.get("root_permalink", "").astype(str)
        out["root_permalink"] = out.get("root_permalink", "").astype(str)

        out["root_chat_id"] = out.get("chat_id")
        out["root_msg_id"] = out.get("root_msg_id")
        out["chat_id"] = out.get("chat_id")
        out["msg_id"] = out.get("root_msg_id")

        return out

    # RAW VIEW: feedback_raw_v3/v2/raw
    out = df.copy()
    out = _ensure_doc_id(out)

    if "msg_date" in out.columns:
        out["msg_date"] = pd.to_datetime(out["msg_date"], errors="coerce", utc=True)
    else:
        out["msg_date"] = pd.NaT

    if "full_text" not in out.columns:
        out["full_text"] = out.get("text", "")

    out["full_text"] = out["full_text"].astype(str).map(_normalize_ws)

    out = _ensure_parent_text(out)

    if "root_text" in out.columns:
        out["root_text"] = out["root_text"].astype(str).map(_normalize_ws)
    else:
        out["root_text"] = ""

    if "root_chat_id" not in out.columns:
        out["root_chat_id"] = pd.NA
    if "root_msg_id" not in out.columns:
        out["root_msg_id"] = pd.NA
    if "root_permalink" not in out.columns:
        out["root_permalink"] = ""

    return out

def _cache_bust_mtime() -> float:
    try:
        if os.path.exists(CACHE_BUST_FILE):
            return float(os.path.getmtime(CACHE_BUST_FILE))
    except Exception:
        return 0.0
    return 0.0

def load_data(view_name: Optional[str] = None) -> pd.DataFrame:
    """
    Совместимость с ботом:
    - load_data()              -> грузим RAW_VIEW_NAME
    - load_data("some_view")   -> грузим указанную витрину
    """
    vn = (view_name or RAW_VIEW_NAME or "").strip()
    if not vn:
        raise ValueError("view_name is empty")

    bust_mtime = _cache_bust_mtime()
    cache = _DATA_CACHE_BY_VIEW.get(vn)
    if cache and cache.get("df") is not None:
        ts = float(cache.get("ts", 0.0))
        if (time.time() - ts) < DATA_TTL_SEC and ts >= bust_mtime:
            return cache["df"]

    t0 = time.time()
    sql = f"SELECT * FROM {vn};"
    df = pd.read_sql(sql, _get_engine())
    df = _normalize_df_schema(df)

    if "root_text" not in df.columns:
        df["root_text"] = ""
    if "parent_text" not in df.columns:
        df["parent_text"] = ""

    df["_concat_txt"] = (
        (df["full_text"].astype(str) + " " + df["parent_text"].astype(str) + " " + df["root_text"].astype(str))
        .map(_lower)
        .str.strip()
    )

    df["_lex_txt_raw"] = (
        (df["full_text"].astype(str) + " " + df["parent_text"].astype(str))
        .map(_lower)
        .str.strip()
    )

    # фильтр мусора (короткие тех-сообщения)
    df = _drop_noise_rows(df)

    if "source_username" in df.columns:
        df["_concat_txt"] = (df["_concat_txt"] + " " + df["source_username"].astype(str).map(_lower)).str.strip()

    df = df.sort_values("msg_date", ascending=False, na_position="last").reset_index(drop=True)
    df["_row_idx"] = np.arange(len(df), dtype=np.int64)

    _DATA_CACHE_BY_VIEW[vn] = {"df": df, "ts": time.time()}
    log.info("[LOAD] rows=%d (view=%s) dt=%.2fs", len(df), vn, time.time() - t0)
    return df


# ===================== FAISS stores (threads + micro) =====================
_FAISS_THREADS_STORE = None
_FAISS_THREADS_FP = 0.0
_FAISS_THREADS_DIR_USED = None

_FAISS_MICRO_STORE = None
_FAISS_MICRO_FP = 0.0
_FAISS_MICRO_DIR_USED = None



def _faiss_dir_fingerprint(dir_path: str) -> float:
    """Отпечаток директории индекса: mtime ключевых файлов."""
    try:
        p = Path(dir_path)
        # выбираем то, что точно меняется при пересборке
        candidates = [p / "meta.json", p / "index.faiss", p / "mapping.csv", p / "doc_id_map.npy"]
        mt = 0.0
        for f in candidates:
            if f.exists():
                mt = max(mt, f.stat().st_mtime)
        return mt
    except Exception:
        return 0.0

_FAISS_THREADS_STORE = None
_FAISS_THREADS_FP = 0.0




_FAISS_MICRO_STORE = None
_FAISS_MICRO_FP = 0.0


def _load_faiss_store_cached(dir_path: str, name: str):
    """Единый кеш/перезагрузка для threads и micro."""
    global _FAISS_THREADS_STORE, _FAISS_THREADS_FP, _FAISS_THREADS_DIR_USED
    global _FAISS_MICRO_STORE, _FAISS_MICRO_FP, _FAISS_MICRO_DIR_USED

    if name == "threads":
        store_ref, fp_ref, dir_ref = "_FAISS_THREADS_STORE", "_FAISS_THREADS_FP", "_FAISS_THREADS_DIR_USED"
    else:
        store_ref, fp_ref, dir_ref = "_FAISS_MICRO_STORE", "_FAISS_MICRO_FP", "_FAISS_MICRO_DIR_USED"

    cur_fp = _faiss_dir_fingerprint(dir_path)
    store = globals()[store_ref]
    prev_fp = globals()[fp_ref]
    prev_dir = globals()[dir_ref]

    need_reload = (
        store is None
        or (FAISS_RELOAD_ON_CHANGE and cur_fp and (cur_fp != prev_fp))
        or (prev_dir is not None and str(prev_dir) != str(dir_path))
    )

    if not need_reload:
        return store

    try:
        from semantic_faiss import load_store
        new_store = load_store(dir_path, name)
        if new_store is None:
            main_log.warning("[FAISS] %s store not found at %s", name, dir_path)
            globals()[store_ref] = None
            return None

        globals()[store_ref] = new_store
        globals()[fp_ref] = cur_fp
        globals()[dir_ref] = dir_path

        main_log.info("[FAISS] %s store loaded dir=%s fp=%.3f", name, dir_path, cur_fp)
        return new_store
    except Exception as e:
        main_log.warning("[FAISS] %s load_store failed: %s", name, e)
        globals()[store_ref] = None
        return None

def _get_faiss_threads_store():
    return _load_faiss_store_cached(FAISS_THREADS_DIR, "threads")

def _get_faiss_micro_store():
    return _load_faiss_store_cached(FAISS_MICRO_DIR, "micro")


def _faiss_filter_threads(df: pd.DataFrame, q: str) -> pd.DataFrame:
    """
    Для threads: сузить/упорядочить DF по FAISS hits.
    Важно: df должен быть из thread_docs_all и иметь doc_id=thread_id.
    """
    if (not SEM_USE_FAISS) or df.empty or not (q or "").strip():
        return df
    store = _get_faiss_threads_store()
    if store is None:
        return df

    try:
        from semantic_faiss import search as faiss_search
        hits = faiss_search(
            store=store,
            query_text=q,
            model_name=FAISS_MODEL,
            device=FAISS_DEVICE,
            top_k=FAISS_TOP_K,
        )
        if hits is None or hits.empty or "doc_id" not in hits.columns:
            return df

            # --- score threshold (особенно полезно для узких запросов) ---
        try:
            tokens = _content_tokens(q)
            strict = (len(tokens) > 0 and len(tokens) <= FAISS_STRICT_TOKENS_MAX)
        except Exception:
            strict = False

        if "score" in hits.columns:
            thr = FAISS_MIN_SCORE_THREADS
            hits_thr = hits[hits["score"] >= thr].copy()

            # если запрос узкий — лучше меньше, но точнее
            if strict:
                if not hits_thr.empty:
                    hits = hits_thr
            else:
                # для широких запросов не режем слишком агрессивно
                if len(hits_thr) >= 15:
                    hits = hits_thr

        ids = hits["doc_id"].astype(str).dropna().unique()
        out = df[df["doc_id"].astype(str).isin(ids)].copy()
        if out.empty:
            return df

        if "score" in hits.columns:
            out = out.merge(hits[["doc_id", "score"]], on="doc_id", how="left")
            out = out.sort_values(["score", "msg_date"], ascending=[False, False], na_position="last")

        main_log.info("[FAISS:threads] filtered %d -> %d", len(df), len(out))
        return out
    except Exception as e:
        main_log.warning("[FAISS:threads] search failed: %s", e)
        return df
    
def _faiss_filter_threads_for_count(df: pd.DataFrame, q: str) -> pd.DataFrame:
    """
    Более широкий semantic retrieval для count:
    - берём большой top_k
    - потом режем по более строгому score threshold
    - не ограничиваемся искусственно 120 документами
    """
    if (not SEM_USE_FAISS) or df.empty or not (q or "").strip():
        return df

    store = _get_faiss_threads_store()
    if store is None:
        return df

    try:
        from semantic_faiss import search as faiss_search

        hits = faiss_search(
            store=store,
            query_text=q,
            model_name=FAISS_MODEL,
            device=FAISS_DEVICE,
            top_k=COUNT_FAISS_TOP_K,
        )
        if hits is None or hits.empty or "doc_id" not in hits.columns:
            return df.head(0).copy()

        if "score" in hits.columns:
            hits["score"] = pd.to_numeric(hits["score"], errors="coerce")
            hits = hits[hits["score"] >= COUNT_MIN_SCORE_THREADS].copy()

        if hits.empty:
            main_log.info("[FAISS:threads:count] no hits above threshold=%.3f", COUNT_MIN_SCORE_THREADS)
            return df.head(0).copy()

        ids = hits["doc_id"].astype(str).dropna().unique()
        out = df[df["doc_id"].astype(str).isin(ids)].copy()

        if out.empty:
            return out

        if "score" in hits.columns:
            out = out.merge(hits[["doc_id", "score"]], on="doc_id", how="left")
            out = out.sort_values(["score", "msg_date"], ascending=[False, False], na_position="last")

        main_log.info(
            "[FAISS:threads:count] filtered %d -> %d (top_k=%d thr=%.3f)",
            len(df), len(out), COUNT_FAISS_TOP_K, COUNT_MIN_SCORE_THREADS
        )
        return out

    except Exception as e:
        main_log.warning("[FAISS:threads:count] search failed: %s", e)
        return df.head(0).copy()


def _micro_docs_from_faiss(query: str) -> pd.DataFrame:
    """
    Возвращает DF микродоков (уже “как для summary”), полученных из FAISS micro.
    Важно: doc_id должен совпадать с thread_id из mapping (micro:...),
    поэтому НЕ добавляем префикс 'micro:' повторно.
    """
    if (not SEM_USE_MICRO) or not (query or "").strip():
        return pd.DataFrame()

    store = _get_faiss_micro_store()
    if store is None:
        return pd.DataFrame()

    try:
        from semantic_faiss import search as faiss_search

        hits = faiss_search(
            store=store,
            query_text=query,
            model_name=FAISS_MODEL,
            device=FAISS_DEVICE,
            top_k=FAISS_MICRO_TOP_K,
        )
        if hits is None or hits.empty:
            return pd.DataFrame()

                    # --- score threshold ---
        if "score" in hits.columns:
            hits_thr = hits[hits["score"] >= FAISS_MIN_SCORE_MICRO].copy()
           
            if not hits_thr.empty:
                hits = hits_thr

        if "doc_id" not in hits.columns:
            return pd.DataFrame()

        doc_id = hits["doc_id"].astype(str)

        # текст: prefer doc_text (semantic_faiss добавляет), иначе собираем из root/body
        if "doc_text" in hits.columns:
            full_text = hits["doc_text"].astype(str)
        else:
            rt = hits["root_text"].astype(str) if "root_text" in hits.columns else ""
            bt = hits["body_text"].astype(str) if "body_text" in hits.columns else ""
            full_text = (rt.astype(str) + "\n" + bt.astype(str)).astype(str)

        out = pd.DataFrame({
            "doc_id": doc_id,
            "kind": "micro",
            "source_username": hits["source_username"].astype(str) if "source_username" in hits.columns else "micro",
            "source_title": hits["source_title"].astype(str) if "source_title" in hits.columns else "micro",
            "full_text": full_text.astype(str),
            "parent_text": "",
            "root_text": hits["root_text"].astype(str) if "root_text" in hits.columns else "",
            "permalink": hits["root_permalink"].astype(str) if "root_permalink" in hits.columns else "",
        })

        dt_col = "last_dt" if "last_dt" in hits.columns else ("start_dt" if "start_dt" in hits.columns else None)
        out["msg_date"] = pd.to_datetime(hits[dt_col], errors="coerce", utc=True) if dt_col else pd.NaT

        if "score" in hits.columns:
            out["score"] = pd.to_numeric(hits["score"], errors="coerce")

        out["_concat_txt"] = out["full_text"].astype(str).map(_lower).str.strip()
        out["_row_idx"] = np.arange(len(out), dtype=np.int64)

        return out
    except Exception as e:
        main_log.warning("[FAISS:micro] search failed: %s", e)
        return pd.DataFrame()
    
def _micro_docs_from_faiss_for_count(query: str) -> pd.DataFrame:
    """
    Более широкий и более строгий semantic retrieval по micro для count.
    """
    if (not SEM_USE_MICRO) or not (query or "").strip():
        return pd.DataFrame()

    store = _get_faiss_micro_store()
    if store is None:
        return pd.DataFrame()

    try:
        from semantic_faiss import search as faiss_search

        hits = faiss_search(
            store=store,
            query_text=query,
            model_name=FAISS_MODEL,
            device=FAISS_DEVICE,
            top_k=COUNT_MICRO_TOP_K,
        )
        if hits is None or hits.empty or "doc_id" not in hits.columns:
            return pd.DataFrame()

        if "score" in hits.columns:
            hits["score"] = pd.to_numeric(hits["score"], errors="coerce")
            hits = hits[hits["score"] >= COUNT_MIN_SCORE_MICRO].copy()

        if hits.empty:
            main_log.info("[FAISS:micro:count] no hits above threshold=%.3f", COUNT_MIN_SCORE_MICRO)
            return pd.DataFrame()

        doc_id = hits["doc_id"].astype(str)

        if "doc_text" in hits.columns:
            full_text = hits["doc_text"].astype(str)
        else:
            rt = hits["root_text"].astype(str) if "root_text" in hits.columns else ""
            bt = hits["body_text"].astype(str) if "body_text" in hits.columns else ""
            full_text = (rt.astype(str) + "\n" + bt.astype(str)).astype(str)

        out = pd.DataFrame({
            "doc_id": doc_id,
            "kind": "micro",
            "source_username": hits["source_username"].astype(str) if "source_username" in hits.columns else "micro",
            "source_title": hits["source_title"].astype(str) if "source_title" in hits.columns else "micro",
            "full_text": full_text.astype(str),
            "parent_text": "",
            "root_text": hits["root_text"].astype(str) if "root_text" in hits.columns else "",
            "permalink": hits["root_permalink"].astype(str) if "root_permalink" in hits.columns else "",
        })

        dt_col = "last_dt" if "last_dt" in hits.columns else ("start_dt" if "start_dt" in hits.columns else None)
        out["msg_date"] = pd.to_datetime(hits[dt_col], errors="coerce", utc=True) if dt_col else pd.NaT

        if "score" in hits.columns:
            out["score"] = pd.to_numeric(hits["score"], errors="coerce")

        out["_concat_txt"] = out["full_text"].astype(str).map(_lower).str.strip()
        out["_row_idx"] = np.arange(len(out), dtype=np.int64)

        main_log.info(
            "[FAISS:micro:count] selected=%d (top_k=%d thr=%.3f)",
            len(out), COUNT_MICRO_TOP_K, COUNT_MIN_SCORE_MICRO
        )
        return out

    except Exception as e:
        main_log.warning("[FAISS:micro:count] search failed: %s", e)
        return pd.DataFrame()


# ===================== ПЕРИОД =====================
MONTH_LEX = {
    1: ("январ",),
    2: ("феврал",),
    3: ("март",),
    4: ("апрел",),
    5: ("май", "мая"),
    6: ("июн",),
    7: ("июл",),
    8: ("август",),
    9: ("сентябр",),
    10: ("октябр",),
    11: ("ноябр",),
    12: ("декабр",),
}
YEAR_RE = re.compile(r"\b(20\d{2})\b")
NUM_MM_YYYY = re.compile(r"\b(0?[1-9]|1[0-2])[./\-](20\d{2})\b")

RECENT_DAYS_RE = re.compile(
    r"(?:последн\w*\s+(\d{1,3})\s*(?:дн|дня|дней)|за\s+(\d{1,3})\s*(?:дн|дня|дней))",
    re.I,
)

def _extract_recent_days(q: str) -> int:
    ql = _lower(q)
    if re.search(r"\bнедел", ql):
        return 7
    if re.search(r"\bмесяц", ql):
        return 30
    if re.search(r"\bсутк", ql):
        return 1
    m = RECENT_DAYS_RE.search(ql)
    if not m:
        return 0
    a = m.group(1) or m.group(2)
    try:
        return int(a)
    except Exception:
        return 0

def _apply_recent_days_filter(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if df.empty or "msg_date" not in df.columns:
        return df
    n = _extract_recent_days(q)
    if not n:
        return df
    cutoff = datetime.now(timezone.utc) - timedelta(days=n)
    out = df.loc[df["msg_date"] >= cutoff].copy()
    main_log.info("[PERIOD] last_%ddays -> %d/%d", n, len(out), len(df))
    return out

def _apply_period_filter(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if df.empty or "msg_date" not in df.columns:
        return df

    ql = _lower(q)
    years: Set[int] = set(int(y) for y in YEAR_RE.findall(ql))
    pairs: Set[Tuple[int, int]] = set()

    for mm, yy in NUM_MM_YYYY.findall(ql):
        pairs.add((int(yy), int(mm)))

    months_found: Set[int] = set()
    for m, stems in MONTH_LEX.items():
        if any(st in ql for st in stems):
            months_found.add(m)

    if months_found and years:
        for y in years:
            for m in months_found:
                pairs.add((y, m))

    dtc = df["msg_date"]

    if pairs:
        mask = pd.Series(False, index=df.index)
        for y, m in sorted(pairs):
            mask |= ((dtc.dt.year == y) & (dtc.dt.month == m))
        out = df.loc[mask].copy()
        main_log.info("[PERIOD] pairs=%s -> %d", sorted(pairs), len(out))
        return out

    if years:
        out = df.loc[dtc.dt.year.isin(sorted(years))].copy()
        main_log.info("[PERIOD] years=%s -> %d", sorted(years), len(out))
        return out

    if months_found:
        out = df.loc[dtc.dt.month.isin(sorted(months_found))].copy()
        main_log.info("[PERIOD] months=%s -> %d", sorted(months_found), len(out))
        return out

    return df


# ===================== ЛОКАЦИЯ (MVP) =====================
REQ_MCD = re.compile(r"\bмцд\s*[-]?\s*([1-4])\b|\bd\s*[-]?\s*([1-4])\b", re.I)
MCD_PAT = {
    1: re.compile(r"\b(?:мцд[\s\-]*1|d[\s\-]*1|диаметр\w*\s*1)\b", re.I),
    2: re.compile(r"\b(?:мцд[\s\-]*2|d[\s\-]*2|диаметр\w*\s*2)\b", re.I),
    3: re.compile(r"\b(?:мцд[\s\-]*3|d[\s\-]*3|диаметр\w*\s*3)\b", re.I),
    4: re.compile(r"\b(?:мцд[\s\-]*4|d[\s\-]*4|диаметр\w*\s*4)\b", re.I),
}

def _parse_mcd_lines(q: str) -> List[int]:
    ql = _lower(q)
    lines: Set[int] = set()
    for m in REQ_MCD.finditer(ql):
        for g in m.groups():
            if g and g.isdigit():
                lines.add(int(g))
    return sorted(lines)

def _apply_location_scope(df: pd.DataFrame, q: str) -> Tuple[pd.DataFrame, Dict]:
    if df.empty:
        return df, {}

    ql = _lower(q)
    txt = df["_concat_txt"]

    meta: Dict = {"mcd": [], "mcc": False, "stations": []}
    before0 = len(df)

    # МЦД-n
    lines = _parse_mcd_lines(q)
    if lines:
        before = len(df)
        mask = pd.Series(False, index=df.index)
        for ln in lines:
            mask |= txt.str.contains(MCD_PAT[ln], na=False, regex=True)
        df = df.loc[mask].copy()
        meta["mcd"] = lines
        main_log.info("[SCOPE] MCD lines=%s -> %d/%d", lines, len(df), before)

    # МЦД (любой)
    if (("мцд" in ql) or ("диаметр" in ql)) and not lines:
        before = len(df)
        su = df.get("source_username", pd.Series([""] * len(df), index=df.index)).astype(str).map(_lower)
        any_mcd_pat = r"\bмцд\b|\bдиаметр\w*\b|\bd\s*[-]?\s*[1-4]\b|\bмцд[\s\-]*[1-4]\b"
        mask = (
            txt.str.contains(any_mcd_pat, na=False, regex=True)
            | su.str.contains(r"diameters|mcd|мцд|d[1-4]", na=False, regex=True)
        )
        df = df.loc[mask].copy()
        meta["mcd"] = ["any"]
        main_log.info("[SCOPE] MCD any -> %d/%d", len(df), before)

    # МЦК
    if re.search(r"\bмцк\b|московск\w+\s+центральн\w+\s+кольц\w+", ql):
        before = len(df)
        mask = txt.str.contains(r"\bмцк\b", na=False, regex=True) | txt.str.contains("центральн", na=False)
        df = df.loc[mask].copy()
        meta["mcc"] = True
        main_log.info("[SCOPE] MCC -> %d/%d", len(df), before)

    # станции
    stations = _stations_from_query(q)
    if stations:
        before = len(df)
        m = pd.Series(False, index=df.index)
        for st in stations:
            pat = _station_pat(st)
            if pat is None:
                continue
            m |= txt.str.contains(pat, na=False, regex=True)
        df = df.loc[m].copy()
        meta["stations"] = stations
        main_log.info("[SCOPE] stations=%s -> %d/%d", stations, len(df), before)

    if meta.get("mcd") or meta.get("mcc") or meta.get("stations"):
        main_log.info("[SCOPE] done meta=%s total %d -> %d", meta, before0, len(df))

    return df, meta


# ===================== ЛЕКСИКА (MVP) =====================
STOP = {
    "сколько", "количество", "число",
    "покажи", "показать", "выведи", "дай", "найди", "найти",
    "пример", "примеры", "примера",
    "сводка", "итоги", "итог", "основные", "главные", "дайджест",

    "жалобы", "жалоба", "жалоб",
    "проблемы", "проблема", "проблем",
    "случаи", "случай", "случаев",
    "обсуждение", "обсуждения", "обсуждений",
    "комментарий", "комментарии", "комментариев", "комментов",
    "сообщение", "сообщения", "сообщений",
    "пост", "посты", "постов",
    "отзывы", "отзыв",

    "что", "где", "на", "по", "про",
    "какие", "какой", "какая", "какое", "каковы",
    "кто", "почему", "зачем", "как", "когда",
    "куда", "откуда", "есть", "ли",

    "пассажиры", "пассажир",
    "мцд", "мцк", "метро", "линия", "ветка", "станция", "станции",

    "январь", "февраль", "март", "апрель", "май", "июнь", "июль",
    "август", "сентябрь", "октябрь", "ноябрь", "декабрь",

    "последние", "последний", "дней", "дня", "день",
    "неделя", "неделю", "недели", "месяц", "месяца", "месяцев",
    "год", "года", "году", "годом", "годы", "лет",

    "запрос", "запросы", "что-то", "чтото", "пишут", "говорят", "известно"
}

def _content_tokens(q: str) -> List[str]:
    ql = _lower(q)
    toks = [t for t in TOKEN_RE.findall(ql) if len(t) >= 3]
    out = []
    for t in toks:
        if t in STOP:
            continue
        if t.isdigit():
            continue
        if re.fullmatch(r"20\d{2}", t):
            continue
        if re.fullmatch(r"мцд-?\d+", t):
            continue
        if re.fullmatch(r"d\d+", t):
            continue
        out.append(t)

    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def _lex_masks(
    df: pd.DataFrame,
    q: str,
    text_col: str = "_concat_txt",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    tokens = _content_tokens(q)
    if df.empty or not tokens:
        main_log.info("[LEX] tokens=%s -> or=%d and=%d (total=%d, col=%s)", tokens, 0, 0, len(df), text_col)
        return np.zeros(len(df), bool), np.zeros(len(df), bool), []

    if text_col not in df.columns:
        text_col = "_concat_txt"

    txt = df[text_col].fillna("").astype(str)

    per = []
    used = []
    for t in tokens:
        stem = re.escape(t[:5])
        pat = re.compile(r"\b" + stem + r"\w*", re.I)
        m = txt.str.contains(pat, na=False, regex=True).values
        per.append(m)
        used.append(t)

    mask_or = per[0].copy()
    for m in per[1:]:
        mask_or |= m

    mask_and = per[0].copy()
    for m in per[1:]:
        mask_and &= m

    main_log.info(
        "[LEX] tokens=%s -> or=%d and=%d (total=%d, col=%s)",
        used, int(mask_or.sum()), int(mask_and.sum()), len(df), text_col
    )
    return mask_or, mask_and, used





# ===================== EXPORT (public clean) =====================
EXPORT_PREFER_COLS = [
    "msg_date",
    "_count_bucket",
    "digest_topic_title",
    "digest_topic_key",
    "source_username",
    "source_title",
    "kind",
    "full_text",
    "parent_text",
    "root_text",
    "permalink",
    "root_chat_id",
    "root_msg_id",
    "root_permalink",
]

EXPORT_DROP_TECH_COLS = {
    "id", "row_id",
    "chat_id", "msg_id",
    "parent_msg_id",
    "sender_id",
    "peer_id", "access_hash",
    "raw_json",
    "source_id",
    "doc_id",
    "thread_id", "doc_kind", "body_text", "n_items", "start_dt", "last_dt",
    "score",
}

EXPORT_COLS_RU = {
    "msg_date": "Дата",
    "_count_bucket": "Причина попадания",
    "digest_topic_title": "Тема дайджеста",
    "digest_topic_key": "Код темы дайджеста",
    "source_username": "Источник (username)",
    "source_title": "Источник (канал/чат)",
    "kind": "Тип",
    "full_text": "Текст",
    "parent_text": "Текст (reply_to)",
    "root_text": "Текст (корень)",
    "permalink": "Ссылка",
    "root_chat_id": "ID чата (корень)",
    "root_msg_id": "ID сообщения (корень)",
    "root_permalink": "Ссылка (корень)",
}

EXPORT_KEYWORD_RE = re.compile(
    r"(задерж|опозда|сбой|отмен|турникет|валидатор|оплат|qr|"
    r"холод|жарк|гряз|толп|давк|безопас|полици|драка|"
    r"лифт|эскалатор|платформ|перрон|касс|переход|выход|вход)",
    re.IGNORECASE
)

def _compress_discussion_text(body_text: str) -> str:
    """
    Ужимает длинное обсуждение:
    - оставляет начало
    - выбирает информативные строки по ключевым словам
    - оставляет хвост
    - ограничивает общий размер
    """
    body = (body_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not body:
        return ""

    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    if not lines:
        return ""

    # если коротко — не трогаем
    if len(body) <= EXPORT_THREAD_BODY_MAX_CHARS and len(lines) <= (EXPORT_THREAD_HEAD_LINES + EXPORT_THREAD_TAIL_LINES + 4):
        return body

    selected_idx = []
    total = len(lines)

    # 1) начало
    selected_idx.extend(range(min(EXPORT_THREAD_HEAD_LINES, total)))

    # 2) строки с полезными ключевыми словами
    keyword_hits = []
    for i, line in enumerate(lines):
        if EXPORT_KEYWORD_RE.search(line):
            keyword_hits.append(i)
    if keyword_hits:
        selected_idx.extend(keyword_hits[:EXPORT_THREAD_KEY_LINES])

    # 3) хвост
    tail_start = max(0, total - EXPORT_THREAD_TAIL_LINES)
    selected_idx.extend(range(tail_start, total))

    # уникализируем и сортируем
    selected_idx = sorted(set(i for i in selected_idx if 0 <= i < total))

    parts = []
    prev_i = None
    skipped_blocks = 0

    for i in selected_idx:
        if prev_i is not None and i - prev_i > 1:
            gap = i - prev_i - 1
            skipped_blocks += gap
            parts.append(f"... [пропущено {gap} реплик] ...")
        parts.append(lines[i])
        prev_i = i

    result = "\n".join(parts).strip()

    # финальный страховочный лимит по символам
    if len(result) > EXPORT_THREAD_BODY_MAX_CHARS:
        result = result[:EXPORT_THREAD_BODY_MAX_CHARS].rstrip() + "\n... [обрезано по длине]"

    return result

def _pretty_public_text(row) -> str:
    doc_kind = str(row.get("doc_kind") or "").strip().lower()
    root_text = str(row.get("root_text") or "").strip()
    body_text = str(row.get("body_text") or "").strip()
    full_text = str(row.get("full_text") or "").strip()

    if doc_kind in {"channel", "thread", "post_thread"}:
        parts = []
        if root_text:
            parts.append("Пост:\n" + root_text)

        if body_text:
            body = _compress_discussion_text(body_text)
            parts.append("Обсуждение:\n" + body)

        return "\n\n".join([p for p in parts if p]).strip() or full_text

    if doc_kind in {"chat_reply", "micro"}:
        if body_text:
            body = _compress_discussion_text(body_text)
            if root_text:
                return f"Корневое сообщение:\n{root_text}\n\nОбсуждение:\n{body}".strip()
            return f"Обсуждение:\n{body}".strip()

    return full_text

def _prepare_export_public(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    
    if "kind" in out.columns:
        k = out["kind"].fillna("").astype(str).str.lower()
        dk = out.get("doc_kind", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str).str.lower()

        def _map_kind(kind_val: str, doc_kind_val: str) -> str:
            if doc_kind_val in {"channel", "thread", "post_thread"}:
                return "пост с обсуждением"
            if doc_kind_val in {"chat_reply", "micro"}:
                return "обсуждение"

            if "forward" in kind_val or "fwd" in kind_val:
                return "пересланное"
            if "comment" in kind_val:
                return "комментарий"
            if "post" in kind_val:
                return "пост"
            if "chat" in kind_val:
                return "сообщение чата"
            if "thread" in kind_val:
                return "тред"

            return "сообщение"

        out["kind"] = [
            _map_kind(kind_val, doc_kind_val)
            for kind_val, doc_kind_val in zip(k.tolist(), dk.tolist())
        ]

    # Красивый многострочный текст для thread/micro документов
    if "full_text" in out.columns:
        out["full_text"] = out.apply(_pretty_public_text, axis=1)

        

    drop_cols = [c for c in out.columns if c.startswith("_") and c != "_count_bucket"]
    out = out.drop(columns=drop_cols, errors="ignore")
    out = out.drop(columns=[c for c in EXPORT_DROP_TECH_COLS if c in out.columns], errors="ignore")

    cols = [c for c in EXPORT_PREFER_COLS if c in out.columns]
    if cols:
        out = out[cols].copy()

    out = out.rename(columns={c: EXPORT_COLS_RU.get(c, c) for c in out.columns})
    return out

def build_digest_topic_docs(df: pd.DataFrame, topic_key: str, topic_title: str) -> pd.DataFrame:
    """
    Собирает компактные документы для digest/export:
    - если есть root_chat_id/root_msg_id -> агрегируем обсуждение в один документ
    - если нет root-структуры -> оставляем как одиночное сообщение

    Это нужно, чтобы:
    1) не дублировать root_text на каждый комментарий,
    2) вернуть в export человекочитаемое "Пост / Обсуждение",
    3) уменьшить шум в summary.
    """
    if df is None or df.empty:
        return df

    work = df.copy()

    # чтобы дальше не упасть на отсутствующих колонках
    for col in [
        "msg_date", "source_username", "source_title", "kind", "full_text",
        "parent_text", "root_text", "permalink", "root_chat_id", "root_msg_id",
        "root_permalink", "doc_kind"
    ]:
        if col not in work.columns:
            work[col] = ""

    # единый topic label для export
    work["digest_topic_key"] = topic_key
    work["digest_topic_title"] = topic_title

    has_root_cols = ("root_chat_id" in work.columns and "root_msg_id" in work.columns)

    if not has_root_cols:
        return work

    # --- 1) строки, у которых есть корень -> агрегируем ---
    rooted = work[
        work["root_chat_id"].notna() &
        work["root_msg_id"].notna()
    ].copy()

    # --- 2) строки без корня -> оставляем как есть ---
    single = work[
        work["root_chat_id"].isna() |
        work["root_msg_id"].isna()
    ].copy()

    docs = []

    if not rooted.empty:
        grp_cols = ["root_chat_id", "root_msg_id"]

        for (rch, rmsg), g in rooted.groupby(grp_cols, dropna=False):
            g = g.sort_values("msg_date", ascending=True, na_position="last").copy()

            root_text = str(g["root_text"].dropna().astype(str).replace("", pd.NA).dropna().iloc[0]) \
                if not g["root_text"].dropna().astype(str).replace("", pd.NA).dropna().empty else ""

            source_username = str(g["source_username"].dropna().astype(str).iloc[0]) if "source_username" in g.columns else ""
            source_title = str(g["source_title"].dropna().astype(str).iloc[0]) if "source_title" in g.columns else ""
            root_permalink = str(g["root_permalink"].dropna().astype(str).iloc[0]) if "root_permalink" in g.columns else ""
            msg_date = g["msg_date"].max() if "msg_date" in g.columns else pd.NaT

            # Собираем обсуждение из уникальных реплик, не повторяя root_text
            body_lines = []
            seen = set()

            for _, row in g.iterrows():
                txt = str(row.get("full_text") or "").strip()
                if not txt:
                    continue
                if root_text and txt == root_text:
                    continue

                norm = _normalize_ws(txt)
                if not norm or norm in seen:
                    continue
                seen.add(norm)

                # оставляем реплики как есть; сжатие произойдёт позже в _prepare_export_public
                body_lines.append(norm)

            body_text = "\n".join(body_lines).strip()
            full_text = "\n".join([x for x in [root_text, body_text] if x]).strip()

            docs.append({
                "msg_date": msg_date,
                "source_username": source_username,
                "source_title": source_title,
                "kind": "thread",
                "doc_kind": "channel" if root_text else "chat_reply",
                "full_text": full_text,
                "parent_text": "",
                "root_text": root_text,
                "body_text": body_text,
                "permalink": root_permalink,
                "root_chat_id": rch,
                "root_msg_id": rmsg,
                "root_permalink": root_permalink,
                "digest_topic_key": topic_key,
                "digest_topic_title": topic_title,
            })

    docs_df = pd.DataFrame(docs)

    # Для одиночных сообщений тоже проставим topic label
    if not single.empty:
        single = single.copy()
        single["digest_topic_key"] = topic_key
        single["digest_topic_title"] = topic_title

    if docs_df.empty and single.empty:
        return work.head(0).copy()
    if docs_df.empty:
        return single
    if single.empty:
        return docs_df

    return pd.concat([docs_df, single], ignore_index=True, sort=False)

def _log_answer_text(text: str, limit: int = 2000) -> None:
    try:
        s = (text or "").strip()
        if not s:
            main_log.info("[ANSWER] <empty>")
            return
        if len(s) > limit:
            main_log.info("[ANSWER] %s ... [truncated %d chars]", s[:limit], len(s) - limit)
        else:
            main_log.info("[ANSWER] %s", s)
    except Exception as e:
        main_log.warning("[ANSWER] log failed: %s", e)

def _export_df(df: pd.DataFrame, basename: str) -> Optional[str]:
    if df is None or df.empty:
        return None

    base = _safe_name(basename)[:120]
    rows = len(df)

    out = _prepare_export_public(df)

    path_xlsx = os.path.join(EXPORT_DIR, f"{base}.xlsx")
    try:
        if rows <= XLSX_ROWS_MAX:
            with pd.ExcelWriter(path_xlsx, engine="xlsxwriter") as wr:
                # Важно: после _prepare_export_public колонка уже называется "Дата"
                if "Дата" in out.columns:
                    try:
                        out["Дата"] = pd.to_datetime(out["Дата"], errors="coerce", utc=True).dt.tz_convert(None)
                    except Exception:
                        try:
                            out["Дата"] = pd.to_datetime(out["Дата"], errors="coerce").dt.tz_localize(None)
                        except Exception:
                            out["Дата"] = out["Дата"].astype(str)

                elif "msg_date" in out.columns:
                    try:
                        out["msg_date"] = pd.to_datetime(out["msg_date"], errors="coerce", utc=True).dt.tz_convert(None)
                    except Exception:
                        try:
                            out["msg_date"] = pd.to_datetime(out["msg_date"], errors="coerce").dt.tz_localize(None)
                        except Exception:
                            out["msg_date"] = out["msg_date"].astype(str)

                out.to_excel(wr, index=False, sheet_name="data")
                ws = wr.sheets["data"]

                cols = list(out.columns)
                text_fmt = wr.book.add_format({"text_wrap": True, "valign": "top"})

                for i, c in enumerate(cols):
                    if c in ("text", "full_text", "parent_text", "root_text", "Текст", "Текст (reply_to)", "Текст (корень)"):
                        ws.set_column(i, i, XLSX_WIDE_COL, text_fmt)
                    elif c in ("permalink", "root_permalink", "Ссылка", "Ссылка (корень)"):
                        ws.set_column(i, i, 45)
                    else:
                        ws.set_column(i, i, 18)

                ws.freeze_panes(1, 0)
                ws.autofilter(0, 0, max(0, rows), max(0, len(cols) - 1))

            main_log.info("[EXPORT] XLSX -> %s (%d rows)", path_xlsx, rows)
            return path_xlsx
    except Exception as e:
        main_log.warning("[EXPORT] XLSX failed (%s), fallback to CSV", e)

    path_csv = os.path.join(EXPORT_DIR, f"{base}.csv")
    out2 = out.copy()

    if "Дата" in out2.columns:
        try:
            out2["Дата"] = pd.to_datetime(out2["Дата"], errors="coerce", utc=True).dt.tz_convert(None)
        except Exception:
            try:
                out2["Дата"] = pd.to_datetime(out2["Дата"], errors="coerce").dt.tz_localize(None)
            except Exception:
                out2["Дата"] = out2["Дата"].astype(str)

    if rows > CSV_ROWS_MAX:
        out2 = out2.head(CSV_ROWS_MAX).copy()

    out2.to_csv(path_csv, index=False, encoding="utf-8-sig", sep=";", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    main_log.info("[EXPORT] CSV -> %s (%d rows)", path_csv, len(out2))
    return path_csv


# ===================== INTENT =====================
INTENT_PAT = {
    "greeting": re.compile(r"\b(привет|здравствуй|добрый|доброе)\b", re.I),
    "examples": re.compile(r"\b(пример|примеры|покажи|выведи|дай)\b", re.I),
    "count": re.compile(r"\b(сколько|количество|число)\b", re.I),

    # Явные признаки именно сводки
    "summary": re.compile(
        r"\b(?:сводка|итог[аи]?|основные|главные|общая\s+картина|дайджест)\b",
        re.I,
    ),

    # Свободные вопросительные / исследовательские запросы
    "qa": re.compile(
        r"(?:\?"
        r"|\b(?:почему|зачем|как|какие|какая|какой|каковы|"
        r"что\s+известно|что\s+говорят|что\s+пишут|"
        r"в\s+чем|есть\s+ли|бывают\s+ли|связано\s+ли|"
        r"из-за\s+чего|из\s+за\s+чего)\b)",
        re.I,
    ),
}

def _parse_intent(q: str) -> str:
    ql = _lower(q)

    if INTENT_PAT["greeting"].search(ql):
        return "greeting"

    has_ex = INTENT_PAT["examples"].search(ql) is not None
    has_cnt = INTENT_PAT["count"].search(ql) is not None
    has_sum = INTENT_PAT["summary"].search(ql) is not None
    has_qa = INTENT_PAT["qa"].search(ql) is not None

    # examples — самый приоритетный явный режим
    if has_ex:
        return "examples"

    # count — только если это не summary и не qa-вопрос
    if has_cnt:
        return "count"

    # Явная сводка
    if has_sum:
        return "summary"

    # Любой вопрос / "что пишут", "что говорят", "какие жалобы" и т.д.
    if has_qa:
        return "qa"

    # По умолчанию свободный текст считаем QA, а не count
    return "qa"

# ===================== SUMMARY fallback =====================
def _summary_fallback(df: pd.DataFrame, q: str, k: int = 7) -> str:
    txt = " ".join(df["full_text"].astype(str).tolist()).lower()
    words = [w for w in re.findall(r"[а-яa-z]{4,}", txt) if w not in STOP]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    lines = [f"{BULLET} топ-слова: " + ", ".join([f"{w}({c})" for w, c in top])]

    ex = df.head(k)
    for _, r in ex.iterrows():
        s = _normalize_ws(str(r.get("full_text") or ""))[:300]
        p = _normalize_ws(str(r.get("parent_text") or ""))[:200]
        rt = _normalize_ws(str(r.get("root_text") or ""))[:200]
        src = r.get("source_username") or "?"
        dtv = r.get("msg_date")
        if p:
            lines.append(f"{BULLET} {src} | {dtv}: {s} (reply_to: {p})")
        elif rt and len(s) < 70:
            lines.append(f"{BULLET} {src} | {dtv}: {s} (к посту: {rt})")
        else:
            lines.append(f"{BULLET} {src} | {dtv}: {s}")

    return "✅ Сводка (fallback без LLM)\n\n" + "\n".join(lines)


# ===================== SUMMARY text helper =====================
def _make_summary_text(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "full_text" not in out.columns:
        out["full_text"] = ""
    if "parent_text" not in out.columns:
        out["parent_text"] = ""
    if "root_text" not in out.columns:
        out["root_text"] = ""

    ft = out["full_text"].astype(str)
    pt = out["parent_text"].astype(str)
    rt = out["root_text"].astype(str)

    short = ft.str.len().fillna(0) < 70
    has_parent = pt.str.len().fillna(0) > 0
    has_root = rt.str.len().fillna(0) > 0

    out["summary_text"] = ft
    out.loc[short & has_parent, "summary_text"] = ft + " (в ответ на: " + pt.str.slice(0, 300) + ")"
    out.loc[short & (~has_parent) & has_root, "summary_text"] = ft + " (к посту: " + rt.str.slice(0, 300) + ")"
    return out

def _extract_mcd_lines_from_text(text: str) -> List[int]:
    """
    Ищем упоминания МЦД-линий в произвольном тексте:
    мцд-1, мцд 1, d1, d-1, диаметр 1 и т.п.
    """
    if not text:
        return []

    ql = _lower(text)
    lines: Set[int] = set()

    for ln, pat in MCD_PAT.items():
        try:
            if pat.search(ql):
                lines.add(int(ln))
        except Exception:
            pass

    return sorted(lines)


def _extract_station_hits_from_text(text: str) -> List[str]:
    """
    Возвращает список станций, которые явно встретились в тексте.
    Использует stations.csv через уже существующий load_stations().
    """
    if not text:
        return []

    st = load_stations()
    if st is None or st.empty or "name_norm" not in st.columns:
        return []

    ql = _lower(text)
    hits: List[str] = []

    for name in st["name_norm"].dropna().unique():
        name = str(name).strip()
        if not name:
            continue
        pat = _station_pat(name)
        if pat is not None and pat.search(ql):
            hits.append(name)

    # дедуп с сохранением порядка
    seen = set()
    out = []
    for x in hits:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _pretty_station_name(name_norm: str) -> str:
    """
    Преобразуем station normal form в более читаемый вид для текста.
    """
    s = (name_norm or "").strip()
    if not s:
        return s
    parts = re.split(r"[\s\-]+", s)
    return " ".join(p.capitalize() for p in parts if p)


def _summary_title_tokens(title: str) -> List[str]:
    """
    Токены из названия тега/пункта сводки.
    Например:
      'зацепинг'
      'задержки поездов'
      'неисправность валидатора'
    """
    s = _lower(title or "")
    s = s.replace("_", " ").replace("-", " ")
    toks = [t for t in TOKEN_RE.findall(s) if len(t) >= 3]

    # отдельный stop-лист именно для заголовков сводки
    bad = {
        "жалобы", "жалоба", "жалоб",
        "проблемы", "проблема", "проблем",
        "случаи", "случай", "случаев",
        "ситуации", "ситуация",
        "вопрос", "вопросы",
        "работы", "работа",
        "движения", "движение",
    }

    out = []
    seen = set()
    for t in toks:
        if t in STOP or t in bad:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _summary_row_text(row: pd.Series) -> str:
    """
    Берём максимум контекста по строке для привязки тега к станции/линии.
    """
    parts = [
        str(row.get("summary_text") or ""),
        str(row.get("full_text") or ""),
        str(row.get("parent_text") or ""),
        str(row.get("root_text") or ""),
        
    ]
    return "\n".join([p for p in parts if p]).strip()


def _subset_for_summary_title(df: pd.DataFrame, title: str) -> pd.DataFrame:
    """
    Подбираем subset строк, относящихся к конкретному заголовку summary.
    Делаем максимально просто и детерминированно: по токенам заголовка.
    """
    if df is None or df.empty:
        return df

    toks = _summary_title_tokens(title)
    if not toks:
        return df.head(0).copy()

    work = df.copy()

    if "_summary_geo_text" not in work.columns:
        work["_summary_geo_text"] = work.apply(_summary_row_text, axis=1).map(_lower)

    txt = work["_summary_geo_text"]

    per = []
    for t in toks:
        stem = re.escape(t[:5])
        pat = re.compile(r"\b" + stem + r"\w*", re.I)
        per.append(txt.str.contains(pat, na=False, regex=True))

    if not per:
        return work.head(0).copy()

    # сначала пытаемся AND, если слишком узко — OR
    mask_and = per[0].copy()
    for m in per[1:]:
        mask_and &= m

    mask_or = per[0].copy()
    for m in per[1:]:
        mask_or |= m

    if int(mask_and.sum()) >= 2:
        out = work.loc[mask_and].copy()
    elif int(mask_or.sum()) > 0:
        out = work.loc[mask_or].copy()
    else:
        out = work.head(0).copy()

    if "msg_date" in out.columns:
        out = out.sort_values("msg_date", ascending=False, na_position="last")

    return out


def _build_geo_suffix_for_subset(
    df_subset: pd.DataFrame,
    max_lines: int = SUMMARY_GEO_MAX_LINES,
    max_stations: int = SUMMARY_GEO_MAX_STATIONS,
    min_count: int = SUMMARY_GEO_MIN_MENTION_COUNT,
) -> str:
    """
    По subset строк строим короткую географическую подпись:
    - линии
    - станции
    """
    if df_subset is None or df_subset.empty:
        return ""

    line_counts: Dict[int, int] = {}
    station_counts: Dict[str, int] = {}

    for _, r in df_subset.iterrows():
        txt = _summary_row_text(r)

        # линии
        lines = _extract_mcd_lines_from_text(txt)
        for ln in lines:
            line_counts[ln] = line_counts.get(ln, 0) + 1

        # станции
        stations = _extract_station_hits_from_text(txt)
        for st in stations:
            station_counts[st] = station_counts.get(st, 0) + 1

    top_lines = [
        f"МЦД-{ln}"
        for ln, cnt in sorted(line_counts.items(), key=lambda x: (-x[1], x[0]))
        if cnt >= min_count
    ][:max_lines]

    top_stations = [
        _pretty_station_name(st)
        for st, cnt in sorted(station_counts.items(), key=lambda x: (-x[1], x[0]))
        if cnt >= min_count
    ][:max_stations]

    parts: List[str] = []

    if top_lines:
        if len(top_lines) == 1:
            parts.append(f"Чаще встречается на {top_lines[0]}")
        else:
            parts.append(f"Чаще встречается на {', '.join(top_lines[:-1])} и {top_lines[-1]}")

    if top_stations:
        if len(top_stations) == 1:
            parts.append(f"Среди упоминаний выделяется станция {top_stations[0]}")
        else:
            parts.append(f"Среди упоминаний выделяются станции {', '.join(top_stations[:-1])} и {top_stations[-1]}")

    if not parts:
        return ""

    return ". " + "; ".join(parts) + "."


def enrich_summary_with_geo(summary_text: str, df: pd.DataFrame) -> str:
    """
    Пост-обработка готовой сводки:
    1) парсим bullet line
    2) берём title до первого ':'
    3) строим subset по title
    4) добавляем suffix с линиями/станциями
    """
    if not summary_text or df is None or df.empty:
        return summary_text

    lines = summary_text.splitlines()
    out_lines: List[str] = []

    for ln in lines:
        raw = (ln or "").rstrip()

        # обрабатываем только bullet-строки
        m = re.match(r"^\s*[🔹•\-\–\—\*]\s*(.+)$", raw)
        if not m:
            out_lines.append(raw)
            continue

        body = m.group(1).strip()

        # title берём до первого двоеточия
        p = body.find(":")
        if p <= 0:
            out_lines.append(raw)
            continue

        title = body[:p].strip()
        desc = body[p + 1 :].strip()

        if not title or not desc:
            out_lines.append(raw)
            continue

        try:
            subset = _subset_for_summary_title(df, title)
            suffix = _build_geo_suffix_for_subset(subset)
        except Exception as e:
            main_log.warning("[SUMMARY_GEO] failed for title=%r: %s", title, e)
            suffix = ""

        if suffix:
            out_lines.append(f"{BULLET} {title}: {desc}{suffix}")
        else:
            out_lines.append(raw)

    return "\n".join(out_lines)

def _sort_hits(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    if "score" in out.columns:
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
        if "msg_date" in out.columns:
            return out.sort_values(["score", "msg_date"], ascending=[False, False], na_position="last")
        return out.sort_values(["score"], ascending=[False], na_position="last")

    if "msg_date" in out.columns:
        return out.sort_values("msg_date", ascending=False, na_position="last")

    return out


def _dedup_key_col(df: pd.DataFrame) -> str:
    if "doc_id" in df.columns:
        return "doc_id"
    if "_concat_txt" in df.columns:
        return "_concat_txt"
    return df.columns[0]


def _select_qa_hits(
    df: pd.DataFrame,
    mask_or: np.ndarray,
    mask_and: np.ndarray,
    tokens: List[str],
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    base = _sort_hits(df)
    key_col = _dedup_key_col(base)

    if not tokens:
        main_log.info("[QA] no lexical tokens -> use semantic/top scope pool")
        return base.drop_duplicates(subset=[key_col], keep="first").head(QA_DF_CAP).copy()

    n_or = int(mask_or.sum())
    n_and = int(mask_and.sum())

    if n_and >= QA_MIN_AND_HITS:
        main_log.info("[QA] prefer AND (or=%d and=%d)", n_or, n_and)
        out = df.loc[mask_and].copy()
    elif n_and >= 1 and (n_or / max(1, n_and)) >= QA_AND_RATIO:
        main_log.info("[QA] narrow to AND by ratio (or=%d and=%d ratio=%.1f)", n_or, n_and, n_or / max(1, n_and))
        out = df.loc[mask_and].copy()
    elif n_or > 0:
        main_log.info("[QA] use OR (or=%d and=%d)", n_or, n_and)
        out = df.loc[mask_or].copy()
    else:
        main_log.info("[QA] zero lexical hits -> keep semantic/top scope pool")
        out = base.copy()

    out = _sort_hits(out)
    out = out.drop_duplicates(subset=[key_col], keep="first")

    if len(out) < min(5, QA_CONTEXT_TOP_K):
        rest = base.drop_duplicates(subset=[key_col], keep="first")
        rest = rest.loc[~rest[key_col].astype(str).isin(set(out[key_col].astype(str)))]
        need = max(0, QA_CONTEXT_TOP_K - len(out))
        if need > 0 and not rest.empty:
            out = pd.concat([out, rest.head(need)], ignore_index=True, sort=False)
            out = out.drop_duplicates(subset=[key_col], keep="first")

    return out.head(QA_DF_CAP).copy()

def _select_count_hits(
    df: pd.DataFrame,
    mask_or: np.ndarray,
    mask_and: np.ndarray,
    tokens: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Возвращает 3 набора для count/export:
    1) df_and      — явно подходит по всем токенам
    2) df_sem_or   — семантически в shortlist и подходит хотя бы по одному токену
    3) df_sem_only — семантически в shortlist, но lexical совпадений нет

    Идея:
    - count работает как QA-lite без LLM
    - число считаем по объединению этих трёх групп без дублей
    - в export показываем приоритетно: AND -> SEM+OR -> SEM only
    """
    if df is None or df.empty:
        empty = df.head(0).copy()
        return empty, empty, empty

    base = _sort_hits(df)
    key_col = _dedup_key_col(base)

    if not tokens:
        df_sem_only = base.drop_duplicates(subset=[key_col], keep="first").copy()
        empty = base.head(0).copy()
        return empty, empty, df_sem_only

    n_or = int(mask_or.sum())
    n_and = int(mask_and.sum())

    if n_and > 0:
        main_log.info("[COUNT] explicit AND block (or=%d and=%d)", n_or, n_and)
        df_and = df.loc[mask_and].copy()
    else:
        df_and = base.head(0).copy()

    if n_or > 0:
        df_sem_or = df.loc[mask_or].copy()
        if not df_and.empty:
            keys_and = set(df_and[key_col].astype(str))
            df_sem_or = df_sem_or.loc[~df_sem_or[key_col].astype(str).isin(keys_and)].copy()
    else:
        df_sem_or = base.head(0).copy()

    used_keys = set()
    if not df_and.empty:
        used_keys |= set(df_and[key_col].astype(str))
    if not df_sem_or.empty:
        used_keys |= set(df_sem_or[key_col].astype(str))

    # semantic only = осталось из shortlist после lexical блоков
    df_sem_only = base.loc[~base[key_col].astype(str).isin(used_keys)].copy()

    df_and = _sort_hits(df_and).drop_duplicates(subset=[key_col], keep="first")
    df_sem_or = _sort_hits(df_sem_or).drop_duplicates(subset=[key_col], keep="first")
    df_sem_only = _sort_hits(df_sem_only).drop_duplicates(subset=[key_col], keep="first")

    main_log.info(
        "[COUNT] buckets and=%d sem_or=%d sem_only=%d total=%d",
        len(df_and), len(df_sem_or), len(df_sem_only),
        len(pd.concat([df_and, df_sem_or, df_sem_only], ignore_index=True).drop_duplicates(subset=[key_col], keep="first"))
    )

    return df_and, df_sem_or, df_sem_only




def _build_qa_context_blocks(df: pd.DataFrame, limit: int = 12) -> List[str]:
    if df is None or df.empty:
        return []

    base = _sort_hits(df)

    seen = set()
    blocks: List[str] = []

    for _, r in base.iterrows():
        full_text = _normalize_ws(str(r.get("full_text") or ""))
        if not full_text:
            continue

        norm_key = _lower(full_text[:500])
        if not norm_key or norm_key in seen:
            continue
        seen.add(norm_key)

        source = _normalize_ws(str(r.get("source_username") or r.get("source_title") or "?"))
        kind = _normalize_ws(str(r.get("kind") or ""))
        dtv = r.get("msg_date")
        dt_txt = ""
        if pd.notna(dtv):
            try:
                dt_txt = pd.to_datetime(dtv, errors="coerce").strftime("%Y-%m-%d %H:%M")
            except Exception:
                dt_txt = str(dtv)

        parent = _normalize_ws(str(r.get("parent_text") or ""))
        root = _normalize_ws(str(r.get("root_text") or ""))
        link = _normalize_ws(str(r.get("permalink") or ""))

        text_main = full_text[:900]
        extra = ""
        if parent:
            extra = parent[:280]
            extra_label = "Контекст reply_to"
        elif root and _lower(root) != _lower(full_text):
            extra = root[:280]
            extra_label = "Корневой пост"
        else:
            extra_label = ""

        lines = []
        lines.append(f"Источник: {source}")
        if kind:
            lines.append(f"Тип: {kind}")
        if dt_txt:
            lines.append(f"Дата: {dt_txt}")
        lines.append(f"Сообщение: {text_main}")
        if extra:
            lines.append(f"{extra_label}: {extra}")
        if link:
            lines.append(f"Ссылка: {link}")

        blocks.append("\n".join(lines))

        if len(blocks) >= limit:
            break

    return blocks

# ===================== MAIN PIPE =====================
def process_query(user_query: str) -> Tuple[str, Optional[str]]:
    maybe_reload_on_bust_file()
    q = (user_query or "").strip()
    intent = _parse_intent(q)

    main_log.info("[QUERY] ? %s", q)
    main_log.info("[INTENT] %s", intent)

    if intent == "greeting":
        return (
            "✅ Ответ\n\n"
            "🔹 👋 Привет! Я анализирую сообщения из Telegram (посты+комментарии+чаты).\n"
            "🔹 Примеры:\n"
            "🔹 «сколько сообщений про мцд-3 за 2025»\n"
            "🔹 «покажи 3 примера про турникеты на мцд-4»\n"
            "🔹 «сводка по мцд-3 за январь 2026»\n"
            "🔹 «какие жалобы на расцепление вагонов?»",
            None,
        )

    # summary и qa работаем по threads view + micro-пулу
    
    use_threads_view = (
        intent in {"summary", "qa", "examples", "count"} and SUMMARY_USE_THREADS
    )
    view = THREADS_VIEW_NAME if use_threads_view else RAW_VIEW_NAME
    df = load_data(view)

    main_log.info("[SIZE] total=%d (view=%s)", len(df), view)

    
    # период / локация
    df0 = _apply_recent_days_filter(df, q)
    df1 = _apply_period_filter(df0, q)
    df2, loc_meta = _apply_location_scope(df1, q)

    

    # semantic shortlist для summary/qa
    if intent in {"summary", "qa", "examples", "count"}:
        if use_threads_view:
            if intent == "count":
                df2 = _faiss_filter_threads_for_count(df2, q)
            else:
                df2 = _faiss_filter_threads(df2, q)

        if intent == "count":
            df_micro = _micro_docs_from_faiss_for_count(q)
        else:
            df_micro = _micro_docs_from_faiss(q)
        if not df_micro.empty:
            df_micro = _apply_recent_days_filter(df_micro, q)
            df_micro = _apply_period_filter(df_micro, q)
            df_micro, _ = _apply_location_scope(df_micro, q)
            df_micro = _drop_noise_rows(df_micro)

            main_log.info("[MICRO] pool=%d", len(df_micro))

            df2 = pd.concat([df2, df_micro], ignore_index=True, sort=False)

            if "doc_id" in df2.columns:
                df2["doc_id"] = df2["doc_id"].astype(str)
                df2 = df2.drop_duplicates(subset=["doc_id"], keep="first")
            else:
                df2 = df2.drop_duplicates(subset=["_concat_txt"], keep="first")

            df2 = _sort_hits(df2)

    main_log.info("[SIZE] after scope=%d (meta=%s)", len(df2), loc_meta)

    if df2.empty:
        ans = "Не найдено подходящих сообщений за указанный период/локацию."
        main_log.info("[ANSWER_FULL] %s", ans)
        return ans, None

    

    mask_or, mask_and, tokens = _lex_masks(df2, q, text_col="_concat_txt")

    # ===== summary lexical behavior =====
    if intent == "summary" and tokens and int(mask_or.sum()) == 0:
        main_log.info("[LEX] zero hits by tokens=%s -> skip lex filter for summary", tokens)
        tokens = []

    if intent == "summary" and tokens:
        n_or = int(mask_or.sum())
        n_and = int(mask_and.sum())

        if n_and >= LEX_AND_MIN and (n_or / max(1, n_and)) >= LEX_AND_RATIO:
            main_log.info("[LEX] summary prefers AND (or=%d and=%d ratio=%.1f)", n_or, n_and, n_or / max(1, n_and))
            mask_or = mask_and

    # ===== count =====
   
    # ===== count =====
    if intent == "count":
        ql = _lower(q)

        label = "обсуждений"
        if re.search(r"\bкомментар", ql):
            label = "комментариев"
        elif re.search(r"\bсообщен", ql):
            label = "сообщений"
        elif re.search(r"\bпост", ql):
            label = "постов"
        elif re.search(r"\bчат", ql):
            label = "чат-сообщений"
        elif re.search(r"\bжалоб", ql):
            label = "жалоб"

        df_and, df_sem_or, df_sem_only = _select_count_hits(df2, mask_or, mask_and, tokens)

        pieces = [x for x in [df_and, df_sem_or, df_sem_only] if x is not None and not x.empty]
        if not pieces:
            ans = f"Не найдено подходящих {label} по теме запроса."
            _log_answer_text(ans)
            return ans, None

        exp_parts = []

        if not df_and.empty:
            tmp = df_and.copy()
            tmp["_count_bucket"] = "AND"
            exp_parts.append(tmp)

        if not df_sem_or.empty:
            tmp = df_sem_or.copy()
            tmp["_count_bucket"] = "SEM+LEX"
            exp_parts.append(tmp)

        if not df_sem_only.empty:
            tmp = df_sem_only.copy()
            tmp["_count_bucket"] = "SEM"
            exp_parts.append(tmp)

        exp = pd.concat(exp_parts, ignore_index=True, sort=False)

        key_col = _dedup_key_col(exp)
        exp = exp.drop_duplicates(subset=[key_col], keep="first")

        confirmed_parts = [x for x in [df_and, df_sem_or] if x is not None and not x.empty]

        if confirmed_parts:
            confirmed = pd.concat(confirmed_parts, ignore_index=True, sort=False)
            confirmed_key_col = _dedup_key_col(confirmed)
            confirmed = confirmed.drop_duplicates(subset=[confirmed_key_col], keep="first")
            n = len(confirmed)
        else:
            n = 0

        text = (
            f"✅ Ответ\n\n"
            f"🔹 Найдено релевантных {label}: {n}\n"
            f"🔹 Режим подсчёта: агрегированные обсуждения\n"
            f"🔹 Явные совпадения: {len(df_and)}\n"
            f"🔹 Семантически близкие + 1+ ключевое слово: {len(df_sem_or)}\n"
            f"🔹 Семантически близкие: {len(df_sem_only)}"
        )

        if len(exp) > COUNT_EXPORT_MAX:
            exp = exp.head(COUNT_EXPORT_MAX).copy()

        path = _export_df(exp, q)
        _log_answer_text(text)
        return text, path

    # ===== examples =====
    if intent == "examples":
        df_hits = _select_qa_hits(df2, mask_or, mask_and, tokens)

        if df_hits.empty:
            ans = "Не найдено подходящих сообщений по теме запроса (после фильтрации)."
            _log_answer_text(ans)
            return ans, None

        k = 3
        m = re.search(r"\b(\d{1,2})\b", _lower(q))
        if m:
            k = max(1, min(10, int(m.group(1))))

        base = _sort_hits(df_hits).head(min(EXAMPLES_EXPORT_MAX, len(df_hits))).copy()

        lines = []
        for _, r in base.head(k).iterrows():
            txt = _normalize_ws(_pretty_public_text(r))[:MAX_EXAMPLE_CHARS]
            src = r.get("source_username") or "?"
            dtv = r.get("msg_date")
            kind = r.get("kind") or ""
            link = r.get("permalink") or r.get("root_permalink") or ""

            suffix = f" ({kind}, {src}, {dtv})"
            if link:
                suffix += f" | {link}"

            lines.append(f"{BULLET} {txt}{suffix}")

        text = "✅ Ответ\n\n🔹 Примеры:\n" + "\n".join(lines)
        path = _export_df(base, q)
        _log_answer_text(text)
        return text, path

    # ===== qa =====
    if intent == "qa":
        df_hits = _select_qa_hits(df2, mask_or, mask_and, tokens)

        if df_hits.empty:
            ans = "Не найдено достаточно релевантных сообщений для ответа на вопрос."
            main_log.info("[ANSWER_FULL] %s", ans)
            return ans, None

        df_hits = _sort_hits(df_hits)
        if len(df_hits) > QA_DF_CAP:
            df_hits = df_hits.head(QA_DF_CAP).copy()

        context_blocks = _build_qa_context_blocks(df_hits, limit=QA_CONTEXT_TOP_K)
        main_log.info("[QA] context_blocks=%d", len(context_blocks))

        qa_text = ""
        if answer_with_context is not None:
            try:
                qa_text = answer_with_context(
                    query_text=q,
                    context_blocks=context_blocks,
                )
            except Exception as e:
                main_log.warning("[QA] answer_with_context failed: %s", e)

        if not qa_text:
            preview = _sort_hits(df_hits).head(QA_PREVIEW_EXAMPLES)
            lines = [f"Найдено релевантных сообщений: {len(df_hits)}"]
            for _, r in preview.iterrows():
                txt = _normalize_ws(str(r.get("full_text") or ""))[:350]
                link = _normalize_ws(str(r.get("permalink") or ""))
                if link:
                    lines.append(f"{BULLET} {txt} | {link}")
                else:
                    lines.append(f"{BULLET} {txt}")
            qa_text = "\n".join(lines)

        text = qa_text if (qa_text or "").lstrip().startswith("✅") else ("✅ Ответ\n\n" + qa_text)
        path = _export_df(df_hits, q + " _для_ответа")
        _log_answer_text(text)
        return text, path

    # ===== summary =====
    df_hits = df2.loc[mask_or].copy() if tokens else df2

    if df_hits.empty:
        ans = "Не найдено подходящих сообщений по теме запроса (после фильтрации)."
        main_log.info("[ANSWER_FULL] %s", ans)
        return ans, None

    if len(df_hits) < SUMMARY_MIN_DOCS_FOR_LLM:
        k = min(3, len(df_hits))
        base = df_hits.sort_values("msg_date", ascending=False).head(k)

        lines = [f"{BULLET} Найдено сообщений: {len(df_hits)}"]
        for _, r in base.iterrows():
            txt = _normalize_ws(str(r.get("full_text") or ""))[:350]
            link = r.get("permalink") or ""
            if link:
                lines.append(f"{BULLET} {txt} | {link}")
            else:
                lines.append(f"{BULLET} {txt}")

        path = _export_df(df_hits, q + " _для_сводки")
        return "\n".join(lines), path

    df_for_sum = df_hits
    if len(df_for_sum) > SUMMARY_DF_CAP:
        df_for_sum = df_for_sum.head(SUMMARY_DF_CAP).copy()
        main_log.info("[CAP] summary df cap=%d", SUMMARY_DF_CAP)

    df_for_sum2 = _make_summary_text(df_for_sum)

    if summarize_topn is not None:
        summ = summarize_topn(
            df=df_for_sum2,
            query_text=q,
            text_col="summary_text",
            datetime_col="msg_date",
            top_n=min(SUMMARY_TOPN, len(df_for_sum2)),
        )

        # enrich линиями/станциями поверх готовой сводки
        summ = enrich_summary_with_geo(summ, df_for_sum2)

        text = summ if (summ or "").lstrip().startswith("✅") else ("✅ Ответ\n\n" + summ)
    else:
        text = _summary_fallback(df_for_sum, q)

    path = _export_df(df_for_sum, q + " _для_сводки")
    _log_answer_text(text)
    return text, path



if __name__ == "__main__":
    print(process_query("сводка по мцд-4 за январь 2026"))