# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import faiss
from sentence_transformers import SentenceTransformer

from sqlalchemy import create_engine, text


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - faiss-micro - %(levelname)s - %(message)s"
)
log = logging.getLogger("faiss-micro")


# ---------------- CONFIG ----------------
PG_DSN = (os.getenv("PG_DSN") or "").strip()

FAISS_DIR = os.getenv("FAISS_MICRO_DIR", os.path.join(os.path.dirname(__file__), "faiss_micro"))
EMBED_MODEL = os.getenv("EMBEDDER", "intfloat/multilingual-e5-base")
EMB_BATCH = int(os.getenv("EMB_BATCH", "256"))
EMB_TRUNC = int(os.getenv("EMB_TRUNC", "1200"))

# Порог кластеризации (cosine, т.к. normalize_embeddings=True)
MICRO_THRESHOLD = float(os.getenv("MICRO_THRESHOLD", "0.82"))

# Разрыв между сообщениями (минут) -> новая сессия активности
SESSION_GAP_MIN = int(os.getenv("MICRO_SESSION_GAP_MIN", "90"))

# Максимум сообщений в одной сессии (защита)
SESSION_MAX_MSGS = int(os.getenv("MICRO_SESSION_MAX_MSGS", "1200"))

# Какие kind считать "чатами"
# если вдруг у тебя kind называется иначе — просто добавь сюда
CHAT_KINDS = os.getenv("MICRO_CHAT_KINDS", "chat,group,discussion,chat_msg").split(",")

# Минимум сообщений в кластере, чтобы его сохранять как документ
MIN_CLUSTER_SIZE = int(os.getenv("MICRO_MIN_CLUSTER_SIZE", "3"))

# Сколько сообщений максимум склеивать в один документ (чтобы не раздувать)
DOC_MAX_MSGS = int(os.getenv("MICRO_DOC_MAX_MSGS", "80"))


os.makedirs(FAISS_DIR, exist_ok=True)


def _device_pick() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def _clean_text(s: str) -> str:
    s = str(s or "").replace("\r", " ")
    s = " ".join(s.split())
    if EMB_TRUNC and len(s) > EMB_TRUNC:
        s = s[:EMB_TRUNC]
    return s


def get_engine():
    if not PG_DSN:
        raise SystemExit("PG_DSN не задан (.env)")
    return create_engine(PG_DSN, pool_pre_ping=True)


def load_free_chat_msgs(engine) -> pd.DataFrame:
    """
    Берём только "чаты" (не post/comment) и только сообщения без reply (parent_msg_id IS NULL),
    потому что микротреды нужны именно для каши.
    """
    kinds = [k.strip() for k in CHAT_KINDS if k.strip()]
    if not kinds:
        kinds = ["chat"]

    # NB: kind у тебя text -> фильтруем через IN
    sql = """
    SELECT
      doc_id,
      kind,
      source_username,
      source_title,
      chat_id,
      msg_id,
      msg_date,
      parent_msg_id,
      text,
      permalink
    FROM public.feedback_raw_v3
    WHERE kind = ANY(:kinds)
      AND chat_id IS NOT NULL
      AND msg_id IS NOT NULL
      AND msg_date IS NOT NULL
      AND COALESCE(text,'') <> ''
      AND parent_msg_id IS NULL
    ORDER BY chat_id, msg_date, msg_id;
    """

    df = pd.read_sql(text(sql), engine, params={"kinds": kinds})
    if df.empty:
        return df

    df["msg_date"] = pd.to_datetime(df["msg_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["msg_date", "chat_id", "msg_id"])
    df["chat_id"] = df["chat_id"].astype("int64")
    df["msg_id"] = df["msg_id"].astype("int64")
    df["doc_id"] = df["doc_id"].astype(str)

    df["text"] = df["text"].astype(str).map(_clean_text)
    df = df[df["text"].str.len() > 0].copy()
    return df


def split_sessions(df_chat: pd.DataFrame, gap_min: int) -> List[pd.DataFrame]:
    """
    df_chat отсортирован по времени.
    Режем на сессии по паузе > gap_min минут.
    """
    if df_chat.empty:
        return []

    times = df_chat["msg_date"].to_numpy()
    # разница между соседними
    deltas = np.diff(times).astype("timedelta64[s]").astype(np.int64)
    gap_sec = gap_min * 60

    # индексы разрывов
    cut_points = np.where(deltas > gap_sec)[0]
    # начало/конец сегментов
    starts = [0] + [int(i + 1) for i in cut_points]
    ends = [int(i + 1) for i in cut_points] + [len(df_chat)]

    sessions = []
    for a, b in zip(starts, ends):
        sub = df_chat.iloc[a:b].copy()
        if len(sub) > SESSION_MAX_MSGS:
            sub = sub.iloc[:SESSION_MAX_MSGS].copy()
        sessions.append(sub)
    return sessions


@dataclass
class Cluster:
    centroid: np.ndarray  # float32 normalized
    items: List[int]      # indices in session list


def cluster_session(vecs: np.ndarray, threshold: float) -> List[Cluster]:
    """
    Online clustering: vecs уже нормализованы.
    """
    clusters: List[Cluster] = []
    for i in range(vecs.shape[0]):
        v = vecs[i]
        best_j = -1
        best_sim = -1.0

        for j, c in enumerate(clusters):
            sim = float(np.dot(v, c.centroid))
            if sim > best_sim:
                best_sim = sim
                best_j = j

        if best_j >= 0 and best_sim >= threshold:
            c = clusters[best_j]
            c.items.append(i)
            # обновим центроид (среднее) + нормализуем
            new_cent = (c.centroid * (len(c.items) - 1) + v) / float(len(c.items))
            new_cent = new_cent / (np.linalg.norm(new_cent) + 1e-12)
            c.centroid = new_cent.astype(np.float32)
        else:
            clusters.append(Cluster(centroid=v.astype(np.float32), items=[i]))

    return clusters


def make_doc_text(session_df: pd.DataFrame, item_idxs: List[int]) -> str:
    """
    Склейка сообщений кластера в документ.
    """
    item_idxs = item_idxs[:DOC_MAX_MSGS]
    sub = session_df.iloc[item_idxs].copy()

    lines = []
    for _, r in sub.iterrows():
        # время оставим в UTC-строке
        ts = pd.to_datetime(r["msg_date"], utc=True).strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{ts}] {r['text']}")
    return "\n".join(lines).strip()


def main():
    device = _device_pick()
    log.info("device=%s torch=%s cuda=%s", device, torch.__version__, torch.version.cuda)

    engine = get_engine()

    # Дебаг: где мы вообще подключились
    with engine.connect() as c:
        row = c.execute(text("""
            SELECT current_database(), current_user,
                   current_setting('search_path'),
                   (SELECT count(*) FROM public.feedback_raw_v3)
        """)).fetchone()
    log.info("[DBG] db=%s user=%s search_path=%s feedback_raw_v3_count=%s", row[0], row[1], row[2], row[3])

    df = load_free_chat_msgs(engine)
    if df.empty:
        raise SystemExit("Нет данных для микротредов (kind=chat..., parent_msg_id IS NULL). Проверь MICRO_CHAT_KINDS / данные.")

    log.info("rows_free_chat_msgs=%d chats=%d", len(df), df["chat_id"].nunique())

    log.info("load embedder: %s (device=%s)", EMBED_MODEL, device)
    model = SentenceTransformer(EMBED_MODEL, device=device)

    docs: List[Dict[str, Any]] = []

    # Идём по чатам
    for chat_id, df_chat in df.groupby("chat_id", sort=False):
        df_chat = df_chat.sort_values(["msg_date", "msg_id"]).reset_index(drop=True)
        sessions = split_sessions(df_chat, SESSION_GAP_MIN)

        for s_idx, sess in enumerate(sessions):
            if len(sess) < MIN_CLUSTER_SIZE:
                continue

            texts = sess["text"].astype(str).tolist()
            # E5: documents = passage:
            inp = [f"passage: {t}" for t in texts]

            vecs = model.encode(
                inp,
                batch_size=EMB_BATCH,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

            clusters = cluster_session(vecs, MICRO_THRESHOLD)

            # сохраним только содержательные
            for c_idx, cl in enumerate(clusters):
                if len(cl.items) < MIN_CLUSTER_SIZE:
                    continue

                start_dt = sess.iloc[min(cl.items)]["msg_date"]
                last_dt = sess.iloc[max(cl.items)]["msg_date"]

                doc_text = make_doc_text(sess, cl.items)
                if not doc_text:
                    continue

                # thread_id (строковый)
                # session_key: unix начального времени сессии (чтобы было стабильнее)
                sess_start_unix = int(pd.to_datetime(sess.iloc[0]["msg_date"], utc=True).timestamp())
                thread_id = f"micro:{int(chat_id)}:{sess_start_unix}:{s_idx}:{c_idx}"

                # root_text как "заголовок" (первое сообщение)
                root_text = str(sess.iloc[cl.items[0]]["text"])[:400]

                # пермалинк (первый)
                root_permalink = sess.iloc[cl.items[0]].get("permalink", None)

                docs.append({
                    "thread_id": thread_id,
                    "doc_kind": "micro",
                    "source_username": sess.iloc[0].get("source_username", None),
                    "source_title": sess.iloc[0].get("source_title", None),
                    "chat_id": int(chat_id),
                    "root_msg_id": int(sess.iloc[cl.items[0]]["msg_id"]),
                    "start_dt": pd.to_datetime(start_dt, utc=True),
                    "last_dt": pd.to_datetime(last_dt, utc=True),
                    "root_text": root_text,
                    "body_text": doc_text,
                    "n_items": int(len(cl.items)),
                    "root_permalink": root_permalink,
                })

        if len(docs) and (len(docs) % 2000 == 0):
            log.info("progress docs=%d (last chat_id=%s)", len(docs), chat_id)

    if not docs:
        raise SystemExit("Микротреды не получились (все кластеры оказались слишком маленькими). Попробуй MICRO_MIN_CLUSTER_SIZE=2 или порог ниже.")

    df_docs = pd.DataFrame(docs)
    log.info("micro_docs=%d", len(df_docs))

    # ---- эмбеддим сами документы (root+body) ----
    texts_docs = (df_docs["root_text"].fillna("") + "\n\n" + df_docs["body_text"].fillna("")).map(_clean_text).tolist()
    inp = [f"passage: {t}" for t in texts_docs]

    t0 = time.time()
    vecs = model.encode(
        inp,
        batch_size=EMB_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    vecs = np.ascontiguousarray(vecs)
    log.info("embeddings shape=%s dt=%.2fs", vecs.shape, time.time() - t0)

    dim = int(vecs.shape[1])
    faiss_ids = np.arange(len(df_docs), dtype=np.int64)

    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(vecs, faiss_ids)

    # ---- save ----
    idx_path  = os.path.join(FAISS_DIR, "index.faiss")
    meta_path = os.path.join(FAISS_DIR, "meta.json")
    map_path  = os.path.join(FAISS_DIR, "mapping.csv")
    npy_path  = os.path.join(FAISS_DIR, "doc_id_map.npy")

    faiss.write_index(index, idx_path)

    meta = {
        "type": "microthreads",
        "model": EMBED_MODEL,
        "dim": dim,
        "rows": int(vecs.shape[0]),
        "built_on_unix": time.time(),
        "micro_threshold": MICRO_THRESHOLD,
        "session_gap_min": SESSION_GAP_MIN,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "chat_kinds": [k.strip() for k in CHAT_KINDS if k.strip()],
        "id_scheme": "faiss_id:int64 -> mapping.csv/doc_id_map.npy -> thread_id:str",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    df_out = df_docs.copy()
    df_out.insert(0, "faiss_id", faiss_ids)

    # нормализуем даты для CSV
    df_out["start_dt"] = pd.to_datetime(df_out["start_dt"], utc=True).astype(str)
    df_out["last_dt"] = pd.to_datetime(df_out["last_dt"], utc=True).astype(str)

    df_out.to_csv(map_path, index=False, encoding="utf-8-sig", sep=";")

    np.save(npy_path, df_docs["thread_id"].astype(str).to_numpy())

    log.info("saved: %s | %s | %s | %s", idx_path, meta_path, map_path, npy_path)
    log.info("DONE")


if __name__ == "__main__":
    main()
