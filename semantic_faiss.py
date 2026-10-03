# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
from gigachat_embeddings import embed_texts as giga_embed_texts

import numpy as np
import pandas as pd

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").strip().lower()

# ленивые импорты (ускоряет старт, если семантика выключена)
_faiss = None
_st = None
_torch = None

_MODEL_CACHE = {"key": None, "model": None}


def _lazy_imports():
    global _faiss, _st, _torch
    if _faiss is None:
        import faiss  # type: ignore
        _faiss = faiss
    if _st is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _st = SentenceTransformer
    if _torch is None:
        import torch  # type: ignore
        _torch = torch


def _device_pick(want: str) -> str:
    _lazy_imports()
    want = (want or "cpu").lower()
    try:
        if want == "cuda" and _torch.cuda.is_available():
            return "cuda"
        if want == "mps" and getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def get_model(model_name: str, device: str):
    _lazy_imports()
    device = _device_pick(device)
    key = f"{model_name}@@{device}"
    if _MODEL_CACHE["key"] == key and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"]
    m = _st(model_name, device=device)
    _MODEL_CACHE["key"] = key
    _MODEL_CACHE["model"] = m
    return m


@dataclass
class FaissStore:
    name: str
    dir_path: str
    index: object
    mapping: pd.DataFrame
    meta: Dict
    doc_id_map: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.meta.get("dim") or 0)


def load_store(dir_path: str, name: str) -> Optional[FaissStore]:
    dirp = Path(dir_path)
    idx_path = dirp / "index.faiss"
    map_path = dirp / "mapping.csv"
    meta_path = dirp / "meta.json"
    npy_path = dirp / "doc_id_map.npy"

    if not idx_path.exists() or not map_path.exists() or not meta_path.exists() or not npy_path.exists():
        return None

    _lazy_imports()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    index = _faiss.read_index(str(idx_path))

    # mapping.csv у тебя с sep=";"
    mapping = pd.read_csv(map_path, sep=";", encoding="utf-8-sig")
    if "faiss_id" in mapping.columns:
        mapping = mapping.sort_values("faiss_id").reset_index(drop=True)
    else:
        mapping.insert(0, "faiss_id", np.arange(len(mapping), dtype=np.int64))

    doc_id_map = np.load(str(npy_path), allow_pickle=True)

    return FaissStore(
        name=name,
        dir_path=str(dirp),
        index=index,
        mapping=mapping,
        meta=meta,
        doc_id_map=doc_id_map,
    )


def _pick_text(mapping: pd.DataFrame) -> pd.Series:
    # универсально для threads и micro
    for c in ("doc_text", "body_text", "replies_text", "comments_text", "cluster_text", "text"):
        if c in mapping.columns:
            return mapping[c].fillna("").astype(str)
    return pd.Series([""] * len(mapping), index=mapping.index)


def search(
    store: FaissStore,
    query_text: str,
    model_name: str,
    device: str,
    top_k: int = 80,
) -> pd.DataFrame:
    if store is None:
        return pd.DataFrame()

    q = (query_text or "").strip()
    if not q:
        return pd.DataFrame()

    if EMBED_PROVIDER == "gigachat":
        qvec = giga_embed_texts([q], normalize=True).astype(np.float32)
    else:
        model = get_model(model_name, device)
        qvec = model.encode(
            [f"query: {q}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    scores, ids = store.index.search(qvec, int(top_k))
    ids = ids.reshape(-1)
    scores = scores.reshape(-1)

    rows = []
    for faiss_id, sc in zip(ids, scores):
        if faiss_id < 0:
            continue
        faiss_id = int(faiss_id)
        if faiss_id >= len(store.mapping):
            continue
        r = store.mapping.iloc[faiss_id].to_dict()
        r["score"] = float(sc)
        # doc_id (строка) — как “унифицированный ключ”
        try:
            r["doc_id"] = str(store.doc_id_map[faiss_id])
        except Exception:
            r["doc_id"] = str(r.get("thread_id") or r.get("micro_id") or faiss_id)
        rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # нормализуем текстовую колонку
    if "doc_text" not in df.columns:
        df["doc_text"] = _pick_text(df)

    return df
