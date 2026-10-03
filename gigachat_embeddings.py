# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import uuid
import logging
from typing import List, Optional, Union

import numpy as np
import requests


log = logging.getLogger("gigachat-emb")


_GIGA_TOKEN_CACHE = {
    "token": None,
    "expires_at": 0,
}


def _verify_arg() -> Union[bool, str]:
    """
    Для requests.verify:
    - True / False
    - либо путь к CA bundle
    """
    ca_file = (os.getenv("GIGACHAT_CA_BUNDLE_FILE") or "").strip()
    if ca_file:
        return ca_file

    raw = (os.getenv("GIGACHAT_VERIFY_SSL", "") or os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "1")).strip().lower()
    return raw not in {"0", "false", "no"}


def _auth_key() -> str:
    val = (
        os.getenv("GIGACHAT_AUTH_KEY")
        or os.getenv("GIGACHAT_CREDENTIALS")
        or ""
    ).strip()
    return val


def _scope() -> str:
    return (os.getenv("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS").strip()


def _timeout() -> int:
    return int(
        os.getenv("GIGACHAT_TIMEOUT_SEC")
        or os.getenv("GIGACHAT_TIMEOUT")
        or "180"
    )


def _retries() -> int:
    return int(os.getenv("GIGACHAT_RETRIES", "2"))


def _embed_model() -> str:
    # Рекомендую начать с Embeddings-2
    return (os.getenv("GIGACHAT_EMBED_MODEL") or "Embeddings-2").strip()


def _batch_size() -> int:
    return int(os.getenv("GIGACHAT_EMBED_BATCH", "1"))


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return v / norms


def get_access_token(force_refresh: bool = False) -> str:
    now_ts = int(time.time())

    if not force_refresh:
        token = _GIGA_TOKEN_CACHE.get("token")
        exp = int(_GIGA_TOKEN_CACHE.get("expires_at") or 0)
        if token and exp - 60 > now_ts:
            return str(token)

    auth_value = _auth_key()
    if not auth_value:
        raise RuntimeError("Не задан GIGACHAT_AUTH_KEY / GIGACHAT_CREDENTIALS")

    if not auth_value.lower().startswith("basic "):
        auth_value = f"Basic {auth_value}"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "RqUID": str(uuid.uuid4()),
        "Authorization": auth_value,
    }

    data = {
        "scope": _scope(),
    }

    r = requests.post(
        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        headers=headers,
        data=data,
        timeout=_timeout(),
        verify=_verify_arg(),
    )
    r.raise_for_status()

    payload = r.json()
    token = (payload.get("access_token") or "").strip()
    expires_at = int(payload.get("expires_at") or 0)

    if not token:
        raise RuntimeError("GigaChat не вернул access_token")

    _GIGA_TOKEN_CACHE["token"] = token
    _GIGA_TOKEN_CACHE["expires_at"] = expires_at
    return token


def embed_texts(
    texts: List[str],
    *,
    normalize: bool = True,
    model: Optional[str] = None,
) -> np.ndarray:
    """
    Возвращает np.ndarray shape=(N, dim), dtype=float32
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    use_model = (model or _embed_model()).strip()
    batch_size = _batch_size()

    cleaned = [str(t or "").strip() for t in texts]
    cleaned = [t if t else " " for t in cleaned]

    chunks = [cleaned[i:i + batch_size] for i in range(0, len(cleaned), batch_size)]
    vectors: List[np.ndarray] = []

    last_err = None
    for chunk in chunks:
        for attempt in range(_retries() + 1):
            try:
                token = get_access_token(force_refresh=(attempt > 0))

                headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                }

                payload = {
                    "model": use_model,
                    "input": chunk,
                }

                r = requests.post(
                    "https://gigachat.devices.sberbank.ru/api/v1/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=_timeout(),
                    verify=_verify_arg(),
                )
                r.raise_for_status()

                data = r.json()
                rows = data.get("data") or []
                if not rows:
                    raise RuntimeError(f"Пустой ответ embeddings: {data}")

                # Сортируем по index, чтобы сохранить исходный порядок входов
                rows = sorted(rows, key=lambda x: int(x.get("index", 0)))
                arr = np.array([row["embedding"] for row in rows], dtype=np.float32)

                if normalize:
                    arr = _normalize(arr)

                vectors.append(arr)
                break

            except Exception as e:
                last_err = e
                log.warning("[GIGA_EMB] try %s fail: %s", attempt + 1, e)
                _GIGA_TOKEN_CACHE["token"] = None
                _GIGA_TOKEN_CACHE["expires_at"] = 0
                if attempt < _retries():
                    time.sleep(1.5 * (attempt + 1))
                else:
                    raise

    if not vectors:
        raise RuntimeError(f"Не удалось получить embeddings: {last_err}")

    return np.vstack(vectors).astype(np.float32)