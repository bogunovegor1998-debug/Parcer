# -*- coding: utf-8 -*-
"""
summary_llm.py — человеко-понятные сводки через теги (tag → group)

Пайплайн:
1) (опционально) semantic top-N по запросу
2) LLM: теги на каждый отзыв (1–3)
3) частотные теги
4) LLM: для каждого тега 1 строка "тег: пояснение"
5) пост-фильтрация + анти-галлюцинации
fallback: rule-based агрегатор

Публичная функция:
  summarize_topn(df, query_text, text_col="...", datetime_col=None, top_n=60, skip_sem=None) -> str
"""

from __future__ import annotations

import os
import re
import logging
import warnings
import hashlib
import time
from typing import List, Dict, Iterable, Optional
from gigachat_embeddings import embed_texts as giga_embed_texts

import numpy as np
import pandas as pd

# ===================== ЛОГИ =====================

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("summary_llm")
    if getattr(logger, "_inited", False):
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "summary.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    for noisy in ("httpx", "requests", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logger._inited = True
    return logger

log = _setup_logger()

# ===================== КОНФИГ =====================

BULLET = "🔹"

# LLM / Ollama
USE_OLLAMA = os.getenv("SUMMARY_USE_OLLAMA", os.getenv("USE_OLLAMA", "1")).lower() not in {"0", "false", "no"}
OLLAMA_HOST = os.getenv("SUMMARY_OLLAMA_HOST", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
OLLAMA_MODEL = os.getenv("SUMMARY_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"))

OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SEC", "180"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))

# Embeddings
EMBED_MODEL      = os.getenv("EMBEDDER", "intfloat/multilingual-e5-base")
EMBED_DEVICE_ENV = (os.getenv("EMBED_DEVICE") or "").strip().lower()  # cuda|cpu|mps|""
EMB_BATCH_SIZE   = int(os.getenv("EMB_BATCH_SIZE", "128"))
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").strip().lower()

# LLM generation
LLM_TEMP    = float(os.getenv("LLM_TEMPERATURE", "0.15"))
LLM_TOP_P   = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_CTX     = int(os.getenv("LLM_CTX", "8192"))
LLM_MAX_TOK = int(os.getenv("LLM_MAX_TOKENS", "700"))

# summary
SUMMARY_TOP_N        = int(os.getenv("SUMMARY_TOP_N", "60"))
SUMMARY_MIN_BULLETS  = int(os.getenv("SUMMARY_MIN_BULLETS", "4"))
SUMMARY_MAX_BULLETS  = int(os.getenv("SUMMARY_MAX_BULLETS", "8"))
MAX_COMMENT_CHARS    = int(os.getenv("MAX_COMMENT_CHARS", "900"))

SUMMARY_SKIP_SEM = os.getenv("SUMMARY_SKIP_SEM", "0").lower() in {"1", "true", "yes"}
SUMMARY_POOL_MULT = int(os.getenv("SUMMARY_POOL_MULT", "6"))

TAG_BATCH_SIZE       = int(os.getenv("TAG_BATCH_SIZE", "12"))
TAGS_PER_COMMENT_MAX = int(os.getenv("TAGS_PER_COMMENT_MAX", "3"))
QA_MAX_CONTEXT_ITEMS = int(os.getenv("QA_MAX_CONTEXT_ITEMS", "10"))
QA_MAX_CONTEXT_CHARS = int(os.getenv("QA_MAX_CONTEXT_CHARS", "1500"))
QA_MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "900"))
QA_MAX_BULLETS = int(os.getenv("QA_MAX_BULLETS", "5"))

# ===================== ТЕКСТ-УТИЛЫ =====================

NON_DESIRED_RE = re.compile(
    "[\u0000-\u001F\u007F-\u009F\u2000-\u200F\u2028\u2029\u202A-\u202E\u2066-\u2069\uFE00-\uFE0F]+"
)
MD_TRASH_RE  = re.compile(r"(\*\*|__|\*|_|~~|`+)")
TOKEN_RE     = re.compile(r"[a-zа-я0-9\-]+", re.I)
RU_LETTERS_RE = re.compile(r"[А-Яа-яЁё]")
LETTERS_RE    = re.compile(r"[A-Za-zА-Яа-яЁё]")
OUTPUT_ALLOWED_RE = re.compile(r"[^0-9А-Яа-яЁё \n\.\,\:\;\!\?\(\)\-\—\–…]")

def _strip_noise(s: str) -> str:
    if not s:
        return ""
    s = NON_DESIRED_RE.sub("", str(s))
    s = MD_TRASH_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_abbreviations(s: str) -> str:
    if not s:
        return s
    s = re.sub(r"\bWi\s*:\s*Fi\b", "Wi-Fi", s, flags=re.I)
    s = re.sub(r"\bQR\s*:\s*код(ами|ов|ы|у|ом)?\b", r"QR-код\1", s, flags=re.I)
    s = re.sub(r"\bQR\s*:\s*кодов\b", "QR-кодов", s, flags=re.I)
    s = re.sub(r"\bQR\s*:\s*", "QR-", s, flags=re.I)
    s = re.sub(r"\bМЦД\s*:\s*([1-4])\b", r"МЦД-\1", s, flags=re.I)
    s = re.sub(r"\bD\s*:\s*([1-4])\b", r"D\1", s, flags=re.I)
    s = re.sub(r"\bМЦД\s*[–—-]\s*([1-4])\b", r"МЦД-\1", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()

def _lower(s: str) -> str:
    return _strip_noise((s or "")).lower().replace("ё", "е").strip()

def _only_ru_digits_punct(s: str) -> str:
    s = OUTPUT_ALLOWED_RE.sub("", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _clean_for_summary(t: str) -> str:
    t = _strip_noise(_normalize_abbreviations(t))
    return t[:MAX_COMMENT_CHARS].strip()

def _ru_ratio(s: str) -> float:
    letters = LETTERS_RE.findall(s or "")
    if not letters:
        return 1.0
    ru = RU_LETTERS_RE.findall(s or "")
    return len(ru) / len(letters) if letters else 1.0


# ===================== ЭМБЕДДИНГИ / TOP-N =====================

_embedder = None
_emb_cache: Dict[str, np.ndarray] = {}

def _pick_device() -> str:
    want = EMBED_DEVICE_ENV
    try:
        import torch
        if want in {"cuda", "gpu"} and torch.cuda.is_available():
            return "cuda"
        if want == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        dev = _pick_device()
        log.info("[EMB] load %s (device=%s)", EMBED_MODEL, dev)
        try:
            _embedder = SentenceTransformer(EMBED_MODEL, device=dev)
        except Exception as e:
            log.warning("[EMB] fallback to CPU (%s)", e)
            _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _embedder

def _hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _embed_texts(texts: List[str], is_query: bool) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if EMBED_PROVIDER == "gigachat":
        # для GigaChat не используем префиксы query:/passage:
        # просто даем чистый текст
        cleaned = [str(t or "").strip() for t in texts]
        return giga_embed_texts(cleaned, normalize=True)

    model = _get_embedder()

    if is_query:
        inp = [f"query: {t}" for t in texts]
        vec = model.encode(
            inp,
            batch_size=EMB_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return vec.astype(np.float32)

    inp = [f"passage: {t}" for t in texts]
    out: List[Optional[np.ndarray]] = []
    miss: List[int] = []

    for i, t in enumerate(inp):
        h = _hash_text(t)
        if h in _emb_cache:
            out.append(_emb_cache[h])
        else:
            out.append(None)
            miss.append(i)

    if miss:
        new = model.encode(
            [inp[i] for i in miss],
            batch_size=EMB_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for j, i in enumerate(miss):
            _emb_cache[_hash_text(inp[i])] = new[j]
            out[i] = new[j]

    return np.vstack(out).astype(np.float32)

def _mmr_diverse_top(sim: np.ndarray, dv: np.ndarray, k: int, dup_thr: float = 0.88) -> List[int]:
    order = np.argsort(-sim)
    selected: List[int] = []
    for idx in order:
        if all(float(dv[idx] @ dv[j]) < dup_thr for j in selected):
            selected.append(int(idx))
            if len(selected) >= k:
                break
    if not selected:
        selected = [int(x) for x in order[:k]]
    return selected

def _dedup_by_text(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    norm = (
        df[text_col].astype(str)
        .str.lower()
        .str.replace("ё", "е")
        .str.replace(r"[^\w\s]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    tmp = df.assign(_k=norm)
    tmp = tmp.sort_index(kind="mergesort").drop_duplicates("_k", keep="first")
    return tmp.drop(columns=["_k"])

def _semantic_select_top(
    df: pd.DataFrame,
    q: str,
    text_col: str,
    n: int,
    datetime_col: Optional[str] = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    base = df.copy()

    if datetime_col and datetime_col in base.columns:
        try:
            base = base.sort_values(datetime_col, ascending=False, na_position="last")
        except Exception:
            pass

    base = _dedup_by_text(base, text_col)

    if SUMMARY_SKIP_SEM or len(base) <= n:
        return base.head(n)

    pool_n = min(len(base), max(n, n * SUMMARY_POOL_MULT))
    pool = base.head(pool_n).copy()

    texts_raw = pool[text_col].astype(str).tolist()
    texts = [_clean_for_summary(t) for t in texts_raw]
    idx = [i for i, t in enumerate(texts) if t]
    if not idx:
        return pool.head(n)

    texts = [texts[i] for i in idx]
    sub = pool.iloc[idx].copy()

    qv = _embed_texts([_clean_for_summary(q)], is_query=True)
    dv = _embed_texts(texts, is_query=False)
    if dv.shape[0] == 0:
        return sub.head(n)

    sims = dv @ qv[0]

    k_margin = min(n * 2, len(sub))
    sel_idx = _mmr_diverse_top(sims, dv, k_margin, dup_thr=0.88)
    sel = sub.iloc[sel_idx].copy()
    sel = _dedup_by_text(sel, text_col)

    if len(sel) > n:
        ii = np.linspace(0, len(sel) - 1, num=n, dtype=int)
        sel = sel.iloc[ii]

    log.info("[SEM] selected=%d from pool=%d", len(sel), len(sub))
    return sel


# ===================== LLM BACKEND =====================

import uuid

SUMMARY_PROVIDER = os.getenv("SUMMARY_PROVIDER", "gigachat").strip().lower()

GIGACHAT_AUTH_KEY = (os.getenv("GIGACHAT_AUTH_KEY") or os.getenv("GIGACHAT_CREDENTIALS") or "").strip()
GIGACHAT_SCOPE = (os.getenv("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS").strip()
GIGACHAT_MODEL = (os.getenv("GIGACHAT_MODEL") or "GigaChat-2-Pro").strip()
GIGACHAT_TIMEOUT = int(os.getenv("GIGACHAT_TIMEOUT_SEC", "180"))
GIGACHAT_RETRIES = int(os.getenv("GIGACHAT_RETRIES", "2"))
def _gigachat_verify_arg():
    ca_file = (os.getenv("GIGACHAT_CA_BUNDLE_FILE") or "").strip()
    if ca_file:
        return ca_file

    raw = (
        os.getenv("GIGACHAT_VERIFY_SSL", "")
        or os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "1")
    ).strip().lower()

    return raw not in {"0", "false", "no"}

_GIGA_TOKEN_CACHE = {
    "token": None,
    "expires_at": 0,
}


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    return text.strip()


def _ollama_generate(prompt: str, max_tokens: int) -> str:
    import requests

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": LLM_CTX,
            "num_predict": max_tokens,
            "temperature": LLM_TEMP,
            "top_p": LLM_TOP_P,
            "repeat_penalty": 1.1,
            "num_batch": 256,
        },
    }

    last_err = None
    for attempt in range(OLLAMA_RETRIES + 1):
        try:
            r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return _strip_think((data.get("response") or "").strip())
        except Exception as e:
            last_err = e
            log.warning("[LLM:ollama] try %d/%d fail: %s", attempt + 1, OLLAMA_RETRIES + 1, e)
            if attempt < OLLAMA_RETRIES:
                time.sleep(1.2 * (attempt + 1))

    log.error("[LLM:ollama] final fail: %s", last_err)
    return ""


def _gigachat_get_token() -> str:
    import requests

    now_ts = int(time.time())
    cached_token = _GIGA_TOKEN_CACHE.get("token")
    cached_exp = int(_GIGA_TOKEN_CACHE.get("expires_at") or 0)

    if cached_token and cached_exp - 60 > now_ts:
        return cached_token

    if not GIGACHAT_AUTH_KEY:
        log.error("[LLM:gigachat] GIGACHAT_AUTH_KEY not set")
        return ""

    auth_value = GIGACHAT_AUTH_KEY
    if not auth_value.lower().startswith("basic "):
        auth_value = f"Basic {auth_value}"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "RqUID": str(uuid.uuid4()),
        "Authorization": auth_value,
    }

    data = {
        "scope": GIGACHAT_SCOPE,
    }

    r = requests.post(
        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        headers=headers,
        data=data,
        timeout=GIGACHAT_TIMEOUT,
        verify=_gigachat_verify_arg(),
    )
    r.raise_for_status()

    payload = r.json()
    token = (payload.get("access_token") or "").strip()
    expires_at = int(payload.get("expires_at") or 0)

    if not token:
        raise RuntimeError("GigaChat access_token is empty")

    _GIGA_TOKEN_CACHE["token"] = token
    _GIGA_TOKEN_CACHE["expires_at"] = expires_at
    return token


def _gigachat_generate(prompt: str, max_tokens: int) -> str:
    import requests

    last_err = None

    for attempt in range(GIGACHAT_RETRIES + 1):
        try:
            token = _gigachat_get_token()
            if not token:
                return ""

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            }

            payload = {
                "model": GIGACHAT_MODEL,
                "temperature": LLM_TEMP,
                "top_p": LLM_TOP_P,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            }

            r = requests.post(
                "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=GIGACHAT_TIMEOUT,
                verify=_gigachat_verify_arg(),
            )
            r.raise_for_status()

            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                return ""

            msg = choices[0].get("message") or {}
            content = (msg.get("content") or "").strip()
            return _strip_think(content)

        except Exception as e:
            last_err = e
            log.warning("[LLM:gigachat] try %d/%d fail: %s", attempt + 1, GIGACHAT_RETRIES + 1, e)

            # если токен протух / сбой авторизации — сбрасываем кеш
            _GIGA_TOKEN_CACHE["token"] = None
            _GIGA_TOKEN_CACHE["expires_at"] = 0

            if attempt < GIGACHAT_RETRIES:
                time.sleep(1.5 * (attempt + 1))

    log.error("[LLM:gigachat] final fail: %s", last_err)
    return ""


def _llm_generate(prompt: str, max_tokens: int) -> str:
    provider = SUMMARY_PROVIDER

    if provider == "ollama":
        return _ollama_generate(prompt, max_tokens)

    if provider == "gigachat":
        return _gigachat_generate(prompt, max_tokens)

    log.warning("[LLM] unknown provider=%s -> empty response", provider)
    return ""


# ===================== ПОСТ-ОБРАБОТКА LLM-ТЕКСТА =====================

def _strip_numbering(s: str) -> str:
    s = re.sub(r"^\s*(?:\d+[\).\:]|\d+\s*-\s*|\(\d+\))\s*", "", s)
    return s.strip()

def _sentence_limit(text: str, max_sent: int = 2) -> str:
    parts = re.split(r"(?<=[\.!\?])\s+", text)
    return text if len(parts) <= max_sent else " ".join(parts[:max_sent]).strip()

META_TALK_RE = re.compile(
    r"\b("
    r"здравствуйте|добрый\s+день|день\s+добрый|добрый\s+вечер|доброе\s+утро|привет|уважаем"
    r"|как\s+модель|формат|вывод|итог|начну\s+с|во-первых|во\s+вторых|пользоват"
    r")\b",
    re.I,
)
BAD_META_PAT = re.compile(r"(?:^|\s)(?:bullet|черновик|черновые|см\. выше|см\. ниже)", re.I)

def _post_format(lines: Iterable[str], max_n: int) -> List[str]:
    out: List[str] = []
    for raw in lines:
        ln = (raw or "").strip()
        if not ln:
            continue
        ln = re.sub(r'^(?:[🔹•\-\–\—\*]\s*)+', '', ln)
        ln = _strip_numbering(ln)
        ln = re.sub(r"\s*[:—-]\s*", ": ", ln, count=1)
        if ":" not in ln:
            continue
        title, body = ln.split(":", 1)
        title = _only_ru_digits_punct(_normalize_abbreviations(_strip_noise(title)))
        body = _only_ru_digits_punct(_normalize_abbreviations(_strip_noise(body)))
        if not title or not body:
            continue
        if META_TALK_RE.search(ln) or BAD_META_PAT.search(ln):
            continue
        if _ru_ratio(title) < 0.9 or _ru_ratio(body) < 0.85:
            continue
        tl = [w for w in title.split() if len(w) >= 2]
        if len(tl) == 0:
            continue
        if len(tl) > 4:
            title = " ".join(tl[:4])
        body = _sentence_limit(body, 2)
        if len(body) > 300:
            body = body[:298] + "…"
        if len(title) > 80:
            title = title[:78] + "…"
        out.append(f"{BULLET} {title}: {body}")

    # дедуп
    seen = set()
    normed: List[str] = []

    def _norm(x: str) -> str:
        x = _lower(x)
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"[^\w\s:,-]+", "", x)
        return x.strip()

    for b in out:
        k = _norm(b)
        if k and k not in seen:
            seen.add(k)
            normed.append(b)

    return normed[:max_n]

def _token_bag(texts: List[str]) -> set:
    bag = set()
    for t in texts:
        for tok in TOKEN_RE.findall(_lower(t)):
            if len(tok) >= 4:
                bag.add(tok)
    return bag

def _anti_hallucination_filter(bullets: List[str], source_texts: List[str], min_hits: int = 2) -> List[str]:
    if not bullets:
        return bullets
    bag = _token_bag(source_texts)

    def _filter_with(h: int) -> List[str]:
        ok = []
        for b in bullets:
            core = _lower(re.sub(r"^[🔹\s]+", "", b))
            parts = core.split(":", 1)
            tail = parts[1] if len(parts) == 2 else core
            toks = [t for t in TOKEN_RE.findall(tail) if len(t) >= 4]
            hits = sum(1 for t in toks if t in bag)
            if hits >= h:
                ok.append(b)
        return ok

    out = _filter_with(min_hits)
    if len(out) < max(3, SUMMARY_MIN_BULLETS // 2):
        out = _filter_with(1)
    return out[:SUMMARY_MAX_BULLETS]


# ===================== FALLBACK АГРЕГАТОР =====================

CATALOG = [
    ("QR/СБП не проходит",
     re.compile(r"\b(qr|сбп|код|сканир|считыв)\w*\b", re.I),
     "Жалобы на проблемы со считыванием QR-кодов и оплатой."),
    ("Двойные списания",
     re.compile(r"\b(двойн|повторн)\w*\s+списан|\bсписал\w*\s*(?:дважд|2\s*раз)", re.I),
     "Жалобы на повторные или двойные списания."),
    ("Пополнение не зачислено",
     re.compile(r"\b(попол|зачисл|не\s*прошл|не\s*пришл)\w*\b", re.I),
     "Жалобы на незачисление пополнения."),
    ("Блокировка карты",
     re.compile(r"\b(блокир|стоп[\-\s]?лист|заблок)\w*\b", re.I),
     "Жалобы на блокировку карты или попадание в стоп-лист."),
    ("Турникет не открыл",
     re.compile(r"\b(турник|створк|валидатор|не\s*открыл|не\s*пропуст)\w*\b", re.I),
     "Жалобы на проблемы с проходом через турникеты и валидацией."),
    ("Температура/кондиционер",
     re.compile(r"\b(температур|жарк|духот|кондиционер|конде[йи]|холод|мороз)\w*\b", re.I),
     "Жалобы на температуру, духоту, холод или работу кондиционеров."),
    ("Задержки движения",
     re.compile(r"\b(задержк|опоздан|стойм|простоя|сбой|полом)\w*\b", re.I),
     "Жалобы на задержки, сбои движения и простои."),
    ("Переполненность",
     re.compile(r"\b(толп|давк|переполн|тесн)\w*\b", re.I),
     "Жалобы на переполненность вагонов, платформ и вестибюлей."),
]

def _aggregate_humanish(df: pd.DataFrame, text_col: str, q: str, want_n: int = 12) -> List[str]:
    texts = df[text_col].astype(str).tolist()
    counts = {name: 0 for name, _, _ in CATALOG}
    for t in texts:
        tl = _lower(t)
        for name, pat, _ in CATALOG:
            if pat.search(tl):
                counts[name] += 1
    items = []
    for name, _, expl in CATALOG:
        n = counts[name]
        if n <= 0:
            continue
        items.append((n, f"{BULLET} {name}: {expl}"))
    items.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in items[:want_n]]


# ===================== ТЕГИ =====================



def _normalize_tag(tag: str) -> str:
    s = (tag or "").strip().lower()

    # Убираем мусорные кавычки/скобки
    s = re.sub(r"[\"«»\(\)\[\]]+", "", s)

    # Приводим разделители к пробелам, а не к двоеточиям
    s = s.replace("_", " ").replace("-", " ")

    # Схлопываем пробелы
    s = re.sub(r"\s+", " ", s)

    # Убираем хвостовую пунктуацию
    s = re.sub(r"[.,;:!?]+$", "", s).strip()

    bad_exact = {
        "",
        "нет жалоб",
        "нет жалоб ",
        "нет_жалоб",
        "нет-жалоб",
        "жалоб нет",
        "без жалоб",
    }
    if s in bad_exact:
        return ""

    if len(s) <= 2:
        return ""

    return s

def _tagging_prompt(query_text: str, comments: List[str]) -> str:
    header = (
        "Пользователь спрашивает:\n\n"
        f"«{query_text}»\n\n"
        "Ниже перечислены отзывы пассажиров. Для каждого отзыва придумай 1–3 коротких ТЕГА,\n"
        "которые описывают проблему или тему жалобы.\n\n"
        "Правила:\n"
        "• Пиши теги по-русски, 1–3 слова, без точки на конце.\n"
        "• Не пиши даты, номера поездов и т.п. — только смысл проблемы.\n"
        "• Если в отзыве нет явной жалобы, напиши: нет жалоб.\n"
        "• Ничего не выдумывай сверх текста.\n"
        "• НЕ пиши пояснения, только теги.\n\n"
        "Формат ответа строго такой (без лишних строк):\n"
        "1) тег1; тег2\n"
        "2) тег1\n"
        "3) тег1; тег2; тег3\n"
        "...\n\n"
        "Отзывы:\n"
    )
    lines = []
    for i, c in enumerate(comments, start=1):
        lines.append(f"{i}) {c}")
    return header + "\n".join(lines) + "\n\nОтвет:\n"

def _parse_tagging_output(raw: str, base_idx: int, tags_per_comment: List[List[str]]):
    if not raw:
        return
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in lines:
        ln_ = re.sub(r'^(?:[🔹•\-\–\—\*]\s*)+', '', ln)
        m = re.match(r"^\s*(\d+)[\).\:\-]\s*(.+)$", ln_)
        if not m:
            continue
        idx_local = int(m.group(1)) - 1
        idx_global = base_idx + idx_local
        if idx_global < 0 or idx_global >= len(tags_per_comment):
            continue
        rest = m.group(2)
        parts = re.split(r"[;,]", rest)
        for p in parts:
            t = _normalize_tag(p)
            if not t:
                continue
            if len(tags_per_comment[idx_global]) >= TAGS_PER_COMMENT_MAX:
                break
            if t not in tags_per_comment[idx_global]:
                tags_per_comment[idx_global].append(t)

def _tag_summary_prompt(tag: str, examples: List[str]) -> str:
    header = (
        f"Тег проблемы: «{tag}».\n\n"
        "Ниже несколько отзывов пассажиров, где эта проблема встречается.\n"
        "На основе этих отзывов сформулируй ОДНУ строку в формате:\n"
        f"{tag}: краткое пояснение\n\n"
        "Требования:\n"
        "• Пояснение — 1–2 коротких предложения по-русски.\n"
        "• Не давай советов, не используй проценты.\n"
        "• Не начинай с «во-первых», «итог», «вывод» и т.п.\n"
        "• Не упоминай, что ты модель. Только факт по жалобам.\n\n"
        "Отзывы:\n"
    )
    body = "\n".join(f"- {e}" for e in examples)
    return f"{header}{body}\n\nОтвет:\n"

def _qa_trim_context_block(text: str) -> str:
    text = _strip_noise(_normalize_abbreviations(text or ""))
    if len(text) > QA_MAX_CONTEXT_CHARS:
        text = text[:QA_MAX_CONTEXT_CHARS].rstrip() + "…"
    return text


def _qa_answer_prompt(query_text: str, context_blocks: List[str]) -> str:
    blocks = []
    for i, blk in enumerate(context_blocks[:QA_MAX_CONTEXT_ITEMS], start=1):
        blocks.append(f"[Фрагмент {i}]\n{_qa_trim_context_block(blk)}")

    context_text = "\n\n".join(blocks)

    return (
        "Ты анализируешь найденные сообщения пассажиров из Telegram.\n\n"
        f"Вопрос пользователя:\n{query_text}\n\n"
        "Ниже приведён найденный контекст. Отвечай ТОЛЬКО на основе этого контекста.\n"
        "Нельзя придумывать факты, причины или выводы, которых нет в тексте.\n"
        "Если прямых подтверждений мало или данные неоднозначны — скажи это явно.\n"
        "Не превращай ответ в общую сводку по всем темам. Отвечай именно на вопрос пользователя.\n\n"
        "Требования к ответу:\n"
        "1. Первая строка: 'Краткий ответ: ...'\n"
        "2. Затем 2–5 пунктов, каждый с маркером '🔹'\n"
        "3. Пункты должны быть конкретными и относиться именно к вопросу\n"
        "4. Без приветствий, без мета-комментариев, без упоминаний модели\n"
        "5. Без рекомендаций и без канцелярских формулировок\n\n"
        "Контекст:\n"
        f"{context_text}\n\n"
        "Ответ:\n"
    )


def answer_with_context(
    query_text: str,
    context_blocks: List[str],
    max_bullets: int = QA_MAX_BULLETS,
) -> str:
    if not context_blocks:
        return "Не найдено достаточно контекста для ответа."

    prompt = _qa_answer_prompt(query_text, context_blocks)
    raw = _llm_generate(prompt, max_tokens=QA_MAX_TOKENS)
    raw = _strip_think(raw or "").strip()

    if not raw:
        return ""

    # Чистим служебные/дублирующиеся заголовки
    raw = re.sub(r"^\s*✅\s*Ответ\s*", "", raw, flags=re.I).strip()
    raw = re.sub(r"^\s*🔹\s*✅\s*Ответ\s*", "", raw, flags=re.I).strip()
    raw = re.sub(r"^\s*✅\s*Ответ\s*$", "", raw, flags=re.I | re.M).strip()
    raw = re.sub(r"^\s*🔹\s*✅\s*Ответ\s*$", "", raw, flags=re.I | re.M).strip()

    # Отсекаем мета/дисклеймерные ответы модели
    raw_l = raw.lower()
    bad_markers = [
        "генеративные языковые модели",
        "не обладают собственным мнением",
        "не имею собственного мнения",
        "как языковая модель",
        "как модель",
        "чувствительные темы могут быть ограничены",
        "я не могу помочь",
        "я не могу ответить",
        "я не могу предоставить",
        "не могу обсуждать",
    ]
    if any(m in raw_l for m in bad_markers):
        _setup_logger("[QA] dropped meta/disclaimer answer")
        return ""

    lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""

    out: List[str] = []
    bullet_count = 0
    have_summary_line = False

    for ln in lines:
        s = ln.strip()
        if not s:
            continue

        s = re.sub(r"^\s*(?:[-•*—–]+)\s*", f"{BULLET} ", s)

        if s.lower().startswith("краткий ответ:"):
            if not have_summary_line:
                out.append(s)
                have_summary_line = True
            continue

        if s.startswith(BULLET):
            if bullet_count < max_bullets:
                out.append(s)
                bullet_count += 1
            continue

        if not have_summary_line:
            out.append(f"Краткий ответ: {s}")
            have_summary_line = True
        elif bullet_count < max_bullets:
            out.append(f"{BULLET} {s}")
            bullet_count += 1

    if not out:
        return ""

    if not any(line.lower().startswith("краткий ответ:") for line in out):
        first = out[0]
        if first.startswith(BULLET):
            out.insert(0, "Краткий ответ: по найденным сообщениям удалось выделить несколько релевантных наблюдений.")
        else:
            out.insert(0, "Краткий ответ: " + first)

    return "\n".join(out)

# ===================== ПУБЛИЧНАЯ ФУНКЦИЯ =====================

def summarize_topn(
    df: pd.DataFrame,
    query_text: str,
    text_col: str = "Описание",
    datetime_col: Optional[str] = None,
    top_n: int = SUMMARY_TOP_N,
    skip_sem: Optional[bool] = None,
) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "Не найдено подходящих комментариев."
    if text_col not in df.columns:
        return f"Нет колонки текста '{text_col}'."

    # sem on/off
    if skip_sem is None:
        do_sem = not SUMMARY_SKIP_SEM
    else:
        do_sem = not bool(skip_sem)

    # 1) top-N
    if do_sem:
        few = _semantic_select_top(df, query_text, text_col=text_col, n=top_n, datetime_col=datetime_col)
    else:
        base = df.copy()
        if datetime_col and datetime_col in base.columns:
            try:
                base = base.sort_values(datetime_col, ascending=False, na_position="last")
            except Exception:
                pass
        few = _dedup_by_text(base, text_col).head(top_n)

    if few.empty:
        return "Не найдено подходящих комментариев."

    raw_texts = few[text_col].astype(str).tolist()
    items: List[str] = []
    for t in raw_texts:
        c = _clean_for_summary(t)
        if c:
            items.append(c)
    if not items:
        return "Не найдено подходящих комментариев."

    log.info("[LLM] inputs=%d do_sem=%s", len(items), do_sem)

    # 2) tagging
    tags_per_comment: List[List[str]] = [[] for _ in items]

    if SUMMARY_PROVIDER in {"ollama", "gigachat"}:
        for start in range(0, len(items), TAG_BATCH_SIZE):
            chunk = items[start:start + TAG_BATCH_SIZE]
            prompt = _tagging_prompt(query_text, chunk)
            resp = _llm_generate(prompt, max_tokens=LLM_MAX_TOK)
            _parse_tagging_output(resp, base_idx=start, tags_per_comment=tags_per_comment)

    tag_to_indices: Dict[str, List[int]] = {}
    for i, tags in enumerate(tags_per_comment):
        for t in tags:
            tag_to_indices.setdefault(t, []).append(i)

    if not tag_to_indices:
        log.warning("[TAGS] empty -> fallback aggregator")
        bullets = _aggregate_humanish(few, text_col, query_text, want_n=SUMMARY_MAX_BULLETS)
        if not bullets:
            bullets = [f"{BULLET} Жалобы: {items[0][:300]}"]
        return "\n".join(bullets[:SUMMARY_MAX_BULLETS])

    # 3) pick tags
    tag_counts = {t: len(set(idxs)) for t, idxs in tag_to_indices.items()}
    sorted_tags = sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)

    main_tags: List[str] = [t for t, c in sorted_tags if c >= 2]
    if not main_tags:
        main_tags = [t for t, _ in sorted_tags]
    main_tags = main_tags[:SUMMARY_MAX_BULLETS]
    log.info("[TAGS] picked=%s", main_tags)

    # 4) one-line per tag
    bullets_raw: List[str] = []
    if SUMMARY_PROVIDER in {"ollama", "gigachat"}:
        for tag in main_tags:
            idxs = list(dict.fromkeys(tag_to_indices.get(tag, [])))
            if not idxs:
                continue
            examples = [items[i] for i in idxs[:5]]
            prompt = _tag_summary_prompt(tag, examples)
            resp = _llm_generate(prompt, max_tokens=LLM_MAX_TOK)
            if not resp:
                continue
            lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
            if not lines:
                continue
            line0 = lines[0]
            if ":" not in line0:
                line0 = f"{tag}: {line0}"
            fmt = _post_format([line0], max_n=1)
            if fmt:
                bullets_raw.extend(fmt)

    bullets = _anti_hallucination_filter(bullets_raw, items, min_hits=2)

    if len(bullets) < SUMMARY_MIN_BULLETS:
        extra = [b for b in bullets_raw if b not in bullets]
        for b in extra:
            if len(bullets) >= SUMMARY_MAX_BULLETS:
                break
            bullets.append(b)

    if not bullets:
        log.warning("[ANSWER] empty after filters -> fallback")
        bullets = _aggregate_humanish(few, text_col, query_text, want_n=SUMMARY_MAX_BULLETS)
        if not bullets:
            bullets = [f"{BULLET} Жалобы: {items[0][:300]}"]

    return "\n".join(bullets[:SUMMARY_MAX_BULLETS])
