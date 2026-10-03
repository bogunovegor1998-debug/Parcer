from __future__ import annotations

import os
import re
import json
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError, RPCError
from telethon.tl.types import Message

from psycopg_pool import ConnectionPool

import sys
from datetime import datetime

LOG_PATH = Path(__file__).with_name("ingest_scheduler.log")

def logp(*args):
    s = " ".join(str(a) for a in args)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {s}\n"

    # в консоль (если есть)
    try:
        print(line, end="")
        sys.stdout.flush()
    except Exception:
        pass

    # в файл
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass



# ---------------------------
# Helpers
# ---------------------------

def _load_env():
    load_dotenv(Path(__file__).with_name(".env"))
    load_dotenv(override=False)


def utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def norm_channel(s: str) -> str:
    s = (s or "").strip()
    s = s.lstrip("@").strip()
    return s


def load_list(env_var: str, filename: str) -> List[str]:
    """
    Берём источники из:
      1) ENV (например TG_CHANNELS или TG_GROUPS) — через запятую
      2) файла рядом (channels.txt / groups.txt) — по одному на строку, можно с @, можно #комменты
    """
    items: List[str] = []

    env = (os.getenv(env_var) or "").strip()
    if env:
        items.extend([norm_channel(c) for c in env.split(",") if c.strip()])

    path = Path(__file__).with_name(filename)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(norm_channel(s))

    # unique preserve order
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_post_link(username: Optional[str], msg_id: int) -> Optional[str]:
    if not username:
        return None
    return f"https://t.me/{username}/{msg_id}"

def build_msg_link(username: Optional[str], chat_id: int, msg_id: int) -> Optional[str]:
    """
    Универсальная ссылка на сообщение:
    - если есть username: https://t.me/<username>/<msg_id>
    - иначе (приватные группы/мегагруппы): https://t.me/c/<internal>/<msg_id>
    """
    if msg_id is None:
        return None
    if username:
        return f"https://t.me/{username}/{int(msg_id)}"

    cid = int(chat_id or 0)
    if cid == 0:
        return None

    # Telegram internal id для /c/ ссылок: abs(chat_id) - 1000000000000
    if cid < 0:
        internal = abs(cid) - 1000000000000
    else:
        internal = cid
    return f"https://t.me/c/{internal}/{int(msg_id)}"


def parse_int_flag(argv: List[str], name: str) -> int:
    """
    --limit=123
    --comments-backfill=200
    --since-days=365
    --batch=300
    """
    pat = re.compile(rf"--{re.escape(name)}=(\d+)\b")
    m = pat.search(" ".join(argv))
    return int(m.group(1)) if m else 0

def parse_str_flag(argv: List[str], name: str) -> str:
    pat = re.compile(rf"--{re.escape(name)}=([0-9\-]+)\b")
    m = pat.search(" ".join(argv))
    return (m.group(1) if m else "").strip()


def build_telethon_proxy_from_env() -> Optional[dict]:
    """
    Собирает proxy-конфиг для Telethon из .env.

    Поддерживаются 2 варианта:
    1) TG_PROXY=socks5://user:pass@host:port
       TG_PROXY=http://user:pass@host:port
    2) TG_PROXY_TYPE=socks5
       TG_PROXY_HOST=127.0.0.1
       TG_PROXY_PORT=1080
       TG_PROXY_USER=...
       TG_PROXY_PASSWORD=...
       TG_PROXY_RDNS=true

    Возвращает dict для TelegramClient(..., proxy=...)
    или None, если прокси не задан.
    """
    from urllib.parse import urlparse, unquote

    tg_proxy = (os.getenv("TG_PROXY") or "").strip()
    if tg_proxy:
        p = urlparse(tg_proxy)

        scheme = (p.scheme or "").lower().strip()
        if scheme not in {"socks5", "socks4", "http"}:
            raise SystemExit(
                f"Неподдерживаемая схема TG_PROXY: {scheme!r}. "
                f"Ожидается socks5 / socks4 / http"
            )

        host = (p.hostname or "").strip()
        port = int(p.port or 0)
        if not host or not port:
            raise SystemExit("В TG_PROXY не распознаны host/port")

        username = unquote(p.username) if p.username else None
        password = unquote(p.password) if p.password else None

        return {
            "proxy_type": scheme,
            "addr": host,
            "port": port,
            "username": username,
            "password": password,
            "rdns": True,
        }

    proxy_type = (os.getenv("TG_PROXY_TYPE") or "").strip().lower()
    proxy_host = (os.getenv("TG_PROXY_HOST") or "").strip()
    proxy_port = (os.getenv("TG_PROXY_PORT") or "").strip()
    proxy_user = (os.getenv("TG_PROXY_USER") or "").strip() or None
    proxy_password = (os.getenv("TG_PROXY_PASSWORD") or "").strip() or None
    proxy_rdns = (os.getenv("TG_PROXY_RDNS") or "true").strip().lower() in {"1", "true", "yes", "y"}

    if not proxy_type and not proxy_host and not proxy_port:
        return None

    if proxy_type not in {"socks5", "socks4", "http"}:
        raise SystemExit(
            f"Неподдерживаемый TG_PROXY_TYPE: {proxy_type!r}. "
            f"Ожидается socks5 / socks4 / http"
        )

    if not proxy_host:
        raise SystemExit("Не задан TG_PROXY_HOST")

    if not proxy_port.isdigit():
        raise SystemExit("Не задан или некорректен TG_PROXY_PORT")

    return {
        "proxy_type": proxy_type,
        "addr": proxy_host,
        "port": int(proxy_port),
        "username": proxy_user,
        "password": proxy_password,
        "rdns": proxy_rdns,
    }

# ---------------------------
# Storage (PostgreSQL)
# ---------------------------

@dataclass
class SourceRow:
    id: int
    username: str
    title: Optional[str]
    peer_id: Optional[int]
    access_hash: Optional[int]


class Storage:
    def __init__(self, dsn: str):
        self.pool = ConnectionPool(conninfo=dsn, min_size=1, max_size=5)
        self._tg_comments_cols: Optional[set[str]] = None
        self._tg_chat_cols: Optional[set[str]] = None

    def close(self):
        self.pool.close()

    def ensure_source(self, username: str, title: Optional[str], peer_id: int, access_hash: int) -> SourceRow:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tg_sources(username, title, peer_id, access_hash, updated_at)
                    VALUES (%s, %s, %s, %s, now())
                    ON CONFLICT (username) DO UPDATE
                      SET title = EXCLUDED.title,
                          peer_id = EXCLUDED.peer_id,
                          access_hash = EXCLUDED.access_hash,
                          updated_at = now()
                    RETURNING id, username, title, peer_id, access_hash;
                    """,
                    (username, title, peer_id, access_hash),
                )
                row = cur.fetchone()
                assert row
                source = SourceRow(*row)

                cur.execute(
                    """
                    INSERT INTO tg_source_state(source_id, last_msg_id, last_scan_at)
                    VALUES (%s, 0, NULL)
                    ON CONFLICT (source_id) DO NOTHING;
                    """,
                    (source.id,),
                )
                conn.commit()
                return source

    def get_last_msg_id(self, source_id: int) -> int:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT last_msg_id FROM tg_source_state WHERE source_id=%s;", (source_id,))
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def set_last_msg_id(self, source_id: int, last_msg_id: int):
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tg_source_state
                       SET last_msg_id=%s, last_scan_at=now()
                     WHERE source_id=%s;
                    """,
                    (last_msg_id, source_id),
                )
                conn.commit()

    # ---- thread state (для комментов к постам) ----
    def get_last_comment_id_cur(self, cur, source_id: int, root_chat_id: int, root_msg_id: int) -> int:
        cur.execute(
            """
            SELECT last_comment_id
              FROM tg_thread_state
             WHERE source_id=%s AND root_chat_id=%s AND root_msg_id=%s;
            """,
            (source_id, root_chat_id, root_msg_id),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def set_last_comment_id_cur(self, cur, source_id: int, root_chat_id: int, root_msg_id: int, last_comment_id: int):
        cur.execute(
            """
            INSERT INTO tg_thread_state(source_id, root_chat_id, root_msg_id, last_comment_id, last_scan_at)
            VALUES (%s,%s,%s,%s, now())
            ON CONFLICT (source_id, root_chat_id, root_msg_id) DO UPDATE
              SET last_comment_id=EXCLUDED.last_comment_id,
                  last_scan_at=now();
            """,
            (source_id, root_chat_id, root_msg_id, last_comment_id),
        )

    
    def _ensure_comments_cols(self, cur):
        if self._tg_comments_cols is not None:
            return
        cur.execute(
            """
            SELECT column_name
              FROM information_schema.columns
             WHERE table_schema='public' AND table_name='tg_comments';
            """
        )
        self._tg_comments_cols = {str(r[0]) for r in cur.fetchall()}

    def _ensure_chat_cols(self, cur):
        if self._tg_chat_cols is not None:
            return
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='tg_chat_messages';
            """
        )
        self._tg_chat_cols = {str(r[0]) for r in cur.fetchall()}


    # ---- upsert post/comment with existing cursor (batch commit outside) ----
    def upsert_post_cur(self, cur, source_id: int, chat_id: int, msg: Message, permalink: Optional[str], username: str):
        text = msg.message if msg.message is not None else (getattr(msg, "raw_text", None) or "")
        payload = {
            "id": msg.id,
            "date": msg.date.isoformat() if msg.date else None,
            "message": text,
            "views": getattr(msg, "views", None),
            "forwards": getattr(msg, "forwards", None),
            "replies": getattr(getattr(msg, "replies", None), "replies", None),
            "chat_id": getattr(msg, "chat_id", None),
            "sender_id": getattr(msg, "sender_id", None),
            "username": username,
        }
        doc_id = f"post:{chat_id}:{msg.id}"

        cur.execute(
            """
            INSERT INTO tg_posts(
                source_id, chat_id, msg_id, msg_date, sender_id, text,
                views, forwards, replies_count, permalink, raw_json
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT (chat_id, msg_id) DO UPDATE
            SET msg_date=EXCLUDED.msg_date,
                sender_id=EXCLUDED.sender_id,
                text=EXCLUDED.text,
                views=EXCLUDED.views,
                forwards=EXCLUDED.forwards,
                replies_count=EXCLUDED.replies_count,
                permalink=EXCLUDED.permalink,
                raw_json=EXCLUDED.raw_json;
            """,
            (
                source_id, chat_id, msg.id, utc(msg.date), getattr(msg, "sender_id", None),
                text, getattr(msg, "views", None), getattr(msg, "forwards", None),
                getattr(getattr(msg, "replies", None), "replies", None),
                permalink, json.dumps(payload, ensure_ascii=False),
            ),
        )

    def upsert_chat_message_cur(
        self,
        cur,
        source_id: int,
        chat_id: int,
        msg: Message,
        permalink: Optional[str],
        username: str,
    ):
        text = msg.message if msg.message is not None else (getattr(msg, "raw_text", None) or "")
        reply_to = getattr(getattr(msg, "reply_to", None), "reply_to_msg_id", None)

        payload = {
            "id": msg.id,
            "date": msg.date.isoformat() if msg.date else None,
            "message": text,
            "reply_to": reply_to,
            "chat_id": getattr(msg, "chat_id", None),
            "sender_id": getattr(msg, "sender_id", None),
            "username": username,
        }

        self._ensure_chat_cols(cur)
        cols = self._tg_chat_cols or set()
        has_parent = "parent_msg_id" in cols

        if has_parent:
            cur.execute(
                """
                INSERT INTO tg_chat_messages(
                    source_id, chat_id, msg_id, msg_date, sender_id, text,
                    parent_msg_id, permalink, raw_json
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (chat_id, msg_id) DO UPDATE
                SET msg_date=EXCLUDED.msg_date,
                    sender_id=EXCLUDED.sender_id,
                    text=EXCLUDED.text,
                    parent_msg_id=EXCLUDED.parent_msg_id,
                    permalink=EXCLUDED.permalink,
                    raw_json=EXCLUDED.raw_json;
                """,
                (
                    source_id,
                    chat_id,
                    msg.id,
                    utc(msg.date),
                    getattr(msg, "sender_id", None),
                    text,
                    int(reply_to) if reply_to else None,
                    permalink,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO tg_chat_messages(
                    source_id, chat_id, msg_id, msg_date, sender_id, text,
                    permalink, raw_json
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (chat_id, msg_id) DO UPDATE
                SET msg_date=EXCLUDED.msg_date,
                    sender_id=EXCLUDED.sender_id,
                    text=EXCLUDED.text,
                    permalink=EXCLUDED.permalink,
                    raw_json=EXCLUDED.raw_json;
                """,
                (
                    source_id,
                    chat_id,
                    msg.id,
                    utc(msg.date),
                    getattr(msg, "sender_id", None),
                    text,
                    permalink,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )


    def upsert_comment_cur(
        self,
        cur,
        source_id: int,
        root_chat_id: int,
        root_msg_id: int,
        msg: Message,
        permalink: Optional[str],
        channel_username: str,
        chat_id_override: Optional[int] = None,
    ):
        text = msg.message if msg.message is not None else (getattr(msg, "raw_text", None) or "")

        # chat_id: сначала считаем chat_id (чтобы не было ошибок с doc_id/логикой)
        chat_id = int(chat_id_override or int(getattr(msg, "chat_id", 0) or 0) or 0)
        if chat_id == 0:
            chat_id = int(root_chat_id)

        reply_to = getattr(getattr(msg, "reply_to", None), "reply_to_msg_id", None)
        reply_to_msg_id = int(reply_to) if reply_to else None

        payload = {
            "id": msg.id,
            "date": msg.date.isoformat() if msg.date else None,
            "message": text,
            "reply_to": reply_to,
            "chat_id": getattr(msg, "chat_id", None),
            "sender_id": getattr(msg, "sender_id", None),
            "channel_username": channel_username,
            "root_msg_id": root_msg_id,
            "root_chat_id": root_chat_id,
        }

        # Колонки (чтобы работать на разных схемах)
        self._ensure_comments_cols(cur)
        cols = self._tg_comments_cols or set()
        has_reply_to = "reply_to_msg_id" in cols

        # ВАЖНО: doc_id НЕ вставляем (у тебя он может быть GENERATED ALWAYS)
        if has_reply_to:
            cur.execute(
                """
                INSERT INTO tg_comments(
                    source_id, root_chat_id, root_msg_id, chat_id, msg_id,
                    msg_date, sender_id, text, permalink, raw_json,
                    reply_to_msg_id
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                ON CONFLICT (chat_id, msg_id) DO UPDATE
                SET msg_date=EXCLUDED.msg_date,
                    sender_id=EXCLUDED.sender_id,
                    text=EXCLUDED.text,
                    permalink=EXCLUDED.permalink,
                    raw_json=EXCLUDED.raw_json,
                    reply_to_msg_id=EXCLUDED.reply_to_msg_id;
                """,
                (
                    source_id,
                    root_chat_id,
                    root_msg_id,
                    chat_id,
                    msg.id,
                    utc(msg.date),
                    getattr(msg, "sender_id", None),
                    text,
                    permalink,
                    json.dumps(payload, ensure_ascii=False),
                    reply_to_msg_id,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO tg_comments(
                    source_id, root_chat_id, root_msg_id, chat_id, msg_id,
                    msg_date, sender_id, text, permalink, raw_json
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT (chat_id, msg_id) DO UPDATE
                SET msg_date=EXCLUDED.msg_date,
                    sender_id=EXCLUDED.sender_id,
                    text=EXCLUDED.text,
                    permalink=EXCLUDED.permalink,
                    raw_json=EXCLUDED.raw_json;
                """,
                (
                    source_id,
                    root_chat_id,
                    root_msg_id,
                    chat_id,
                    msg.id,
                    utc(msg.date),
                    getattr(msg, "sender_id", None),
                    text,
                    permalink,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def recent_post_ids(self, source_id: int, limit: int, only_with_replies: bool = True) -> List[int]:
        if limit <= 0:
            return []
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                if only_with_replies:
                    cur.execute(
                        """
                        SELECT msg_id
                          FROM tg_posts
                         WHERE source_id=%s AND COALESCE(replies_count,0) > 0
                         ORDER BY msg_id DESC
                         LIMIT %s;
                        """,
                        (source_id, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT msg_id
                          FROM tg_posts
                         WHERE source_id=%s
                         ORDER BY msg_id DESC
                         LIMIT %s;
                        """,
                        (source_id, limit),
                    )
                return [int(r[0]) for r in cur.fetchall()]




# ---------------------------
# Telegram ingest
# ---------------------------

def _is_channel_entity(entity) -> bool:
    """
    Telethon: у Channel обычно broadcast=True,
    у мегагруппы/чата broadcast=False и megagroup=True.
    """
    try:
        if getattr(entity, "broadcast", False):
            return True
    except Exception:
        pass
    return False


async def ingest_comments_for_post(
    client: TelegramClient,
    store: Storage,
    source_id: int,
    channel_entity,
    channel_username: str,
    root_chat_id: int,
    root_msg_id: int,
    conn,
    cur,
    cutoff_dt: Optional[datetime],
    batch_size: int,
    limit_per_post: int = 500,
) -> int:
    """
    Комментарии к посту (reply_to=root_msg_id).
    Тихие логи: прогресс каждые N комментов, плюс итог по посту.
    """
    default_every = int(os.getenv("INGEST_LOG_EVERY", "500") or "500")
    LOG_EVERY_COMMENTS = int(os.getenv("INGEST_LOG_EVERY_COMMENTS", str(default_every)) or str(default_every))

    last_c = store.get_last_comment_id_cur(cur, source_id, root_chat_id, root_msg_id)
    new_max = last_c
    cnt = 0

    try:
        # если первый раз и есть cutoff — идём от новых к старым и break по дате
        if last_c == 0 and cutoff_dt is not None:
            async for c in client.iter_messages(channel_entity, reply_to=root_msg_id):
                if not getattr(c, "id", None):
                    continue
                if c.date is None:
                    continue
                if utc(c.date) < cutoff_dt:
                    break

                text = c.message if c.message is not None else (getattr(c, "raw_text", None) or "")
                if not text.strip():
                    continue

                permalink = build_post_link(channel_username, root_msg_id)

                store.upsert_comment_cur(
                    cur=cur,
                    source_id=source_id,
                    root_chat_id=root_chat_id,
                    root_msg_id=root_msg_id,
                    msg=c,
                    permalink=permalink,
                    channel_username=channel_username,
                    chat_id_override=None,  # у комментов может быть discussion group id
                )

                new_max = max(new_max, c.id)
                cnt += 1

                if cnt % batch_size == 0:
                    conn.commit()

                # тихий прогресс
                #if LOG_EVERY_COMMENTS > 0 and (cnt % LOG_EVERY_COMMENTS == 0):
                    #logp(f"[..] comments: root={root_chat_id}:{root_msg_id} loaded={cnt} (last_comment_id~={new_max})")

                if limit_per_post and cnt >= limit_per_post:
                    break

        else:
            # инкрементально: только новые после last_c
            async for c in client.iter_messages(channel_entity, reply_to=root_msg_id, min_id=last_c, reverse=True):
                if not getattr(c, "id", None):
                    continue

                if cutoff_dt is not None and c.date is not None and utc(c.date) < cutoff_dt:
                    continue

                text = c.message if c.message is not None else (getattr(c, "raw_text", None) or "")
                if not text.strip():
                    continue

                permalink = build_post_link(channel_username, root_msg_id)

                store.upsert_comment_cur(
                    cur=cur,
                    source_id=source_id,
                    root_chat_id=root_chat_id,
                    root_msg_id=root_msg_id,
                    msg=c,
                    permalink=permalink,
                    channel_username=channel_username,
                    chat_id_override=None,
                )

                new_max = max(new_max, c.id)
                cnt += 1

                if cnt % batch_size == 0:
                    conn.commit()

                if LOG_EVERY_COMMENTS > 0 and (cnt % LOG_EVERY_COMMENTS == 0):
                    logp(f"[..] comments: root={root_chat_id}:{root_msg_id} loaded={cnt} (last_comment_id~={new_max})")

                if limit_per_post and cnt >= limit_per_post:
                    break

        # обновляем курсор по треду только если реально что-то новое нашли
        if new_max != last_c:
            store.set_last_comment_id_cur(cur, source_id, root_chat_id, root_msg_id, new_max)
            conn.commit()

        # итоговый лог только если что-то добавили
        #if cnt > 0:
            #logp(f"[OK] comments: root={root_chat_id}:{root_msg_id} +{cnt} (cursor {last_c} -> {new_max})")

    except FloodWaitError as e:
        await asyncio.sleep(int(e.seconds) + 1)
    except RPCError:
        return 0

    return cnt


async def ingest_channel(
    client: TelegramClient,
    store: Storage,
    username: str,
    entity,
    with_comments: bool,
    limit_new_posts: int,
    comments_backfill: int,
    cutoff_dt: Optional[datetime],
    batch_size: int,
):
    username = norm_channel(username)

    peer_id = int(getattr(entity, "id"))
    access_hash = int(getattr(entity, "access_hash", 0) or 0)
    title = getattr(entity, "title", None)
    public_username = getattr(entity, "username", None) or username

    # как часто печатать прогресс (в постах)
    LOG_EVERY_POSTS = int(os.getenv("INGEST_LOG_EVERY_POSTS", "500") or "500")
    # как часто печатать прогресс бекфилла (в постах)
    LOG_EVERY_BACKFILL = int(os.getenv("INGEST_LOG_EVERY_BACKFILL", str(LOG_EVERY_POSTS)) or str(LOG_EVERY_POSTS))

    source = store.ensure_source(username=username, title=title, peer_id=peer_id, access_hash=access_hash)
    last_id = store.get_last_msg_id(source.id)

    new_max_id = last_id
    posts_scanned = 0
    comments_added_total = 0
    threads_with_new = 0

    with store.pool.connection() as conn:
        with conn.cursor() as cur:
            try:
                # Первый прогон + cutoff: идём от новых к старым и break
                if last_id == 0 and cutoff_dt is not None:
                    async for msg in client.iter_messages(entity):
                        if not getattr(msg, "id", None) or msg.date is None:
                            continue
                        if utc(msg.date) < cutoff_dt:
                            break

                        text = msg.message if msg.message is not None else (getattr(msg, "raw_text", None) or "")
                        if not text.strip():
                            continue

                        permalink = build_post_link(public_username, msg.id)
                        store.upsert_post_cur(cur, source.id, peer_id, msg, permalink, public_username)

                        new_max_id = max(new_max_id, msg.id)
                        posts_scanned += 1

                        if with_comments:
                            added = await ingest_comments_for_post(
                                client, store, source.id, entity, public_username,
                                peer_id, msg.id, conn, cur, cutoff_dt, batch_size,
                                limit_per_post=0  # 0 = без лимита
                            )
                            if added:
                                comments_added_total += int(added)
                                threads_with_new += 1

                        # коммит батчами по постам
                        if posts_scanned % batch_size == 0:
                            conn.commit()

                        # редкий прогресс
                        if LOG_EVERY_POSTS > 0 and posts_scanned % LOG_EVERY_POSTS == 0:
                            logp(
                                f"[..] @{username}: posts_scanned={posts_scanned} "
                                f"comments_added={comments_added_total} threads_with_new={threads_with_new} "
                                f"max_post_id={new_max_id}"
                            )

                        if limit_new_posts and posts_scanned >= limit_new_posts:
                            break

                else:
                    # Обычный режим: только новые по msg_id
                    async for msg in client.iter_messages(entity, min_id=last_id, reverse=True):
                        if not getattr(msg, "id", None):
                            continue

                        if cutoff_dt is not None and msg.date is not None and utc(msg.date) < cutoff_dt:
                            continue

                        text = msg.message if msg.message is not None else (getattr(msg, "raw_text", None) or "")
                        if not text.strip():
                            continue

                        permalink = build_post_link(public_username, msg.id)
                        store.upsert_post_cur(cur, source.id, peer_id, msg, permalink, public_username)

                        new_max_id = max(new_max_id, msg.id)
                        posts_scanned += 1

                        if with_comments:
                            added = await ingest_comments_for_post(
                                client, store, source.id, entity, public_username,
                                peer_id, msg.id, conn, cur, cutoff_dt, batch_size,
                                limit_per_post=0
                            )
                            if added:
                                comments_added_total += int(added)
                                threads_with_new += 1

                        if posts_scanned % batch_size == 0:
                            conn.commit()

                        if LOG_EVERY_POSTS > 0 and posts_scanned % LOG_EVERY_POSTS == 0:
                            logp(
                                f"[..] @{username}: posts_scanned={posts_scanned} "
                                f"comments_added={comments_added_total} threads_with_new={threads_with_new} "
                                f"max_post_id={new_max_id}"
                            )

                        if limit_new_posts and posts_scanned >= limit_new_posts:
                            break

            except FloodWaitError as e:
                await asyncio.sleep(int(e.seconds) + 1)

            conn.commit()

    if new_max_id != last_id:
        store.set_last_msg_id(source.id, new_max_id)

    # ВАЖНО: бекфилл имеет смысл в основном в ИНКРЕМЕНТАЛЬНОМ режиме,
    # когда новые посты не пришли, но у старых могли добавиться комменты.
    # На первом полном прогоне (last_id == 0) его можно пропустить.
    bf_cnt = 0
    if with_comments and comments_backfill > 0 and last_id != 0:
        ids = store.recent_post_ids(source.id, comments_backfill, only_with_replies=True)
        ids = list(reversed(ids))
        with store.pool.connection() as conn:
            with conn.cursor() as cur:
                for i, mid in enumerate(ids, 1):
                    added = await ingest_comments_for_post(
                        client, store, source.id, entity, public_username,
                        peer_id, mid, conn, cur, cutoff_dt, batch_size,
                        limit_per_post=0
                    )
                    bf_cnt += int(added or 0)

                    if LOG_EVERY_BACKFILL > 0 and i % LOG_EVERY_BACKFILL == 0:
                        logp(f"[..] @{username}: backfill scanned={i}/{len(ids)} posts, comments_added~={bf_cnt}")

                conn.commit()

    logp(
        f"[OK] @{username}: posts +{posts_scanned} (cursor {last_id} -> {new_max_id}) "
        f"| comments +{comments_added_total} | backfill_comments +{bf_cnt}"
    )




async def ingest_group(
    client: TelegramClient,
    store: Storage,
    username: str,
    entity,
    limit_new_msgs: int,
    cutoff_dt: Optional[datetime],
    batch_size: int,
):
    """
    Новый формат: группы/чаты.
    Все сообщения пишем в tg_comments.
    root_msg_id:
      - если reply_to есть -> root_msg_id = reply_to
      - иначе root_msg_id = self
    """
    username = norm_channel(username)

    peer_id = int(getattr(entity, "id"))
    access_hash = int(getattr(entity, "access_hash", 0) or 0)
    title = getattr(entity, "title", None)
    public_username = getattr(entity, "username", None) or username

    source = store.ensure_source(username=username, title=title, peer_id=peer_id, access_hash=access_hash)
    last_id = store.get_last_msg_id(source.id)

    new_max_id = last_id
    cnt = 0

    with store.pool.connection() as conn:
        with conn.cursor() as cur:
            try:
                # Первый прогон + cutoff: идём от новых к старым и break
                if last_id == 0 and cutoff_dt is not None:
                    async for msg in client.iter_messages(entity):
                        if not getattr(msg, "id", None):
                            continue
                        if msg.date is None:
                            continue
                        if utc(msg.date) < cutoff_dt:
                            break

                        text = msg.message if msg.message is not None else (getattr(msg, "raw_text", None) or "")
                        if not text.strip():
                            continue

                        reply_to = getattr(getattr(msg, "reply_to", None), "reply_to_msg_id", None)
                        root_msg_id = int(reply_to) if reply_to else int(msg.id)
                        root_chat_id = int(peer_id)

                        permalink = build_msg_link(getattr(entity, "username", None), peer_id, msg.id)

                        store.upsert_chat_message_cur(
                            cur=cur,
                            source_id=source.id,
                            chat_id=peer_id,
                            msg=msg,
                            permalink=permalink,
                            username=public_username,
                        )


                        new_max_id = max(new_max_id, msg.id)
                        cnt += 1

                        if cnt % batch_size == 0:
                            conn.commit()
                            logp(f"[..] @{username}: loaded {cnt} msgs (last_id={new_max_id})")

                        if limit_new_msgs and cnt >= limit_new_msgs:
                            break
                else:
                    # Обычный режим: только новые по msg_id
                    async for msg in client.iter_messages(entity, min_id=last_id, reverse=True):
                        if not getattr(msg, "id", None):
                            continue

                        if cutoff_dt is not None and msg.date is not None and utc(msg.date) < cutoff_dt:
                            continue

                        text = msg.message if msg.message is not None else (getattr(msg, "raw_text", None) or "")
                        if not text.strip():
                            continue

                        reply_to = getattr(getattr(msg, "reply_to", None), "reply_to_msg_id", None)
                        root_msg_id = int(reply_to) if reply_to else int(msg.id)
                        root_chat_id = int(peer_id)

                        permalink = build_msg_link(getattr(entity, "username", None), peer_id, msg.id)

                        store.upsert_chat_message_cur(
                            cur=cur,
                            source_id=source.id,
                            chat_id=peer_id,
                            msg=msg,
                            permalink=permalink,
                            username=public_username,
                        )

                        new_max_id = max(new_max_id, msg.id)
                        cnt += 1

                        if cnt % batch_size == 0:
                            conn.commit()
                            print(f"[..] @{username}: loaded {cnt} msgs (last_id={new_max_id})")

                        if limit_new_msgs and cnt >= limit_new_msgs:
                            break

            except FloodWaitError as e:
                await asyncio.sleep(int(e.seconds) + 1)
            except RPCError:
                logp(f"[WARN] @{username}: RPCError, skip")
                return

            conn.commit()

    if new_max_id != last_id:
        store.set_last_msg_id(source.id, new_max_id)

    logp(f"[OK] @{username}: group msgs +{cnt} (cursor {last_id} -> {new_max_id})")


async def ingest_source(
    client: TelegramClient,
    store: Storage,
    username: str,
    with_comments: bool,
    limit_new_posts: int,
    comments_backfill: int,
    cutoff_dt: Optional[datetime],
    batch_size: int,
):
    username = norm_channel(username)
    entity = await client.get_entity(username)

    # авто-определение: канал или группа
    if _is_channel_entity(entity):
        await ingest_channel(
            client=client,
            store=store,
            username=username,
            entity=entity,
            with_comments=with_comments,
            limit_new_posts=limit_new_posts,
            comments_backfill=comments_backfill,
            cutoff_dt=cutoff_dt,
            batch_size=batch_size,
        )
    else:
        await ingest_group(
            client=client,
            store=store,
            username=username,
            entity=entity,
            limit_new_msgs=limit_new_posts,  # используем тот же --limit
            cutoff_dt=cutoff_dt,
            batch_size=batch_size,
        )


# ---------------------------
# Main
# ---------------------------

async def main():
    _load_env()

    api_id = int((os.getenv("TG_API_ID") or "0").strip() or "0")
    api_hash = (os.getenv("TG_API_HASH") or "").strip()
    session_file = (os.getenv("TG_SESSION_FILE") or "transport_feedback_tg.session").strip()
    session_path = Path(session_file)
    if not session_path.is_absolute():
        session_path = Path(__file__).with_name(session_file)
    session_file = str(session_path)

    pg_dsn = (os.getenv("PG_DSN") or "").strip()

    if not api_id or not api_hash:
        raise SystemExit("TG_API_ID / TG_API_HASH не заданы (.env)")
    if not pg_dsn:
        raise SystemExit("PG_DSN не задан (.env)")

    channels = load_list("TG_CHANNELS", "channels.txt")

    # legacy (если где-то еще используется старый TG_GROUPS / groups.txt)
    groups_legacy = load_list("TG_GROUPS", "groups.txt")

    # новое: дискуссионные группы и “просто чаты”
    discussions = load_list("TG_DISCUSSIONS", "discussions.txt")
    chats = load_list("TG_CHATS", "chats.txt")

    # общий список источников (уникально, с сохранением порядка)
    sources: List[str] = []
    seen = set()
    for x in channels + groups_legacy + discussions + chats:
        if x and x not in seen:
            seen.add(x)
            sources.append(x)

    logp(
        f"[CFG] channels={len(channels)} groups_legacy={len(groups_legacy)} "
        f"discussions={len(discussions)} chats={len(chats)} total_sources={len(sources)}"
    )

    if not sources:
        raise SystemExit(
            "Не задан список источников. Используй TG_CHANNELS/TG_GROUPS/TG_DISCUSSIONS/TG_CHATS "
            "или файлы channels.txt/groups.txt/discussions.txt/chats.txt"
        )

    argv = os.sys.argv[1:]

    with_comments = ("--with-comments" in argv)
    limit_new_posts = parse_int_flag(os.sys.argv, "limit")  # общий лимит
    comments_backfill = parse_int_flag(os.sys.argv, "comments-backfill")
    since_days = parse_int_flag(os.sys.argv, "since-days")
    batch_size = parse_int_flag(os.sys.argv, "batch") or 300

    since_date = parse_str_flag(os.sys.argv, "since-date")

    cutoff_dt = None
    if since_date:
        cutoff_dt = datetime.strptime(since_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        logp(f"[CFG] cutoff_dt = {cutoff_dt.isoformat()} (since-date={since_date})")
    elif since_days and since_days > 0:
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=since_days)
        logp(f"[CFG] cutoff_dt = {cutoff_dt.isoformat()} (since-days={since_days})")

    # Если включили комменты, но бекфилл не указали — небольшой дефолт
    if with_comments and comments_backfill == 0:
        comments_backfill = 200

    logp(
        f"[CFG] sources={len(sources)} with_comments={with_comments} "
        f"limit={limit_new_posts} backfill={comments_backfill} batch={batch_size}"
    )

    # ---------------------------
    # Proxy для Telethon
    # ---------------------------
    proxy = build_telethon_proxy_from_env()
    if proxy:
        safe_proxy = dict(proxy)
        if safe_proxy.get("password"):
            safe_proxy["password"] = "***"
        logp(f"[CFG] Telethon proxy enabled: {safe_proxy}")
    else:
        logp("[CFG] Telethon proxy disabled")

    store = Storage(pg_dsn)

    import inspect
    logp("[DBG] script file =", __file__)
    logp("[DBG] Storage class from =", inspect.getsourcefile(Storage))
    logp("[DBG] Storage dict has recent_post_ids =", "recent_post_ids" in Storage.__dict__)
    logp("[DBG] Storage dict has upsert_comment_cur =", "upsert_comment_cur" in Storage.__dict__)

    client = TelegramClient(
        session_file,
        api_id,
        api_hash,
        proxy=proxy,
        flood_sleep_threshold=60,
    )

    async with client:
        me = await client.get_me()
        logp(f"[TG] connected as id={getattr(me, 'id', None)}")

        for src in sources:
            try:
                logp(f"[SRC] start @{src}")
                await ingest_source(
                    client=client,
                    store=store,
                    username=src,
                    with_comments=with_comments,
                    limit_new_posts=limit_new_posts,
                    comments_backfill=comments_backfill,
                    cutoff_dt=cutoff_dt,
                    batch_size=batch_size,
                )
                logp(f"[SRC] done  @{src}")

            except FloodWaitError as e:
                logp(f"[WAIT] @{src}: FloodWait {e.seconds}s")
                await asyncio.sleep(int(e.seconds) + 1)
            except Exception as e:
                logp(f"[ERR] @{src}: {type(e).__name__}: {e}")

    store.close()


if __name__ == "__main__":
    asyncio.run(main())
