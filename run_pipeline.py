# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import datetime as dt
from zoneinfo import ZoneInfo
from pathlib import Path
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH, override=False)
load_dotenv(override=False)


# ----------------- logging -----------------
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "run_pipeline.log"

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

def resolve_script(name: str) -> str:
    """
    Разрешаем путь к скрипту:
    1) если задан абсолютный путь — используем его
    2) ищем в BASE_DIR
    3) ищем в BASE_DIR/ingest_build
    """
    name = (name or "").strip()
    if not name:
        return ""

    p = Path(name)
    if p.is_absolute():
        return str(p)

    cand1 = BASE_DIR / p
    if cand1.exists():
        return str(cand1)

    cand2 = BASE_DIR / "ingest_build" / p
    if cand2.exists():
        return str(cand2)

    # fallback: пусть subprocess напишет нормальную ошибку, но в логах будет ожидаемый путь
    return str(cand1)



# ----------------- lock (no параллельных запусков) -----------------
def acquire_lock(lock_path: Path, ttl_min: int = 180) -> int:
    """
    Простой lock-файл. Если завис старый — удаляем по TTL.
    Возвращает fd (держим открытым до конца).
    """
    now = time.time()

    if lock_path.exists():
        try:
            st = lock_path.stat()
            age_min = (now - st.st_mtime) / 60.0
            if age_min > ttl_min:
                logp(f"[LOCK] stale lock found (age={age_min:.1f} min) -> remove")
                lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} started={dt.datetime.now().isoformat()}".encode("utf-8"))
        os.fsync(fd)
        return fd
    except FileExistsError:
        raise SystemExit("[LOCK] another pipeline instance is running (lock exists)")


def release_lock(lock_path: Path, fd: int):
    try:
        os.close(fd)
    except Exception:
        pass
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


# ----------------- DB helpers -----------------
def _sqlalchemy_uri_from_env() -> str:
    pg_dsn = (os.getenv("PG_DSN") or "").strip()
    if pg_dsn:
        if pg_dsn.startswith("postgresql://"):
            return "postgresql+psycopg2://" + pg_dsn[len("postgresql://"):]
        return pg_dsn

    user = os.getenv("PG_USER", "postgres")
    pwd  = os.getenv("PG_PASSWORD", "")
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db   = os.getenv("PG_DB", "postgres")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"


def get_engine():
    return create_engine(_sqlalchemy_uri_from_env(), pool_pre_ping=True)


def relation_kind(engine, rel_fqn: str) -> Optional[str]:
    # rel_fqn like "public.thread_docs_channel"
    if "." not in rel_fqn:
        return None
    schema, name = rel_fqn.split(".", 1)
    sql = """
    SELECT c.relkind
      FROM pg_class c
      JOIN pg_namespace n ON n.oid = c.relnamespace
     WHERE n.nspname = :schema AND c.relname = :name
     LIMIT 1;
    """
    with engine.connect() as c:
        row = c.execute(text(sql), {"schema": schema, "name": name}).fetchone()
    return str(row[0]) if row else None


def refresh_matview(engine, rel_fqn: str):
    rk = relation_kind(engine, rel_fqn)
    if rk != "m":
        logp(f"[MV] {rel_fqn} is not a matview (relkind={rk}), skip refresh")
        return

    logp(f"[MV] refreshing {rel_fqn} ...")
    try:
        with engine.begin() as c:
            c.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {rel_fqn};"))
        logp(f"[MV] refreshed concurrently OK: {rel_fqn}")
    except Exception as e:
        logp(f"[MV] concurrent refresh failed: {rel_fqn} -> {type(e).__name__}: {e}")
        # fallback
        with engine.begin() as c:
            c.execute(text(f"REFRESH MATERIALIZED VIEW {rel_fqn};"))
        logp(f"[MV] refreshed (non-concurrent) OK: {rel_fqn}")


def get_threads_input_state(engine) -> Dict[str, Any]:
    """
    Состояние источника для threads-индекса:
    count + max(last_dt) по union двух таблиц.
    """
    sql = """
    WITH t AS (
      SELECT last_dt FROM public.thread_docs_channel
      UNION ALL
      SELECT last_dt FROM public.thread_docs_chat_reply
    )
    SELECT COUNT(*)::bigint AS rows,
           MAX(last_dt) AS max_dt
      FROM t;
    """
    with engine.connect() as c:
        row = c.execute(text(sql)).fetchone()
    rows = int(row[0] or 0)
    max_dt = row[1]
    max_dt_s = str(max_dt) if max_dt is not None else None
    return {"rows": rows, "max_dt": max_dt_s}


def get_micro_input_state(engine) -> Dict[str, Any]:
    """
    Состояние источника для micro-индекса: как в твоём билдере.
    """
    kinds = [k.strip() for k in (os.getenv("MICRO_CHAT_KINDS", "chat,group,discussion,chat_msg").split(",")) if k.strip()]
    sql = """
    SELECT COUNT(*)::bigint AS rows,
           MAX(msg_date) AS max_dt
      FROM public.feedback_raw_v3
     WHERE kind = ANY(:kinds)
       AND chat_id IS NOT NULL
       AND msg_id IS NOT NULL
       AND msg_date IS NOT NULL
       AND COALESCE(text,'') <> ''
       AND parent_msg_id IS NULL;
    """
    with engine.connect() as c:
        row = c.execute(text(sql), {"kinds": kinds}).fetchone()
    rows = int(row[0] or 0)
    max_dt = row[1]
    max_dt_s = str(max_dt) if max_dt is not None else None
    return {"rows": rows, "max_dt": max_dt_s, "kinds": kinds}


# ----------------- state files -----------------
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def same_state(a: Optional[Dict[str, Any]], b: Dict[str, Any]) -> bool:
    if not a:
        return False
    # сравним только ключевые поля
    keys = ["rows", "max_dt"]
    return all(str(a.get(k)) == str(b.get(k)) for k in keys)


# ----------------- subprocess runner -----------------
def run_process(cmd: List[str], env_overrides: Optional[Dict[str, str]] = None) -> int:
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items() if v is not None})

    # для наглядности
    logp("[CWD]", str(BASE_DIR))
    logp("[CMD]", " ".join(cmd))

    p = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),           # рабочая папка Parcer
        env=env,
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
    return int(rc)



# ----------------- atomic index swap -----------------
def atomic_build_with_tmp(build_cmd: List[str], target_dir: Path, env_var_name: str) -> bool:
    """
    Запускаем билд в tmp-директорию, потом атомарно подменяем target_dir.
    """
    tmp_dir = Path(str(target_dir) + "__tmp")
    bak_dir = Path(str(target_dir) + "__bak")

    # cleanup old tmp
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    env_overrides = {env_var_name: str(tmp_dir)}
    # можно принудительно задать устройство
    forced_dev = (os.getenv("PIPELINE_EMBED_DEVICE") or "").strip()
    if forced_dev:
        env_overrides["EMBED_DEVICE"] = forced_dev

    rc = run_process(build_cmd, env_overrides=env_overrides)
    if rc != 0:
        logp("[ERR] build failed, keep old index:", str(target_dir))
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    # swap
    try:
        if bak_dir.exists():
            shutil.rmtree(bak_dir, ignore_errors=True)
        if target_dir.exists():
            target_dir.rename(bak_dir)
        tmp_dir.rename(target_dir)
        shutil.rmtree(bak_dir, ignore_errors=True)
        logp("[OK] index swapped:", str(target_dir))
        return True
    except Exception as e:
        logp("[ERR] swap failed:", type(e).__name__, e)
        # попытка отката
        try:
            if not target_dir.exists() and bak_dir.exists():
                bak_dir.rename(target_dir)
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False


# ----------------- build schedule -----------------
def in_build_window(now_local: dt.datetime, window: str) -> bool:
    """
    window like "00:00-23:59"
    """
    try:
        a, b = window.split("-", 1)
        a = a.strip()
        b = b.strip()
        t1 = dt.time.fromisoformat(a)
        t2 = dt.time.fromisoformat(b)
        t = now_local.time()
        if t1 <= t2:
            return (t >= t1) and (t <= t2)
        # окно через полночь
        return (t >= t1) or (t <= t2)
    except Exception:
        return False


def main():
    lock_path = BASE_DIR / "run_pipeline.lock"
    fd = acquire_lock(lock_path, ttl_min=int(os.getenv("PIPELINE_LOCK_TTL_MIN", "180")))

    try:
        tz_name = (os.getenv("PIPELINE_TZ") or os.getenv("BOT_TZ") or "Europe/Moscow").strip()
        tz = ZoneInfo(tz_name)
        now_local = dt.datetime.now(tz)

        logp("=" * 80)
        logp("[START] now=", now_local.isoformat(), "tz=", tz_name)
        logp("[PY]", sys.executable)

        # 1) INGEST
        ingest_script = resolve_script(os.getenv("PIPELINE_INGEST_SCRIPT") or "ingest_telegram.py")
        ingest_args = (os.getenv("PIPELINE_INGEST_ARGS") or "--with-comments --comments-backfill=200 --batch=300").strip()

        cmd_ingest = [sys.executable, "-u", str(ingest_script)] + ingest_args.split()
        rc = run_process(cmd_ingest)

        if rc != 0:
            logp("[WARN] ingest returned non-zero:", rc, "-> continue (indexes may be skipped)")
        else:
            logp("[OK] ingest done")

        # 2) Решаем, строить ли индексы сейчас
        build_window = (os.getenv("PIPELINE_BUILD_WINDOW") or "00:00-23:59").strip()
        do_build_now = in_build_window(now_local, build_window)

        # once per day
        pipeline_state_path = BASE_DIR / "pipeline_state.json"
        pst = load_json(pipeline_state_path) or {}
        last_build_date = pst.get("last_build_date")  # "YYYY-MM-DD"
        today = now_local.date().isoformat()

        if not do_build_now:
            logp("[SKIP] build window:", build_window, "(now is outside)")
            return

        if last_build_date == today and (os.getenv("PIPELINE_BUILD_EVERY_TIME") or "").strip() != "1":
            logp("[SKIP] already built today:", today)
            return

        logp("[BUILD] window ok:", build_window)

        engine = get_engine()

        # 3) THREADS index
        threads_script = resolve_script((os.getenv("PIPELINE_BUILD_THREADS_SCRIPT") or "build_faiss_threads.py"))
        threads_dir_s = (os.getenv("FAISS_THREADS_DIR") or str(BASE_DIR / "faiss_threads_store")).strip()
        threads_dir = Path(threads_dir_s)
        threads_state_path = threads_dir / "input_state.json"

        # (опционально) refresh matviews перед threads-индексом
        if (os.getenv("PIPELINE_REFRESH_THREAD_MV") or "1").strip() == "1":
            try:
                refresh_matview(engine, "public.thread_docs_channel")
                refresh_matview(engine, "public.thread_docs_chat_reply")
            except Exception as e:
                logp("[WARN] refresh matviews failed:", type(e).__name__, e)

        cur_threads_state = get_threads_input_state(engine)
        prev_threads_state = load_json(threads_state_path)

        if same_state(prev_threads_state, cur_threads_state) and (os.getenv("PIPELINE_FORCE_REBUILD") or "").strip() != "1":
            logp("[SKIP] threads index unchanged:", cur_threads_state)
        else:
            logp("[DO] threads index build, state:", cur_threads_state)
            ok = atomic_build_with_tmp(
                build_cmd=[sys.executable, threads_script],
                target_dir=threads_dir,
                env_var_name="FAISS_THREADS_DIR",
            )
            if ok:
                save_json(threads_state_path, cur_threads_state)

        # 4) MICRO index
        micro_script = resolve_script((os.getenv("PIPELINE_BUILD_MICRO_SCRIPT") or "build_faiss_micro.py"))
        micro_dir_s = (os.getenv("FAISS_MICRO_DIR") or str(BASE_DIR / "faiss_micro")).strip()
        micro_dir = Path(micro_dir_s)
        micro_state_path = micro_dir / "input_state.json"

        cur_micro_state = get_micro_input_state(engine)
        prev_micro_state = load_json(micro_state_path)

        if same_state(prev_micro_state, cur_micro_state) and (os.getenv("PIPELINE_FORCE_REBUILD") or "").strip() != "1":
            logp("[SKIP] micro index unchanged:", cur_micro_state)
        else:
            logp("[DO] micro index build, state:", cur_micro_state)
            ok = atomic_build_with_tmp(
                build_cmd=[sys.executable, micro_script],
                target_dir=micro_dir,
                env_var_name="FAISS_MICRO_DIR",
            )
            if ok:
                save_json(micro_state_path, cur_micro_state)

        # 5) mark built today
        pst["last_build_date"] = today
        pst["last_build_ts"] = now_local.isoformat()
        save_json(pipeline_state_path, pst)

        logp("[DONE] pipeline OK")
        touch_cache_bust(logp, BASE_DIR)

    finally:
        release_lock(lock_path, fd)




def touch_cache_bust(logp, base_dir: Path):
    p = os.getenv("CACHE_BUST_FILE", str(base_dir / ".data_updated"))
    try:
        pp = Path(p)
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.touch()
        logp("[BUST] touched", str(pp))
    except Exception as e:
        logp("[BUST] touch failed:", type(e).__name__, e)


if __name__ == "__main__":
    main()
