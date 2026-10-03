# refresh_mviews.py
import os
from sqlalchemy import create_engine, text

PG_DSN = os.getenv("PG_DSN")

engine = create_engine(PG_DSN, isolation_level="AUTOCOMMIT")

with engine.connect() as conn:
    conn.execute(text("REFRESH MATERIALIZED VIEW public.thread_docs_channel;"))
    conn.execute(text("REFRESH MATERIALIZED VIEW public.thread_docs_chat_reply;"))
print("OK: refreshed")

