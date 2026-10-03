CREATE TABLE IF NOT EXISTS public.stg_tg_posts (
    source_id BIGINT,
    chat_id BIGINT,
    msg_id INTEGER,
    msg_date TIMESTAMPTZ,
    sender_id BIGINT,
    text TEXT,
    views INTEGER,
    forwards INTEGER,
    replies_count INTEGER,
    permalink TEXT,
    raw_json JSONB,
    doc_id TEXT
);

CREATE TABLE IF NOT EXISTS public.stg_tg_comments (
    source_id BIGINT,
    root_chat_id BIGINT,
    root_msg_id INTEGER,
    chat_id BIGINT,
    msg_id INTEGER,
    msg_date TIMESTAMPTZ,
    sender_id BIGINT,
    text TEXT,
    permalink TEXT,
    raw_json JSONB,
    reply_to_msg_id INTEGER,
    doc_id TEXT
);

CREATE TABLE IF NOT EXISTS public.stg_tg_chat_messages (
    source_id BIGINT,
    chat_id BIGINT,
    msg_id INTEGER,
    msg_date TIMESTAMPTZ,
    sender_id BIGINT,
    text TEXT,
    permalink TEXT,
    raw_json JSONB,
    parent_msg_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_stg_tg_posts_chat_msg
    ON public.stg_tg_posts (chat_id, msg_id);

CREATE INDEX IF NOT EXISTS idx_stg_tg_comments_chat_msg
    ON public.stg_tg_comments (chat_id, msg_id);

CREATE INDEX IF NOT EXISTS idx_stg_tg_chat_messages_chat_msg
    ON public.stg_tg_chat_messages (chat_id, msg_id);