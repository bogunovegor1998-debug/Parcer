-- Источники (каналы)
CREATE TABLE IF NOT EXISTS tg_sources (
  id              BIGSERIAL PRIMARY KEY,
  source_type     TEXT NOT NULL DEFAULT 'channel',
  username        TEXT,            -- например "mosmetro"
  invite_link     TEXT,            -- если хочешь хранить
  title           TEXT,
  peer_id         BIGINT,          -- channel_id (реальный)
  access_hash     BIGINT,
  is_active       BOOLEAN NOT NULL DEFAULT TRUE,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (username)
);

-- Состояние обхода по каждому источнику (курсор)
CREATE TABLE IF NOT EXISTS tg_source_state (
  source_id       BIGINT PRIMARY KEY REFERENCES tg_sources(id) ON DELETE CASCADE,
  last_msg_id     INTEGER NOT NULL DEFAULT 0,
  last_scan_at    TIMESTAMPTZ
);

-- Посты каналов (и любые “сообщения верхнего уровня”)
CREATE TABLE IF NOT EXISTS tg_posts (
  id              BIGSERIAL PRIMARY KEY,
  source_id       BIGINT NOT NULL REFERENCES tg_sources(id) ON DELETE CASCADE,
  chat_id         BIGINT NOT NULL,         -- peer id (channel)
  msg_id          INTEGER NOT NULL,        -- message id внутри канала
  msg_date        TIMESTAMPTZ,
  sender_id       BIGINT,
  text            TEXT,
  views           INTEGER,
  forwards        INTEGER,
  replies_count   INTEGER,
  permalink       TEXT,
  raw_json        JSONB,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (chat_id, msg_id)
);

CREATE INDEX IF NOT EXISTS idx_tg_posts_source_date ON tg_posts(source_id, msg_date);
CREATE INDEX IF NOT EXISTS idx_tg_posts_text_gin ON tg_posts USING GIN (to_tsvector('russian', coalesce(text,'')));

-- Комментарии (из обсуждений; важно: chat_id может быть уже group_id)
CREATE TABLE IF NOT EXISTS tg_comments (
  id              BIGSERIAL PRIMARY KEY,
  source_id       BIGINT NOT NULL REFERENCES tg_sources(id) ON DELETE CASCADE,
  root_chat_id    BIGINT NOT NULL,         -- канал
  root_msg_id     INTEGER NOT NULL,        -- id поста в канале
  chat_id         BIGINT NOT NULL,         -- где реально лежит коммент (обычно discussion group)
  msg_id          INTEGER NOT NULL,
  msg_date        TIMESTAMPTZ,
  sender_id       BIGINT,
  text            TEXT,
  permalink       TEXT,
  raw_json        JSONB,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (chat_id, msg_id)
);

CREATE INDEX IF NOT EXISTS idx_tg_comments_root ON tg_comments(root_chat_id, root_msg_id);
CREATE INDEX IF NOT EXISTS idx_tg_comments_text_gin ON tg_comments USING GIN (to_tsvector('russian', coalesce(text,'')));

-- Курсор по комментариям на тред (чтобы не перечитывать всё)
CREATE TABLE IF NOT EXISTS tg_thread_state (
  source_id       BIGINT NOT NULL REFERENCES tg_sources(id) ON DELETE CASCADE,
  root_chat_id    BIGINT NOT NULL,
  root_msg_id     INTEGER NOT NULL,
  last_comment_id INTEGER NOT NULL DEFAULT 0,
  last_scan_at    TIMESTAMPTZ,
  PRIMARY KEY (source_id, root_chat_id, root_msg_id)
);
