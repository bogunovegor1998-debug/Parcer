-- Индексы (полезно)
CREATE INDEX IF NOT EXISTS idx_tg_chat_messages_chat_msg
  ON tg_chat_messages(chat_id, msg_id);

CREATE INDEX IF NOT EXISTS idx_tg_chat_messages_chat_parent
  ON tg_chat_messages(chat_id, parent_msg_id);

-- ------------------------------------------------------------
-- A) CHANNEL THREAD DOCS: post + comments
-- ------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS thread_docs_channel;

CREATE MATERIALIZED VIEW thread_docs_channel AS
SELECT
  ('chan:' || fr.root_chat_id::text || ':' || fr.root_msg_id::text) AS thread_id,
  fr.root_chat_id,
  fr.root_msg_id,
  MIN(fr.msg_date) AS start_dt,
  MAX(fr.msg_date) AS last_dt,

  MAX(fr.source_username) AS source_username,
  MAX(fr.source_title)    AS source_title,
  MAX(fr.root_permalink)  AS root_permalink,

  MAX(fr.root_text) AS root_text,

  COUNT(*) FILTER (WHERE fr.kind = 'comment') AS n_comments,

  -- ✅ ВАЖНО: ORDER BY внутри STRING_AGG
  STRING_AGG(
    CASE WHEN fr.kind = 'comment' THEN COALESCE(fr.text, '') ELSE NULL END,
    E'\n'
    ORDER BY fr.msg_date
  ) FILTER (WHERE fr.kind = 'comment') AS comments_text

FROM feedback_raw_v3 fr
WHERE fr.kind IN ('post','comment')
  AND fr.root_chat_id IS NOT NULL
  AND fr.root_msg_id  IS NOT NULL
GROUP BY fr.root_chat_id, fr.root_msg_id;

CREATE UNIQUE INDEX IF NOT EXISTS ux_thread_docs_channel_thread_id
  ON thread_docs_channel(thread_id);

CREATE INDEX IF NOT EXISTS ix_thread_docs_channel_root
  ON thread_docs_channel(root_chat_id, root_msg_id);

-- ------------------------------------------------------------
-- B) CHAT REPLY THREAD DOCS: root + replies (recursive parent chain)
-- ------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS thread_docs_chat_reply;

CREATE MATERIALIZED VIEW thread_docs_chat_reply AS
WITH RECURSIVE up AS (
  SELECT
    m.chat_id,
    m.msg_id        AS origin_msg_id,
    m.msg_id        AS cur_msg_id,
    m.parent_msg_id AS parent_msg_id,
    0              AS depth
  FROM tg_chat_messages m
  WHERE m.parent_msg_id IS NOT NULL

  UNION ALL

  SELECT
    u.chat_id,
    u.origin_msg_id,
    p.msg_id        AS cur_msg_id,
    p.parent_msg_id AS parent_msg_id,
    u.depth + 1     AS depth
  FROM up u
  JOIN tg_chat_messages p
    ON p.chat_id = u.chat_id
   AND p.msg_id  = u.parent_msg_id
  WHERE u.parent_msg_id IS NOT NULL
    AND u.depth < 50
),
roots_from_chain AS (
  SELECT DISTINCT ON (chat_id, origin_msg_id)
    chat_id,
    origin_msg_id,
    cur_msg_id AS root_msg_id
  FROM up
  ORDER BY chat_id, origin_msg_id, depth DESC
),
roots_all AS (
  SELECT
    m.chat_id,
    m.msg_id AS origin_msg_id,
    m.msg_id AS root_msg_id
  FROM tg_chat_messages m
  WHERE m.parent_msg_id IS NULL

  UNION ALL

  SELECT
    r.chat_id,
    r.origin_msg_id,
    r.root_msg_id
  FROM roots_from_chain r
),
msg_with_root AS (
  SELECT
    m.*,
    r.root_msg_id
  FROM tg_chat_messages m
  JOIN roots_all r
    ON r.chat_id = m.chat_id
   AND r.origin_msg_id = m.msg_id
)
SELECT
  ('chat:' || m.chat_id::text || ':' || m.root_msg_id::text) AS thread_id,
  m.chat_id,
  m.root_msg_id,
  MIN(m.msg_date) AS start_dt,
  MAX(m.msg_date) AS last_dt,

  MAX(m.raw_json->>'username') AS source_username,

  MAX(CASE WHEN m.msg_id = m.root_msg_id THEN m.text END)      AS root_text,
  MAX(CASE WHEN m.msg_id = m.root_msg_id THEN m.permalink END) AS root_permalink,

  COUNT(*) AS n_messages,

  -- ✅ ВАЖНО: ORDER BY внутри STRING_AGG
  STRING_AGG(
    CASE WHEN m.msg_id <> m.root_msg_id THEN COALESCE(m.text, '') ELSE NULL END,
    E'\n'
    ORDER BY m.msg_date
  ) FILTER (WHERE m.msg_id <> m.root_msg_id) AS replies_text

FROM msg_with_root m
GROUP BY m.chat_id, m.root_msg_id;

CREATE UNIQUE INDEX IF NOT EXISTS ux_thread_docs_chat_reply_thread_id
  ON thread_docs_chat_reply(thread_id);

CREATE INDEX IF NOT EXISTS ix_thread_docs_chat_reply_root
  ON thread_docs_chat_reply(chat_id, root_msg_id);

-- ------------------------------------------------------------
-- общий VIEW (удобно для Python)
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW thread_docs_all AS
SELECT
  thread_id,
  'chan'::text AS doc_kind,
  source_username,
  source_title,
  root_chat_id AS chat_id,
  root_msg_id,
  start_dt,
  last_dt,
  root_text,
  comments_text AS body_text,
  n_comments AS n_items,
  root_permalink
FROM thread_docs_channel

UNION ALL

SELECT
  thread_id,
  'chat'::text AS doc_kind,
  source_username,
  NULL::text AS source_title,
  chat_id,
  root_msg_id,
  start_dt,
  last_dt,
  root_text,
  replies_text AS body_text,
  n_messages AS n_items,
  root_permalink
FROM thread_docs_chat_reply;
