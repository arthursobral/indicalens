-- Run once against the Supabase Postgres database (SQL editor or psql).
create extension if not exists vector;

create table if not exists documents (
    id bigserial primary key,
    source text not null,
    series text not null,
    period text not null,
    content text not null,
    citation text not null,
    embedding vector(384) not null,
    created_at timestamptz not null default now(),
    unique (series, period)
);

-- Populated by src/indicator_analyst.py (zero-shot classification), null
-- until that step runs. Only meaningful for source = 'IBGE-PNAD-COMENTARIOS'.
alter table documents add column if not exists theme text;

-- ponytail: no vector index yet — at a few hundred/thousand rows a sequential
-- scan for ORDER BY embedding <=> ... is already fast enough, and ivfflat
-- clusters are trained from whatever data exists at CREATE INDEX time, so
-- building one before the table has real data leaves it degenerate (returns
-- 0 rows on ORDER BY forever, since the planner prefers the broken index
-- over a seq scan). Add an ivfflat/hnsw index once ingestion is large enough
-- for it to matter, and always create it AFTER loading data, never before.

-- No policies defined on purpose: the app connects with the postgres role via
-- DATABASE_URL, which bypasses RLS. This just default-denies the Supabase
-- anon/authenticated PostgREST roles, in case the auto-API is ever enabled
-- for this table.
alter table documents enable row level security;
