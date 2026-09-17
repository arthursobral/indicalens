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

create index if not exists documents_embedding_idx
    on documents using ivfflat (embedding vector_cosine_ops) with (lists = 100);
