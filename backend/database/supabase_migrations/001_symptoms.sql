-- Nirvor migration 001: verified symptom vector store for anti-hallucination grounding.
--
-- HOW TO APPLY: open your Supabase project -> SQL Editor -> paste this whole
-- file -> Run. Then restart the backend so it picks up the live pgvector path.
-- Until this runs, the backend keeps working via its local cosine fallback.
--
-- DESIGN NOTES (do not "simplify" these away):
-- * `dialect_phrase` is the primary key because the backend upserts on it.
-- * `embedding` is vector(512) to match ClinicalMathEngine.VECTOR_DIM.
-- * `similarity` returned by the RPC is a COSINE DISTANCE (0 = identical).
--   The backend converts it with 1 - distance, so keep the alias as-is.
-- * `match_threshold` 0.0 means "accept all verified rows, best first".
-- * Only verified = true rows are ever retrieved for grounding. The curated
--   seed lexicon auto-upserts as verified; new bulk imports must arrive with
--   verified = false until a human reviews them (see import_verified_lexicon.py).

create extension if not exists vector;

create table if not exists public.symptoms (
  dialect_phrase text primary key,
  clinical_term text not null,
  icd10_code text not null default '',
  dialect_region text not null default '',
  embedding vector(512) not null,
  verified boolean not null default true,
  source_file text,
  verified_by text,
  verified_at timestamptz not null default now()
);

-- Exact scan is fine at this scale; the index matters once rows grow past ~10k.
create index if not exists symptoms_embedding_idx
  on public.symptoms using ivfflat (embedding vector_cosine_ops)
  with (lists = 10);

alter table public.symptoms enable row level security;
-- Backend connects with the service key, which bypasses RLS. No public
-- policies are created on purpose: patient-adjacent data stays server-only.

create or replace function public.match_symptoms(
  query_embedding vector(512),
  match_threshold float default 0.0,
  match_count int default 2
)
returns table (
  dialect_region text,
  dialect_phrase text,
  clinical_term text,
  icd10_code text,
  similarity float
)
language sql stable
as $$
  select
    s.dialect_region,
    s.dialect_phrase,
    s.clinical_term,
    s.icd10_code,
    (s.embedding <=> query_embedding)::float as similarity
  from public.symptoms s
  where s.verified = true
    and (1 - (s.embedding <=> query_embedding)) >= match_threshold
  order by s.embedding <=> query_embedding
  limit match_count;
$$;
