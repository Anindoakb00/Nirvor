-- Nirvor migration 002: prescription referral vector store.
--
-- HOW TO APPLY: open your Supabase project -> SQL Editor -> paste this whole
-- file -> Run. Then run the bulk import:
--   cd backend && python database/import_prescription_referrals.py
-- Until this runs, the prescription branch keeps working without referrals
-- (referral_telemetry is null) - nothing breaks.
--
-- DESIGN NOTES (do not "simplify" these away):
-- * `source_file` is the primary key because the importer upserts on it.
-- * `embedding` is vector(512) to match ClinicalMathEngine.VECTOR_DIM, built
--   from the medication blob via ClinicalMathEngine.build_medication_blob.
-- * `similarity` returned by the RPC is a COSINE DISTANCE (0 = identical).
--   The backend converts it with 1 - distance, so keep the alias as-is.
-- * Unlike match_symptoms, match_prescriptions does NOT filter verified rows:
--   referrals are explicitly labeled "similar past case" telemetry, and the
--   `verified` flag is returned so the backend can say which is which.
-- * Bulk imports arrive with verified = false. A row becomes verified = true
--   only after a human reviews its extracted medications.

create extension if not exists vector;

create table if not exists public.prescription_referrals (
  source_file text primary key,
  medication_blob text not null,
  med_count int not null default 0,
  embedding vector(512) not null,
  verified boolean not null default false,
  provider_used text not null default '',
  verified_by text,
  verified_at timestamptz,
  created_at timestamptz not null default now()
);

-- Exact scan is fine at this scale; the index matters once rows grow past ~10k.
create index if not exists prescription_referrals_embedding_idx
  on public.prescription_referrals using ivfflat (embedding vector_cosine_ops)
  with (lists = 10);

alter table public.prescription_referrals enable row level security;
-- Backend connects with the service key, which bypasses RLS. No public
-- policies are created on purpose: patient-adjacent data stays server-only.

create or replace function public.match_prescriptions(
  query_embedding vector(512),
  match_threshold float default 0.0,
  match_count int default 3
)
returns table (
  source_file text,
  medication_blob text,
  med_count int,
  verified boolean,
  similarity float
)
language sql stable
as $$
  select
    r.source_file,
    r.medication_blob,
    r.med_count,
    r.verified,
    (r.embedding <=> query_embedding)::float as similarity
  from public.prescription_referrals r
  where (1 - (r.embedding <=> query_embedding)) >= match_threshold
  order by r.embedding <=> query_embedding
  limit match_count;
$$;
