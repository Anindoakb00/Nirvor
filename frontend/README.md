# Nirvor Frontend (v0.0.4)

Next.js + TypeScript + Tailwind dashboard for the Nirvor FastAPI backend.
Upload voice notes, prescription images, and X-ray scans — view grounded
medications, doctor/patient views, drug-safety alerts, the rendered dosage
card, telemetry, and the execution trace.

## Prereqs

- Node.js 20+
- The backend running (from `backend/`):
  `python -m uvicorn main:app --reload --port 8000`

## Setup

```powershell
cd frontend
npm install
copy .env.example .env.local
# optional: point NEXT_PUBLIC_API_URL at your backend host
npm run dev
```

Open http://localhost:3000 in your browser.

## Build

```powershell
npm run build
npm run start
```

## Layout

- `app/page.tsx` — tabbed intake dashboard (voice / prescription / xray /
  combined) + results area
- `components/results.tsx` — result renderers (views, meds, safety, dosage
  card, telemetry, trace, raw JSON)
- `lib/api.ts` — typed FastAPI client (shapes mirror
  `backend/models/schemas.py`)
