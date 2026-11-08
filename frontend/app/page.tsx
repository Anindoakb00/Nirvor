"use client";

import { useEffect, useState } from "react";
import {
  API_BASE,
  getSystemStatus,
  isRouterCrash,
  postCombined,
  postPrescription,
  postVoice,
  postXray,
  type ClinicalAnalysisResponse,
  type CombinedIntakeResponse,
  type CompleteIntakeResponse,
  type SystemStatus,
} from "@/lib/api";
import {
  ClinicalResult,
  CombinedResult,
  CompleteResult,
  RawJson,
} from "@/components/results";

type TabKey = "voice" | "prescription" | "xray" | "combined";

interface FieldDef {
  key: string;
  label: string;
  accept: string;
  required: boolean;
}

const TABS: { key: TabKey; label: string; fields: FieldDef[] }[] = [
  {
    key: "voice",
    label: "Voice",
    fields: [
      { key: "audio", label: "Voice note", accept: "audio/*,.m4a", required: true },
    ],
  },
  {
    key: "prescription",
    label: "Prescription",
    fields: [
      { key: "prescription", label: "Prescription image", accept: "image/*", required: true },
    ],
  },
  {
    key: "xray",
    label: "X-ray",
    fields: [
      { key: "xray_scan", label: "X-ray scan", accept: "image/*", required: true },
    ],
  },
  {
    key: "combined",
    label: "Combined",
    fields: [
      { key: "prescription", label: "Prescription image", accept: "image/*", required: false },
      { key: "audio", label: "Voice note", accept: "audio/*,.m4a", required: false },
      { key: "xray_scan", label: "X-ray scan", accept: "image/*", required: false },
    ],
  },
];

function hasDoctorView(body: unknown): body is ClinicalAnalysisResponse {
  return (
    typeof body === "object" &&
    body !== null &&
    "doctor_view" in body &&
    "patient_view" in body
  );
}

export default function Home() {
  const [tab, setTab] = useState<TabKey>("prescription");
  const [status, setStatus] = useState<SystemStatus | "offline" | null>(null);
  const [files, setFiles] = useState<Record<string, File | null>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<Partial<Record<TabKey, unknown>>>({});

  useEffect(() => {
    getSystemStatus()
      .then(setStatus)
      .catch(() => setStatus("offline"));
  }, []);

  const active = TABS.find((t) => t.key === tab)!;

  function pickFile(key: string, file: File | null) {
    setFiles((prev) => ({ ...prev, [`${tab}:${key}`]: file }));
  }

  function getFile(key: string): File | null {
    return files[`${tab}:${key}`] ?? null;
  }

  async function submit() {
    setError(null);
    setBusy(true);
    try {
      let body: unknown;
      if (tab === "voice") {
        const audio = getFile("audio");
        if (!audio) throw new Error("Choose a voice file first.");
        body = await postVoice(audio);
      } else if (tab === "prescription") {
        const prescription = getFile("prescription");
        if (!prescription) throw new Error("Choose a prescription image first.");
        body = await postPrescription(prescription);
      } else if (tab === "xray") {
        const xray_scan = getFile("xray_scan");
        if (!xray_scan) throw new Error("Choose an X-ray image first.");
        body = await postXray(xray_scan);
      } else {
        const prescription = getFile("prescription");
        const audio = getFile("audio");
        const xray_scan = getFile("xray_scan");
        if (!prescription && !audio && !xray_scan)
          throw new Error("Provide at least one file.");
        body = await postCombined({ prescription, audio, xray_scan });
      }
      if (isRouterCrash(body)) throw new Error(body.error);
      setResults((prev) => ({ ...prev, [tab]: body }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed.");
    } finally {
      setBusy(false);
    }
  }

  const result = results[tab];

  return (
    <div className="min-h-full flex flex-col">
      <header className="bg-brand-800 text-white">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold">Nirvor Clinical Intake</h1>
            <p className="text-[13px] text-teal-100">
              Voice, prescription &amp; scan analysis
            </p>
          </div>
          {status === null ? (
            <span className="text-sm text-teal-100">Checking backend…</span>
          ) : status === "offline" ? (
            <span className="text-sm font-medium bg-red-700 rounded-full px-3 py-1">
              Backend offline
            </span>
          ) : (
            <span className="text-sm font-medium bg-brand-900 rounded-full px-3 py-1">
              Backend v{status.version} · Meta{" "}
              {status.meta_configured ? "on" : "off"} · Modal{" "}
              {status.modal_configured ? "on" : "off"}
            </span>
          )}
        </div>
      </header>

      <nav className="bg-white border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-6 flex gap-1">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => {
                setTab(t.key);
                setError(null);
              }}
              className={`px-4 py-3 text-sm font-semibold border-b-2 -mb-px ${
                tab === t.key
                  ? "border-brand-700 text-brand-800"
                  : "border-transparent text-slate-500 hover:text-slate-800"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-6 py-6 w-full grid gap-6 lg:grid-cols-[360px_1fr] items-start">
        <section className="bg-white border border-slate-200 rounded-lg p-5">
          <h2 className="text-base font-bold text-slate-900">
            {active.label} intake
          </h2>
          <p className="text-[13px] text-slate-500 mt-1">
            {tab === "combined"
              ? "Provide at least one file; more modalities means richer fusion."
              : "Choose a file, then run the analysis."}
          </p>
          <div className="mt-4 space-y-4">
            {active.fields.map((f) => (
              <div key={f.key}>
                <label className="block text-sm font-medium text-slate-700">
                  {f.label}
                  {!f.required && (
                    <span className="text-slate-400 font-normal"> (optional)</span>
                  )}
                </label>
                <input
                  type="file"
                  accept={f.accept}
                  onChange={(e) => pickFile(f.key, e.target.files?.[0] ?? null)}
                  className="mt-1.5 block w-full text-sm text-slate-700 file:mr-3 file:py-2 file:px-3 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-brand-800 file:text-white hover:file:bg-brand-700"
                />
                {getFile(f.key) && (
                  <p className="mt-1 text-[13px] text-slate-500 truncate">
                    {getFile(f.key)!.name}
                  </p>
                )}
              </div>
            ))}
          </div>
          <button
            onClick={submit}
            disabled={busy}
            className="mt-5 w-full rounded-md bg-brand-800 px-4 py-2.5 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
          >
            {busy ? "Analyzing…" : "Run analysis"}
          </button>
          {status === "offline" && (
            <p className="mt-3 text-[13px] text-red-700">
              Backend not reachable at {API_BASE}. Start it with{" "}
              <span className="font-mono">
                python -m uvicorn main:app --port 8000
              </span>{" "}
              from the backend folder.
            </p>
          )}
        </section>

        <section className="min-w-0">
          {error && (
            <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-900">
              <span className="font-semibold">Error: </span>
              {error}
            </div>
          )}
          {!error && !result && (
            <div className="rounded-lg border border-dashed border-slate-300 bg-white px-6 py-16 text-center">
              <p className="text-sm text-slate-500">
                Results appear here after you run an analysis.
              </p>
            </div>
          )}
          {!error && result !== undefined && (
            <>
              {tab === "voice" &&
                (hasDoctorView(result) ? (
                  <ClinicalResult data={result} />
                ) : (
                  <RawJson data={result} />
                ))}
              {(tab === "prescription" || tab === "xray") && (
                <CompleteResult data={result as CompleteIntakeResponse} />
              )}
              {tab === "combined" && (
                <CombinedResult data={result as CombinedIntakeResponse} />
              )}
            </>
          )}
        </section>
      </main>

      <footer className="mt-auto border-t border-slate-200 bg-white">
        <div className="max-w-6xl mx-auto px-6 py-3 text-[13px] text-slate-500 font-mono">
          API: {API_BASE}
        </div>
      </footer>
    </div>
  );
}
