import type {
  AgentExecutionTraceItem,
  ClinicalAnalysisResponse,
  ClinicalCoherenceReport,
  CombinedIntakeResponse,
  CompleteIntakeResponse,
  CrossModalFusion,
  GroundedMedication,
  JudgeHudSummary,
  PolypharmacyIntervention,
  ReferralTelemetry,
} from "@/lib/api";

/* ---------- small helpers ---------- */

function fmt(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number")
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="bg-white border border-slate-200 rounded-lg p-5">
      <h3 className="text-sm font-semibold tracking-wide text-brand-800 uppercase mb-3">
        {title}
      </h3>
      {children}
    </section>
  );
}

function DictTable({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data);
  if (entries.length === 0)
    return <p className="text-sm text-slate-500">No data.</p>;
  return (
    <dl className="text-sm divide-y divide-slate-100">
      {entries.map(([key, value]) => (
        <div key={key} className="py-1.5 grid grid-cols-[minmax(0,220px)_1fr] gap-3">
          <dt className="font-medium text-slate-700 break-words">{key}</dt>
          <dd className="text-slate-800 break-words font-mono text-[13px]">
            {fmt(value)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/* ---------- shared blocks ---------- */

export function StatusBanner({
  status,
  errorDetails,
}: {
  status: string;
  errorDetails?: string[] | null;
}) {
  const ok = status === "success";
  return (
    <div
      className={`rounded-lg px-4 py-3 border text-sm ${
        ok
          ? "bg-green-50 border-green-200 text-green-900"
          : "bg-amber-50 border-amber-200 text-amber-900"
      }`}
    >
      <span className="font-semibold">Status: {status}</span>
      {errorDetails && errorDetails.length > 0 && (
        <ul className="mt-2 list-disc pl-5 space-y-1">
          {errorDetails.map((e, i) => (
            <li key={i} className="font-mono text-[13px]">
              {e}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ViewsPanel({ data }: { data: ClinicalAnalysisResponse }) {
  const { doctor_view: doc, patient_view: pat } = data;
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Section title="Doctor view">
        <p className="text-sm text-slate-800 whitespace-pre-wrap">
          {doc.clinical_summary}
        </p>
        <div className="mt-3 text-sm">
          <p className="font-medium text-slate-700">Triage question</p>
          <p className="text-slate-800">{doc.active_triage_question}</p>
        </div>
        {doc.grounded_medications.length > 0 && (
          <div className="mt-3 text-sm">
            <p className="font-medium text-slate-700">Grounded medications</p>
            <ul className="list-disc pl-5 text-slate-800">
              {doc.grounded_medications.map((m, i) => (
                <li key={i}>{m}</li>
              ))}
            </ul>
          </div>
        )}
        <div className="mt-3 rounded-md bg-slate-50 border border-slate-200 p-3 text-sm">
          <p className="font-medium text-slate-700">Verification telemetry</p>
          <p className="text-slate-800">
            Match score:{" "}
            {fmt(doc.verification_telemetry.primary_vector_match_score)} ·
            Hallucination risk:{" "}
            <span className="font-semibold">
              {doc.verification_telemetry.hallucination_risk_status}
            </span>
          </p>
          <p className="text-slate-600 text-[13px] mt-1">
            {doc.verification_telemetry.grounding_source_text}
          </p>
        </div>
      </Section>
      <Section title="Patient view">
        <p className="text-sm text-slate-800 whitespace-pre-wrap">
          {pat.layman_explanation}
        </p>
        <div className="mt-3 text-sm">
          <p className="font-medium text-slate-700">Next steps</p>
          <p className="text-slate-800 whitespace-pre-wrap">{pat.next_steps}</p>
        </div>
      </Section>
    </div>
  );
}

function medField(med: GroundedMedication, key: string): string {
  return fmt(med[key]);
}

export function MedsTable({ meds }: { meds: GroundedMedication[] }) {
  if (meds.length === 0)
    return (
      <Section title="Grounded medications">
        <p className="text-sm text-slate-500">None.</p>
      </Section>
    );
  return (
    <Section title="Grounded medications">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-600 border-b border-slate-200">
              <th className="py-2 pr-4 font-semibold">Raw text</th>
              <th className="py-2 pr-4 font-semibold">Dosage</th>
              <th className="py-2 pr-4 font-semibold">Frequency</th>
              <th className="py-2 font-semibold">Scores</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {meds.map((med, i) => (
              <tr key={i}>
                <td className="py-2 pr-4 text-slate-800">
                  {medField(med, "raw_text")}
                </td>
                <td className="py-2 pr-4 font-mono text-[13px]">
                  {medField(med, "dosage")}
                </td>
                <td className="py-2 pr-4 font-mono text-[13px]">
                  {medField(med, "frequency")}
                </td>
                <td className="py-2 font-mono text-[13px] text-slate-700">
                  {["bayesian_posterior_confidence", "cos_voice", "posterior", "confidence", "composite_energy", "entropy", "shannon_entropy_bits"]
                    .filter((k) => med[k] !== undefined && med[k] !== null)
                    .map((k) => `${k}=${fmt(med[k])}`)
                    .join("  ") || "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

const RISK_STYLES: Record<string, string> = {
  CRITICAL: "bg-red-50 border-red-300 text-red-900",
  HIGH: "bg-orange-50 border-orange-300 text-orange-900",
  MODERATE: "bg-amber-50 border-amber-300 text-amber-900",
  LOW: "bg-green-50 border-green-200 text-green-900",
};

export function SafetyPanel({ data }: { data: PolypharmacyIntervention }) {
  return (
    <section
      className={`rounded-lg p-5 border ${
        RISK_STYLES[data.risk_level] ?? "bg-slate-50 border-slate-200"
      }`}
    >
      <h3 className="text-sm font-semibold tracking-wide uppercase mb-2">
        Drug safety — {data.risk_level}
      </h3>
      <p className="text-sm">{data.clinical_mechanism}</p>
      <p className="text-sm mt-2">
        <span className="font-semibold">Action: </span>
        {data.recommended_action}
      </p>
      <p className="text-sm mt-1">{data.bangla_patient_warning}</p>
      {(data.verified_medications_assessed !== undefined ||
        data.unverified_medications_skipped !== undefined) && (
        <p className="text-[13px] mt-2 opacity-80">
          Verified assessed: {data.verified_medications_assessed ?? 0} ·
          Unverified skipped: {data.unverified_medications_skipped ?? 0}
        </p>
      )}
    </section>
  );
}

export function DosageCard({ src }: { src: string }) {
  return (
    <Section title="Patient dosage card">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt="Patient dosage card"
        className="max-w-full h-auto border border-slate-200 rounded-md"
      />
      <a
        href={src}
        download="dosage-card.png"
        className="inline-block mt-3 text-sm font-medium text-brand-700 underline underline-offset-2"
      >
        Download PNG
      </a>
    </Section>
  );
}

export function HudGrid({
  hud,
  coherence,
  referral,
}: {
  hud: JudgeHudSummary;
  coherence?: ClinicalCoherenceReport | null;
  referral?: ReferralTelemetry | null;
}) {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Section title="Run summary">
        <dl className="text-sm space-y-1.5">
          <div className="flex justify-between gap-3">
            <dt className="text-slate-600">Latency</dt>
            <dd className="font-mono">{fmt(hud.total_latency_ms)} ms</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-slate-600">Mean entropy</dt>
            <dd className="font-mono">{fmt(hud.mean_shannon_entropy)} bits</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-slate-600">Hallucinations caught</dt>
            <dd className="font-mono">{fmt(hud.hallucinations_intercepted)}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-slate-600">Verification flags</dt>
            <dd className="font-mono">{fmt(hud.active_verification_flags)}</dd>
          </div>
        </dl>
      </Section>
      <Section title="Coherence">
        {!coherence ? (
          <p className="text-sm text-slate-500">Not reported.</p>
        ) : (
          <div className="text-sm space-y-1.5">
            <p>
              <span className="font-semibold">
                {coherence.is_coherent ? "Coherent" : "Needs review"}
              </span>{" "}
              <span className="font-mono">
                ({fmt(coherence.coherence_score)})
              </span>
            </p>
            {coherence.matched_indications.length > 0 && (
              <p className="text-slate-700">
                Matched: {coherence.matched_indications.join(", ")}
              </p>
            )}
            {coherence.contradiction_flags.length > 0 && (
              <ul className="list-disc pl-5 text-slate-800">
                {coherence.contradiction_flags.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            )}
          </div>
        )}
      </Section>
      <Section title="Referral match">
        {!referral ? (
          <p className="text-sm text-slate-500">Not reported.</p>
        ) : (
          <dl className="text-sm space-y-1.5">
            <div className="flex justify-between gap-3">
              <dt className="text-slate-600">Score</dt>
              <dd className="font-mono">
                {fmt(referral.top_referral_match_score)}
              </dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-slate-600">Confidence</dt>
              <dd className="font-medium">{referral.referral_confidence}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt className="text-slate-600">Verified</dt>
              <dd>{referral.referral_verified ? "Yes" : "No"}</dd>
            </div>
            <p className="text-[13px] text-slate-600 break-words">
              {referral.referral_source_file} — {referral.referral_medications}
            </p>
          </dl>
        )}
      </Section>
    </div>
  );
}

export function FusionPanel({ data }: { data: CrossModalFusion }) {
  return (
    <Section title="Cross-modal fusion">
      <div className="text-sm grid gap-4 md:grid-cols-2">
        <dl className="space-y-1.5">
          <div className="flex justify-between gap-3">
            <dt className="text-slate-600">Fused confidence</dt>
            <dd className="font-mono">{fmt(data.fused_confidence)}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-slate-600">Overall risk</dt>
            <dd className="font-semibold">{data.overall_risk}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-slate-600">Coherent</dt>
            <dd>{data.is_coherent ? "Yes" : "No"}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-slate-600">Voice confidence</dt>
            <dd className="font-mono">{fmt(data.voice_confidence)}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-slate-600">Prescription confidence</dt>
            <dd className="font-mono">{fmt(data.prescription_confidence)}</dd>
          </div>
        </dl>
        <div>
          <p className="text-slate-600">
            Present: {data.modalities_present.join(", ") || "—"}
          </p>
          <p className="text-slate-600 mt-1">
            Succeeded: {data.modalities_succeeded.join(", ") || "—"}
          </p>
          {data.agreement_notes.length > 0 && (
            <ul className="list-disc pl-5 mt-2 text-slate-800">
              {data.agreement_notes.map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </Section>
  );
}

export function TraceTable({ items }: { items: AgentExecutionTraceItem[] }) {
  if (items.length === 0) return null;
  return (
    <Section title="Execution trace">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-600 border-b border-slate-200">
              <th className="py-2 pr-4 font-semibold">Step</th>
              <th className="py-2 pr-4 font-semibold">Status</th>
              <th className="py-2 pr-4 font-semibold">Provider</th>
              <th className="py-2 font-semibold">Latency</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 font-mono text-[13px]">
            {items.map((t, i) => (
              <tr key={i}>
                <td className="py-2 pr-4">{t.step_name}</td>
                <td className="py-2 pr-4">{t.status}</td>
                <td className="py-2 pr-4">{t.provider_used}</td>
                <td className="py-2">{fmt(t.latency_ms)} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

export function RawJson({ data }: { data: unknown }) {
  return (
    <details className="bg-white border border-slate-200 rounded-lg p-5">
      <summary className="text-sm font-semibold text-brand-800 cursor-pointer">
        Raw JSON
      </summary>
      <pre className="mt-3 text-[12px] font-mono text-slate-700 overflow-x-auto whitespace-pre-wrap break-words">
        {JSON.stringify(data, null, 2)}
      </pre>
    </details>
  );
}

/* ---------- full results ---------- */

export function ClinicalResult({ data }: { data: ClinicalAnalysisResponse }) {
  return (
    <div className="space-y-4">
      <ViewsPanel data={data} />
      <RawJson data={data} />
    </div>
  );
}

function stringList(items: unknown[]): string {
  return items.map((x) => fmt(x)).join(", ");
}

export function CompleteResult({ data }: { data: CompleteIntakeResponse }) {
  return (
    <div className="space-y-4">
      <StatusBanner status={data.status} errorDetails={data.error_details} />
      {data.patient_dosage_card && <DosageCard src={data.patient_dosage_card} />}
      <MedsTable meds={data.grounded_medications} />
      {data.polypharmacy_intervention && (
        <SafetyPanel data={data.polypharmacy_intervention} />
      )}
      <HudGrid
        hud={data.judge_hud_summary}
        coherence={data.clinical_coherence_report}
        referral={data.referral_telemetry}
      />
      {data.active_triage_question && (
        <Section title="Triage question">
          <p className="text-sm text-slate-800">{data.active_triage_question}</p>
        </Section>
      )}
      {data.dialect_icd10_alignment.length > 0 && (
        <Section title="Dialect → ICD-10 alignment">
          <p className="text-sm text-slate-800">
            {stringList(data.dialect_icd10_alignment)}
          </p>
        </Section>
      )}
      <Section title="CV telemetry">
        <DictTable data={data.cv_telemetry} />
      </Section>
      {Object.keys(data.scan_biometrics).length > 0 && (
        <Section title="Scan biometrics">
          <DictTable data={data.scan_biometrics} />
        </Section>
      )}
      {data.voice_intake && Object.keys(data.voice_intake).length > 0 && (
        <Section title="Voice intake">
          <DictTable data={data.voice_intake} />
        </Section>
      )}
      <TraceTable items={data.agent_execution_trace} />
      <RawJson data={data} />
    </div>
  );
}

export function CombinedResult({ data }: { data: CombinedIntakeResponse }) {
  return (
    <div className="space-y-4">
      <StatusBanner status={data.status} errorDetails={data.error_details} />
      {data.cross_modal_fusion && <FusionPanel data={data.cross_modal_fusion} />}
      {data.voice && (
        <div className="space-y-3">
          <h3 className="text-base font-bold text-slate-900">Voice branch</h3>
          <ViewsPanel data={data.voice} />
        </div>
      )}
      {data.prescription && (
        <div className="space-y-3">
          <h3 className="text-base font-bold text-slate-900">
            Prescription branch
          </h3>
          <CompleteResult data={data.prescription} />
        </div>
      )}
      {data.xray && (
        <div className="space-y-3">
          <h3 className="text-base font-bold text-slate-900">X-ray branch</h3>
          <CompleteResult data={data.xray} />
        </div>
      )}
      <RawJson data={data} />
    </div>
  );
}
