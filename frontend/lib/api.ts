/* Typed client for the Nirvor FastAPI backend (default http://127.0.0.1:8000).
   Shapes mirror backend/models/schemas.py; see backend/api/routers/intake.py. */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export interface SystemStatus {
  meta_configured: boolean;
  modal_configured: boolean;
  version: string;
}

export interface VerificationTelemetry {
  primary_vector_match_score: number;
  grounding_source_text: string;
  hallucination_risk_status: "Low" | "Medium" | "High";
}

export interface DoctorView {
  clinical_summary: string;
  dialect_icd10_alignment: string[];
  active_triage_question: string;
  grounded_medications: string[];
  verification_telemetry: VerificationTelemetry;
}

export interface PatientView {
  layman_explanation: string;
  next_steps: string;
}

export interface ClinicalAnalysisResponse {
  doctor_view: DoctorView;
  patient_view: PatientView;
}

export interface ClinicalCoherenceReport {
  is_coherent: boolean;
  coherence_score: number;
  matched_indications: string[];
  contradiction_flags: string[];
}

export interface PolypharmacyIntervention {
  risk_level: "CRITICAL" | "HIGH" | "MODERATE" | "LOW";
  clinical_mechanism: string;
  recommended_action: string;
  bangla_patient_warning: string;
  verified_medications_assessed?: number;
  unverified_medications_skipped?: number;
}

export interface JudgeHudSummary {
  total_latency_ms: number;
  hallucinations_intercepted: number;
  mean_shannon_entropy: number;
  active_verification_flags: number;
}

export interface ReferralTelemetry {
  top_referral_match_score: number;
  referral_source_file: string;
  referral_verified: boolean;
  referral_medications: string;
  referral_confidence: "Low" | "Medium" | "High";
}

export interface AgentExecutionTraceItem {
  step_name: string;
  status: string;
  provider_used: string;
  latency_ms: number;
  spark_triggered_reason: string;
}

export type GroundedMedication = Record<string, unknown>;

export interface CompleteIntakeResponse {
  status: string;
  error_details?: string[] | null;
  cv_telemetry: Record<string, unknown>;
  voice_intake?: Record<string, unknown> | null;
  dialect_icd10_alignment: unknown[];
  clinical_coherence_report?: ClinicalCoherenceReport | null;
  active_triage_question?: string | null;
  polypharmacy_intervention?: PolypharmacyIntervention | null;
  patient_dosage_card?: string | null;
  grounded_medications: GroundedMedication[];
  scan_biometrics: Record<string, unknown>;
  referral_telemetry?: ReferralTelemetry | null;
  judge_hud_summary: JudgeHudSummary;
  agent_execution_trace: AgentExecutionTraceItem[];
}

export interface CrossModalFusion {
  modalities_present: string[];
  modalities_succeeded: string[];
  voice_confidence?: number | null;
  prescription_confidence?: number | null;
  xray_biometrics_present: boolean;
  fused_confidence: number;
  overall_risk: "Low" | "Medium" | "High";
  is_coherent: boolean;
  agreement_notes: string[];
}

export interface CombinedIntakeResponse {
  status: string;
  error_details?: string[] | null;
  voice?: ClinicalAnalysisResponse | null;
  prescription?: CompleteIntakeResponse | null;
  xray?: CompleteIntakeResponse | null;
  cross_modal_fusion?: CrossModalFusion | null;
}

export interface RouterCrash {
  error: string;
  details?: string;
}

export function isRouterCrash(body: unknown): body is RouterCrash {
  return (
    typeof body === "object" &&
    body !== null &&
    "error" in body &&
    typeof (body as RouterCrash).error === "string"
  );
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function postFiles(
  path: string,
  files: Record<string, File | null>
): Promise<unknown> {
  const form = new FormData();
  for (const [key, file] of Object.entries(files)) {
    if (file) form.append(key, file);
  }
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", body: form });
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    throw new ApiError(
      res.status,
      typeof detail === "string" ? detail : `Request failed (${res.status})`
    );
  }
  return body;
}

export async function getSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/api/v2/system-status`);
  if (!res.ok) throw new ApiError(res.status, "Backend unreachable");
  return (await res.json()) as SystemStatus;
}

export function postVoice(audio: File): Promise<unknown> {
  return postFiles("/api/v2/intake/voice", { audio });
}

export function postPrescription(prescription: File): Promise<unknown> {
  return postFiles("/api/v2/intake/prescription", { prescription });
}

export function postXray(xray_scan: File): Promise<unknown> {
  return postFiles("/api/v2/intake/xray", { xray_scan });
}

export function postCombined(files: {
  prescription?: File | null;
  audio?: File | null;
  xray_scan?: File | null;
}): Promise<unknown> {
  return postFiles("/api/v2/intake/combined", {
    prescription: files.prescription ?? null,
    audio: files.audio ?? null,
    xray_scan: files.xray_scan ?? null,
  });
}
