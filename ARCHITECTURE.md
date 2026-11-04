# Nirvor: A Neuro-Symbolic Engine Architecture

## 1. Project Vision
Nirvor is an advanced, neuro-symbolic and agentic AI platform designed to handle multimodal clinical inputs such as regional voice notes, messy handwritten prescriptions, and diagnostic scans. The core vision of Nirvor is to surpass traditional linear application structures by relying on an agentic architecture, where specialized AI models and internal modules dynamically collaborate in an intelligent workflow.

## 2. The Agentic Design
Nirvor operates using an agentic orchestration loop, shifting the complexity of multimodal data processing to specialized sub-agents. These agents interact with internal and external APIs (such as Meta's models and Baseten endpoints) in a coordinated multi-step process rather than random, isolated calls.

### Orchestrator (`IntakeOrchestrator`)
The orchestrator manages how different AI models collaborate and pass states/outputs.
1. **CV Processing (Synchronous)**: Extracts and enhances the prescription image for optimal OCR.
2. **Concurrent Task Execution (Asynchronous)**: Analyzes voice notes for regional dialect/symptoms, while simultaneously segmenting diagnostic scans (like X-rays).
3. **Data Fusion & Mathematical Grounding**: Merges OCR output with voice-aligned transcription to perform bayesian entity matching against a comprehensive clinical database.
4. **Agentic Sparks & Guardrails**: Evaluates grounded outputs. Evaluates clinical coherence and contradicts between symptoms and matched medications, creates dosage cards, checks for polypharmacy toxicity and dynamically asks triage questions if Shannon entropy is high.

## 3. System Architecture & Folder Structure

```text
backend/
├── agents/             # Agentic orchestration layer
│   └── orchestrator.py # IntakeOrchestrator manages the flow of agents
├── api/                # FastAPI routing logic
│   └── routers/
│       └── intake.py   # Primary intake pipeline API definition
├── config/             # Environment and settings configuration
│   └── settings.py
├── core/               # Core processing modules and standalone components
│   ├── audio_pipeline.py # Processing for regional voice notes
│   ├── cv_pipeline.py    # Classic CV logic (illumination normalization, deskew)
│   ├── math_engine.py    # Bayes-based clinical math calculations & metrics
│   ├── meta_agent.py     # Endpoints interacting with multimodal models
│   ├── modal_worker.py   # Cloud worker offloading
│   └── prompts.py        # System prompt constants and builders
├── database/           # Static clinical datasets & fallback fixtures
│   ├── dims_drug_db.json
│   └── fallback_fixtures.py
└── main.py             # FastAPI App execution
```

### Sub-Agent Overview
- **Audio Pipeline (`audio_pipeline.py`)**: Dedicated to regional dialect speech-processing.
- **CV Pipeline (`cv_pipeline.py`)**: Uses classical computer vision (e.g. CLAHE, DLT) to clean prescription data.
- **Math Engine (`math_engine.py`)**: The neuro-symbolic heart containing clinical math routines, Bayesian cross-modal fusion, vector embeddings, and entropy calculations.
- **Meta Agent (`meta_agent.py`)**: A wrapper around large multimodal/LLM extraction models that processes the raw tokens and segments.

## 4. Technology Stack
- **Python / FastAPI**: High-performance backend powering the API.
- **Agentic Orchestration**: Custom python classes handling multimodal coordination.
- **NumPy & OpenCV**: Classical CV and numeric computations.
- **Meta Models (Llama/SAM/Muse)**: Inference blocks for complex transcription, segmentation, and multimodal data extraction.
- **Baseten**: Serves specialized ML extraction tasks.
- **Modal**: Serverless cloud execution environment.

## 5. Developer Guidelines
- **Modularity**: Maintain a strict separation of concerns. Do not place complex agentic logic in FastAPI routers. Routers should primarily pass the request context into the orchestration layer.
- **Extending the Network**: When adding new models (e.g. an MRI-analysis agent), wrap the external call in `backend/core/` and update `IntakeOrchestrator` to dispatch this concurrently if possible, adhering to the standard input/output schema.
- **Error Handling**: Use graceful degradation. Ensure AI endpoints have robust fallback mechanisms (e.g., as seen in `database/fallback_fixtures.py`) to handle timeouts or missing API keys in non-production environments.
