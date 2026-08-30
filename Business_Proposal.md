# ⚡ KPI Engine: Detailed Business Proposal
**Accenture Innovation Challenge 2026 — BusinessIntelligence.ai Track**

---

## 1. Problem Framing
### The "AI Trust Gap" in Enterprise Analytics
Modern enterprises are drowning in data but starving for actionable truth. When a regional KPI like "Net Revenue" suddenly drops, data engineering teams spend 48 to 72 hours running SQL queries, building ad-hoc dashboards, and debating correlation versus causation. 

Recent attempts to solve this using Generative AI (LLMs) have introduced a critical flaw: **Hallucination in Causal Inference.** Standard LLMs are language predictors, not statisticians. When asked *why* revenue dropped, they frequently guess based on training data biases rather than mathematical reality. This "AI Trust Gap" prevents enterprise executives from deploying automated analytics for high-stakes financial decisions.

## 2. Solution Design
### The Deterministic-First Architecture
**KPI Engine** bridges this gap by fundamentally separating *finding the truth* from *explaining the truth*. Our core philosophy is: **"Math does the finding. AI does the explaining."**

The solution architecture consists of a strict 4-phase pipeline:
1. **Anomaly Detection (STL):** We use Seasonal-Trend Decomposition using Loess (STL) to isolate the residual noise and mathematically flag anomalous KPI movements (Z-score > 2.0).
2. **Causal Attribution (DoWhy / Counterfactual SCM):** We deploy Structural Causal Models to map the data flow (e.g., Ad Spend → Web Traffic → Revenue). By simulating counterfactuals (Shapley values), we definitively identify the root cause of the anomaly.
3. **Prescriptive Analytics (EconML / Double Machine Learning):** We calculate the Conditional Average Treatment Effect (CATE) to estimate the exact dollar-value revenue lift if the root cause is resolved.
4. **AI Synthesis (Google Gemini):** Only after the math is verified does the LLM step in. It is restricted to translating the JSON payload into plain English, dynamically adjusting its tone based on Role-Based Access Control (RBAC).

**The Abstention Protocol:** If the causal engine detects sparse data (<30 days history), it triggers an "Investigation Required" state, explicitly forbidding the AI from guessing. This guarantees enterprise safety.

## 3. Target Users
KPI Engine is designed for cross-functional enterprise adoption through dynamic, persona-driven interfaces:

*   **Executive Leadership (e.g., VP of Sales):**
    *   **Needs:** High-level strategic visibility, total revenue impact, and macro-level lever adjustments.
    *   **Experience:** Receives concise executive summaries focused on financial lift and strategic directives.
*   **Operational Leaders (e.g., Regional Managers):**
    *   **Needs:** Tactical, ground-level instructions to resolve supply chain or marketing bottlenecks.
    *   **Experience:** Receives detailed operational breakdowns, localized CATE estimates, and immediate action items.
*   **Data Science / Analytics Teams:**
    *   **Needs:** Traceability, lineage, and model confidence metrics to trust the automated output.
    *   **Experience:** Has access to the telemetry footer, DoWhy trace cards, and the Analyst Feedback Loop for continuous model tuning.

## 4. Business Case and Impact
### Return on Investment (ROI)
*   **Time-to-Insight Reduction:** Reduces root-cause analysis time from an average of 72 hours to 3 seconds per anomaly.
*   **Operational Efficiency:** Eliminates "wild goose chases" where operational teams react to correlated symptoms rather than true causal drivers.
*   **Revenue Recovery:** By prescribing exact CATE dollar values (e.g., "$1,500 expected lift if lever restored"), the engine prioritizes interventions based on maximum financial recovery.
*   **Compute Cost Reduction:** By running heavy machine learning models (DoWhy/EconML) exclusively on flagged anomalies rather than the entire dataset, cloud compute costs are minimized. The LLM synthesis costs less than $0.001 per event.

## 5. Phased Roadmap (Hackathon Execution)

### Phase 1: Data Architecture & Ground Truth Setup
* Engineered a multi-grain synthetic data generator (`sales_daily`, `marketing_weekly`, `inventory_snapshot`).
* Injected specific ground-truth anomalies (Marketing Shocks, Supply Chain Stockouts) to validate mathematical accuracy later in the pipeline.
* Defined the semantic data models and causal graphs in `kpi_contract.yaml`.

### Phase 2: Deterministic Causal Engine Implementation
* Implemented Seasonal-Trend Decomposition (STL) for baseline anomaly detection.
* Built a custom Structural Causal Model (SCM) using `dowhy` and `sklearn` to mathematically trace root-cause attribution.
* Integrated `econml` for prescriptive Double Machine Learning (CATE estimation) to calculate exact revenue lift.

### Phase 3: AI Narrative & Role-Based Access
* Integrated Google Gemini 3.6 Flash using strictly typed JSON schemas (Native Structured Outputs).
* Developed the dynamic Persona system to alter narrative depth for Executives (VP) vs. Operational (Regional Managers).
* Programmed the "Abstention Protocol" to hard-stop the LLM during data ambiguity.

### Phase 4: Interactive Dashboard & Telemetry
* Designed a professional Streamlit dashboard featuring glass-morphism UI, trace panels, and Plotly visualizations.
* Built the continuous Human-in-the-Loop Analyst Feedback mechanism.
* Integrated deep runtime telemetry to track model latency, token usage, and API costs for enterprise auditability.

## 6. Key Risks & Mitigations

| Risk Factor | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **LLM Hallucinations** | High | **Strict Determinism:** LLMs are restricted via system prompts to only output what is in the provided JSON. **Abstention Protocol:** Triggers "Investigation Required" on low-confidence data. |
| **Data Sparsity / Cold Starts** | Medium | Causal attribution requires historical baselines. We mitigate this by defaulting to transparent "Low Confidence" UI warnings rather than presenting false certainty. |
| **Causal Graph Misconfiguration** | High | If the DAG (Directed Acyclic Graph) is wrong, the attribution is wrong. **Mitigation:** The DAG is defined declaratively in `kpi_contract.yaml`, requiring human-in-the-loop sign-off from domain experts before deployment. |
| **User Adoption / Trust** | Medium | **Analyst Feedback Loop:** Users can flag incorrect attributions in the UI. We display the math explicitly on screen (Counterfactual Decomposition traces) so the system acts as a "glass box" rather than a black box. |

---
*Prepared for the Accenture Innovation Challenge 2026. Codebase and prototype available in the project repository.*
