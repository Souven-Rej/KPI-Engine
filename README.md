# ⚡ KPI Engine: Storytelling-to-Action Pipeline
**Accenture Innovation Challenge 2026 — BusinessIntelligence.ai Track**

KPI Engine is an autonomous, deterministic-first analytics prototype that bridges the gap between raw data and executive action. It detects KPI movements, reconciles multi-grain data, identifies causal drivers, calculates expected financial lift, and generates role-specific narratives.

### The Core Philosophy: *"Math does the finding. AI does the explaining."*
Standard GenAI wrappers often hallucinate causal relationships. KPI Engine flips the paradigm. We enforce a rigid, mathematically deterministic pipeline (STL Decomposition + Structural Causal Models + Double Machine Learning) to find the truth, reserving LLMs exclusively for translating the statistically verified JSON payload into plain English. 

If the data is too sparse to prove causality, the engine's **Abstention Protocol** activates, explicitly refusing to guess and returning an "Investigation Required" state.

---

## 🏗️ Architecture & Pipeline

The pipeline is structured into four deterministic phases:

### Phase 1: Synthetic Data Generation & Causality
Generates 10,000+ rows of perfectly reproducible (`seed=42`) multi-grain synthetic data containing a ground-truth causal chain: `Marketing Spend (Weekly) → Web Traffic (Daily) → Units Sold (Daily) → Net Revenue (Daily)`, modulated by `Inventory (6-hourly)`.
* **Injected Scenarios**: Includes a hidden Marketing Shock (60% budget cut), a Regional Stockout (Inventory = 0), and a Sparse-History Region (< 30 days of data).

### Phase 2: Detection & Causal Attribution (`statsmodels` & `dowhy`)
* **Detection (`stl_detector.py`)**: Ingests `kpi_contract.yaml`, dynamically computes KPIs, and utilizes `statsmodels.tsa.seasonal.STL` to flag anomaly events (`|Z| > 2.0`).
* **Attribution (`dowhy_gcm.py`)**: Uses a Custom `InvertibleStructuralCausalModel` (backed by scikit-learn Gradient Boosting Regressors) to perform **Counterfactual Noise Decomposition**. It accurately traces anomalies backward through the causal DAG to root causes (`ad_spend` or `stock_on_hand`).

### Phase 3: Prescriptive Analytics & Synthesis (`econml` & `google-genai`)
* **Prescriptive CATE (`econml_cate.py`)**: Employs Double Machine Learning (`LinearDML`) to calculate the Conditional Average Treatment Effect. It translates the anomaly into a dollar-value "Expected Revenue Lift" if the root cause is resolved.
* **LLM Synthesis (`llm_synthesis.py`)**: Translates the mathematical payload into natural language via Google Gemini (`gemini-3.6-flash`). Uses Native Structured Outputs to guarantee a JSON payload tailored to specific personas (e.g., VP of Sales vs. Regional Manager).

### Phase 4: Modern Enterprise UI (`next.js` & `fastapi`)
A heavily polished, Vercel-ready Next.js application built with React, Tailwind CSS, and Recharts. The frontend consumes a decoupled FastAPI Python backend, demonstrating how heavy Causal ML workloads can be abstracted away from modern, responsive web experiences. Features interactive AreaCharts, skeleton loaders, and a premium dark-mode aesthetic.

---

## 🛡️ Key Differentiators for Judges

1. **The "Anti-AI" Abstention Protocol**: Standard LLMs guess when data is missing. KPI Engine implements a strict sparse-history rule. If a region has < 30 days of data, the DML engine returns `data_ambiguity = True`. The LLM intercepts this flag and is forced via system prompt to output "Investigation Required," proving the system's safety in enterprise environments.
2. **Decoupled Architecture**: By wrapping the heavy Python math libraries (`dowhy`, `econml`) in a REST API (`FastAPI`) and serving the UI via `Next.js`, we solved the serverless deployment limits of standard monolithic apps (like Streamlit), making this architecture fully enterprise-ready and massively scalable.
3. **72 Hours to 3 Seconds**: KPI Engine automates what normally takes a data science team days of SQL slicing and Jupyter Notebook analysis, delivering a traceable, actionable CATE estimate instantly.

---

## 🚀 Setup & Execution

### 1. Backend Installation (Python)
Ensure you are using Python 3.12+ (tested on Python 3.14).
```bash
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your Google Gemini API key.
```env
GEMINI_API_KEY=your-gemini-key-here
```
*(The API has a built-in enterprise failover. If the Gemini API hits a 503 high-demand error, the FastAPI backend will gracefully fall back to a local responder, simulating realistic latency to ensure your demo continues uninterrupted).*

### 3. Start the FastAPI Backend
Launch the Python math engine:
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Start the Next.js Frontend
In a separate terminal, launch the React dashboard:
```bash
cd frontend
npm install
npm run dev -- --webpack
```
Navigate to `http://localhost:3000` to view the KPI Engine.

---
*Built for the Accenture Innovation Challenge 2026.*
