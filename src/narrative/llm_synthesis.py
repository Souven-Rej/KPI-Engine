"""
Narrative Synthesis Engine for KPI Engine
==========================================

Translates structured, statistically verified JSON findings into plain English,
conditioned on user persona, using LLMs. Enforces a strict abstention protocol:
"Math does the finding. AI does the explaining."

Features:
    - Pydantic-enforced structured JSON output.
    - Persona conditioning (e.g., vp_sales vs regional_manager).
    - Abstention Protocol: If data_ambiguity == True, refuses to guess.
    - Telemetry tracking (latency, tokens).
"""

import json
import logging
import os
import time
from typing import Any, Tuple

from pydantic import BaseModel, Field

# Support either OpenAI or Anthropic; defaulting to OpenAI for structured outputs
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


# ============================================================
# PYDANTIC SCHEMA
# ============================================================

class SynthesisResponse(BaseModel):
    narrative_summary: str = Field(
        ...,
        description="A plain English summary of the KPI anomaly and its causes, tailored to the persona."
    )
    key_drivers: list[str] = Field(
        ...,
        description="A list of 1-2 sentence bullet points explaining the primary causal drivers."
    )
    recommended_actions: list[str] = Field(
        ...,
        description="A list of prescriptive actions based on the causal and prescriptive analysis."
    )
    confidence_status: str = Field(
        ...,
        description="The confidence status of the analysis. MUST be 'Investigation Required' if data is ambiguous."
    )


# ============================================================
# LLM SYNTHESIS
# ============================================================

def generate_narrative(
    anomaly_data: dict[str, Any],
    attribution_data: dict[str, Any],
    prescriptive_data: dict[str, Any],
    persona: str,
    data_ambiguity: bool = False,
    model: str = "gpt-4o",
) -> Tuple[SynthesisResponse, dict[str, Any]]:
    """
    Generate a role-specific narrative using an LLM.

    Args:
        anomaly_data: Dict of anomaly details (date, region, severity, revenue drop).
        attribution_data: Dict of causal attributions (e.g., ad_spend: 92%).
        prescriptive_data: Dict of prescriptive CATE impacts (e.g., expected revenue lift).
        persona: The target role (e.g., "vp_of_sales", "regional_manager").
        data_ambiguity: Boolean flag from the prescriptive/causal engine indicating sparse history.
        model: LLM model name to use.

    Returns:
        Tuple of (SynthesisResponse, telemetry_dict).
    """
    if OpenAI is None:
        raise ImportError("openai package is not installed. Please install it to use LLM synthesis.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment. Using mock LLM response.")
        return _mock_generate_narrative(
            anomaly_data, attribution_data, prescriptive_data, persona, data_ambiguity
        )

    client = OpenAI(api_key=api_key)

    # 1. Construct System Prompt
    system_prompt = (
        "You are the KPI Storytelling Engine. Your job is to translate statistically verified "
        "data into plain English. \n\n"
        "CORE RULES:\n"
        "1. Math does the finding. AI does the explaining. DO NOT guess causes or calculate data.\n"
        f"2. You are writing for the persona: {persona.upper()}.\n"
    )

    if persona.lower() == "vp_of_sales":
        system_prompt += (
            "   - Tone: Executive summary, strategic, high-level.\n"
            "   - Focus: Revenue impact and strategic lever adjustments.\n"
        )
    elif persona.lower() == "regional_manager":
        system_prompt += (
            "   - Tone: Detailed, operational, action-oriented.\n"
            "   - Focus: Regional execution and immediate operational fixes.\n"
        )
    else:
        system_prompt += "   - Tone: Informative and clear.\n"

    # Abstention Protocol
    if data_ambiguity:
        system_prompt += (
            "\nABSTENTION PROTOCOL ACTIVATED: The data history is too sparse to make a confident "
            "causal attribution. You MUST NOT guess. Set `confidence_status` to 'Investigation Required' "
            "and explain in the narrative that missing or conflicting data prevents a definitive conclusion."
        )

    # 2. Construct User Prompt
    user_prompt = (
        f"Anomaly Detected:\n{json.dumps(anomaly_data, indent=2)}\n\n"
        f"Causal Attribution:\n{json.dumps(attribution_data, indent=2)}\n\n"
        f"Prescriptive Analytics:\n{json.dumps(prescriptive_data, indent=2)}\n\n"
        f"Data Ambiguity Flag: {data_ambiguity}\n\n"
        "Generate the structured synthesis based ONLY on the data provided."
    )

    # 3. API Call with Telemetry
    start_time = time.time()
    
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=SynthesisResponse,
            temperature=0.0,
        )
        end_time = time.time()

        parsed_response = response.choices[0].message.parsed
        
        telemetry = {
            "latency_seconds": round(end_time - start_time, 3),
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "model_used": model
        }

        logger.info(
            "LLM Synthesis Complete: Latency=%.2fs, Tokens=%d",
            telemetry["latency_seconds"],
            telemetry["total_tokens"],
        )
        
        return parsed_response, telemetry

    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        raise


def _mock_generate_narrative(
    anomaly_data: dict,
    attribution_data: dict,
    prescriptive_data: dict,
    persona: str,
    data_ambiguity: bool
) -> Tuple[SynthesisResponse, dict[str, Any]]:
    """Mock responder for local testing without an API key."""
    time.sleep(1.2)  # Simulate network latency
    
    if data_ambiguity:
        resp = SynthesisResponse(
            narrative_summary="Data history is too sparse (<30 days) to confidently determine the cause of the anomaly. More data collection is needed.",
            key_drivers=["Insufficient historical data (Widget_C scenario)."],
            recommended_actions=["Monitor KPI for 30 days before running causal models.", "Manually review regional operational logs."],
            confidence_status="Investigation Required"
        )
    else:
        resp = SynthesisResponse(
            narrative_summary=f"({persona.upper()}) Revenue dropped primarily due to a disruption in {attribution_data.get('primary_driver', 'key metrics')}.",
            key_drivers=[f"{k}: {v}%" for k, v in attribution_data.items() if str(k).endswith("_contribution_pct")],
            recommended_actions=["Restore primary driver to baseline.", "Monitor secondary indicators."],
            confidence_status="High"
        )
        
    telemetry = {
        "latency_seconds": 1.205,
        "prompt_tokens": 150,
        "completion_tokens": 85,
        "total_tokens": 235,
        "model_used": "mock-llm-local"
    }
    
    return resp, telemetry
