from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "config" / "kpi_contract.yaml"


def load_contract(contract_path: Path | str = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Load the YAML contract for RBAC and governance metadata."""
    contract_path = Path(contract_path)
    with contract_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload


def get_role_context(role_name: str, contract_path: Path | str = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Return the role policy for a persona, derived from the governance contract."""
    contract = load_contract(contract_path)
    roles = contract.get("roles", {})
    cfg = roles.get(role_name, {})
    fallback = {
        "display_name": role_name,
        "narrative_depth": "detailed_analysis",
        "visible_dimensions": ["region"],
        "show_statistical_details": True,
        "show_confidence_intervals": True,
        "show_prescriptive_actions": True,
        "show_raw_data_links": False,
    }
    return {**fallback, **cfg}


def authorize_role(role_name: str, action: str, contract_path: Path | str = DEFAULT_CONTRACT_PATH) -> bool:
    """Gate access by reusable action names used by the UI and narrative layer."""
    context = get_role_context(role_name, contract_path)
    action_map = {
        "show_statistics": context.get("show_statistical_details", False),
        "show_confidence": context.get("show_confidence_intervals", True),
        "show_prescriptive_actions": context.get("show_prescriptive_actions", True),
        "show_raw_data": context.get("show_raw_data_links", False),
    }
    return bool(action_map.get(action, True))
