from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "config" / "kpi_contract.yaml"


def load_contract(contract_path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Load the KPI contract for governance metadata."""
    with Path(contract_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_kpi_lineage(kpi_name: str, contract_path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Return a structured lineage summary for a KPI definition."""
    contract = load_contract(contract_path)
    kpi_cfg = contract.get("kpis", {}).get(kpi_name, {})
    data_sources = contract.get("data_sources", {})

    return {
        "kpi": kpi_name,
        "display_name": kpi_cfg.get("display_name", kpi_name),
        "source_table": kpi_cfg.get("source_table"),
        "source_path": kpi_cfg.get("source_path"),
        "formula": kpi_cfg.get("formula"),
        "grain": kpi_cfg.get("grain"),
        "dimensions": kpi_cfg.get("dimensions", []),
        "upstream_sources": [
            {
                "name": source_name,
                "path": source_cfg.get("path"),
                "grain": source_cfg.get("grain"),
                "date_column": source_cfg.get("date_column"),
            }
            for source_name, source_cfg in data_sources.items()
            if source_name in {"sales_daily", "marketing_weekly", "inventory_hourly"}
        ],
        "contract_loaded_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def get_source_freshness(contract_path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Return a freshness summary for known source systems."""
    contract = load_contract(contract_path)
    data_sources = contract.get("data_sources", {})
    freshness: dict[str, Any] = {}
    for name, cfg in data_sources.items():
        freshness[name] = {
            "grain": cfg.get("grain"),
            "path": cfg.get("path"),
            "refresh_cadence": cfg.get("grain"),
            "date_column": cfg.get("date_column"),
        }
    return freshness


def get_governance_summary(contract_path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Return a concise enterprise governance summary."""
    contract = load_contract(contract_path)
    return {
        "kpis": list(contract.get("kpis", {}).keys()),
        "roles": list(contract.get("roles", {}).keys()),
        "source_freshness": get_source_freshness(contract_path),
        "lineage": {
            name: get_kpi_lineage(name, contract_path)
            for name in contract.get("kpis", {}).keys()
        },
    }
