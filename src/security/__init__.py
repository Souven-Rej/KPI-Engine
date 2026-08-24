"""Security and access-policy helpers for the KPI Engine."""

from .rbac import authorize_role, get_role_context

__all__ = ["get_role_context", "authorize_role"]
