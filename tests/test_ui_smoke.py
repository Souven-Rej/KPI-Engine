from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.ui import streamlit_app


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeSessionState(dict):
    pass


def test_main_initializes_with_empty_data(monkeypatch):
    fake_sidebar = SimpleNamespace(
        image=lambda *args, **kwargs: None,
        title=lambda *args, **kwargs: None,
        markdown=lambda *args, **kwargs: None,
        header=lambda *args, **kwargs: None,
        caption=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        button=lambda *args, **kwargs: False,
        selectbox=lambda *args, **kwargs: "analyst",
    )

    fake_st = SimpleNamespace(
        set_page_config=lambda *args, **kwargs: None,
        sidebar=fake_sidebar,
        session_state=_FakeSessionState(),
        spinner=lambda *args, **kwargs: _DummyContext(),
        info=lambda *args, **kwargs: None,
        title=lambda *args, **kwargs: None,
        markdown=lambda *args, **kwargs: None,
        caption=lambda *args, **kwargs: None,
        subheader=lambda *args, **kwargs: None,
        metric=lambda *args, **kwargs: None,
        plotly_chart=lambda *args, **kwargs: None,
        success=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        columns=lambda *args, **kwargs: [SimpleNamespace(metric=lambda *a, **k: None)],
        expander=lambda *args, **kwargs: _DummyContext(),
        progress=lambda *args, **kwargs: None,
        form=lambda *args, **kwargs: _DummyContext(),
        radio=lambda *args, **kwargs: "Yes - Spot On",
        text_area=lambda *args, **kwargs: "",
        form_submit_button=lambda *args, **kwargs: False,
    )

    monkeypatch.setattr(streamlit_app, "st", fake_st)
    monkeypatch.setattr(streamlit_app, "load_and_detect", lambda: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(streamlit_app, "_load_roles_from_contract", lambda: {"analyst": "Analyst"})
    monkeypatch.setattr(streamlit_app, "authorize_role", lambda *args, **kwargs: True)
    monkeypatch.setattr(streamlit_app, "RuntimeTelemetry", lambda *args, **kwargs: SimpleNamespace(record=lambda *a, **k: None, summary=lambda: {}))
    monkeypatch.setattr(streamlit_app, "get_governance_summary", lambda: {"lineage": {}, "source_freshness": {}})
    monkeypatch.setattr(streamlit_app, "get_kpi_lineage", lambda *args, **kwargs: {"display_name": "Revenue", "source_table": "sales_daily"})

    streamlit_app.main()
