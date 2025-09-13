

"""app/toggles.py — Summary/Details radio helpers"""
from __future__ import annotations
import streamlit as st

_DEF = "Summary only"


def init_toggle() -> None:
    """Initialize the view_mode state once per session."""
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = _DEF


def show_details() -> bool:
    """Return True when user selected 'Show full details'."""
    return st.session_state.get("view_mode", _DEF) == "Show full details"


def render_radio_below_summary() -> None:
    """Render a horizontal radio just below the summary section."""
    st.radio(
        "View mode",
        ["Summary only", "Show full details"],
        index=(0 if st.session_state.get("view_mode", _DEF) == _DEF else 1),
        horizontal=True,
        key="view_mode",
    )