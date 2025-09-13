

"""app/ui_components.py — small UI helpers (cards, CSS)"""
from __future__ import annotations
import streamlit as st

_DEF_CARD_CSS = """
<style>
  .info-card{background:#0b5ed7;color:#fff;border:1px solid #084298;border-radius:8px;padding:12px 14px;text-align:center}
  .info-card h4{margin:0 0 6px 0;font-weight:700}
  .info-card .value{font-size:1.25rem;font-weight:700;margin-top:2px}
  .info-card .sub{opacity:.95;font-size:.9rem;margin-top:6px}
</style>
"""

def add_info_card_css() -> None:
  """Inject default CSS for the blue info cards."""
  st.markdown(_DEF_CARD_CSS, unsafe_allow_html=True)

def card(header: str, value: str, sub: str = "") -> None:
  """Render a centered blue card with an optional sublabel."""
  st.markdown(
    f"""
    <div class='info-card'>
      <h4>{header}</h4>
      <div class='value'>{value}</div>
      {f"<div class='sub'>{sub}</div>" if sub else ""}
    </div>
    """,
    unsafe_allow_html=True,
  )