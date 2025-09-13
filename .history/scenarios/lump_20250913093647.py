"""scenarios/lump.py — Lump-sum scenario (compute + render_details)

This module stays UI-light: Streamlit is only used in render_details().
The compute() path is pure and safe to call from tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from core.types import AppConfig, ScenarioResult

# ---------------------------------------------------------------------
# Helpers (local, minimal) — avoid hard dependency on unfinished modules
# ---------------------------------------------------------------------

def _alloc_tail(label: str) -> Optional[str]:
    label = str(label).strip()
    if label.lower().endswith("e"):
        num = label[:-1]
        if num.isdigit():
            return f"{num}E"
    # Common prefixes like "SPX60E" or "Global 60E"
    for tok in label.replace("_", " ").split():
        if tok.endswith("E") and tok[:-1].isdigit():
            return f"{tok[:-1]}E"
    return None


def _list_allocations(df: pd.DataFrame) -> List[str]:
    tails = set()
    for c in df.columns:
        t = _alloc_tail(c)
        if t:
            tails.add(t)
    def _key(x: str) -> int:
        try:
            return -int(x[:-1])  # 100E→0E
        except Exception:
            return 0
    return sorted(tails, key=_key)


def _common_allocs(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[str]:
    return sorted(set(_list_allocations(df_a)) & set(_list_allocations(df_b)), key=lambda x: -int(x[:-1]))


def _find_alloc_column(df: pd.DataFrame, alloc_tail: str) -> Optional[str]:
    # Prefer exact tail match, else look for token containing the tail
    cols = [c for c in df.columns if str(c).strip().endswith(alloc_tail)]
    if cols:
        return cols[0]
    for c in df.columns:
        if alloc_tail in str(c):
            return c
    return None

# ---------------------------------------------------------------------
# Core compute
# ---------------------------------------------------------------------

def compute(
    config: AppConfig,
    df_glob: pd.DataFrame,
    df_spx: pd.DataFrame,
    allocs: List[str] | List,   # can pass list of canonical tails; we also infer if empty
    helpers: Dict[str, Any],
) -> ScenarioResult:
    """Compute min/median ending values for investing the **lump-sum difference** across all historical windows.

    Requirements/assumptions (best-effort):
    - df_glob/df_spx contain per-window FV multipliers in a column named 'fv_multiple', OR
      they contain per-period return factors already window-rolled into columns by allocation (e.g., '60E').
    - When 'fv_multiple' is absent, we treat each allocation column's values as an FV multiple per window.
    - The invested amount is (spend_thinking - spend_frugal) and must be > 0.
    """
    spend_diff = float(max(0.0, config.spend_thinking - config.spend_frugal))
    if spend_diff <= 0:
        empty = pd.DataFrame(columns=[
            "Allocation",
            "Global Minimum Ending Value",
            "SPX Mininimum Ending Value",
            "Global Median Ending Value",
            "SPX Median Ending Value",
        ])
        return ScenarioResult(table=empty.copy(), raw=empty.copy(), extras={}, callouts=["No positive lump-sum difference to invest."])

    # Resolve allocations
    alloc_list: List[str] = list(allocs) if allocs else _common_allocs(df_glob, df_spx)
    if not alloc_list:
        alloc_list = _list_allocations(df_glob)
    if not alloc_list:
        alloc_list = _list_allocations(df_spx)

    rows_raw: List[Dict[str, Any]] = []
    for alloc in alloc_list:
        col_g = _find_alloc_column(df_glob, alloc)
        col_s = _find_alloc_column(df_spx,  alloc)

        g_min = g_med = np.nan
        s_min = s_med = np.nan

        # If there is a universal 'fv_multiple' per window and an 'alloc' filter, respect it via helpers
        # Otherwise assume columns contain FV multiples per window already.
        if col_g is not None:
            series_g = pd.to_numeric(df_glob[col_g], errors="coerce")
            if series_g.notna().any():
                end_g = series_g * spend_diff
                g_min = float(np.nanmin(end_g.values))
                g_med = float(np.nanmedian(end_g.values))
        if col_s is not None:
            series_s = pd.to_numeric(df_spx[col_s], errors="coerce")
            if series_s.notna().any():
                end_s = series_s * spend_diff
                s_min = float(np.nanmin(end_s.values))
                s_med = float(np.nanmedian(end_s.values))

        rows_raw.append({
            "Allocation": alloc,
            "Global Minimum Ending Value": g_min,
            "SPX Mininimum Ending Value": s_min,
            "Global Median Ending Value": g_med,
            "SPX Median Ending Value": s_med,
        })

    raw = pd.DataFrame(rows_raw)[[
        "Allocation",
        "Global Minimum Ending Value",
        "SPX Mininimum Ending Value",
        "Global Median Ending Value",
        "SPX Median Ending Value",
    ]]

    # Formatted table for UI
    fmt_rows = []
    for r in rows_raw:
        fmt_rows.append({
            "Allocation": r["Allocation"],
            "Global Minimum Ending Value": (None if pd.isna(r["Global Minimum Ending Value"]) else f"${r['Global Minimum Ending Value']:,.0f}"),
            "SPX Mininimum Ending Value": (None if pd.isna(r["SPX Mininimum Ending Value"]) else f"${r['SPX Mininimum Ending Value']:,.0f}"),
            "Global Median Ending Value": (None if pd.isna(r["Global Median Ending Value"]) else f"${r['Global Median Ending Value']:,.0f}"),
            "SPX Median Ending Value": (None if pd.isna(r["SPX Median Ending Value"]) else f"${r['SPX Median Ending Value']:,.0f}"),
        })
    table = pd.DataFrame(fmt_rows)[list(raw.columns)]

    return ScenarioResult(table=table, raw=raw, extras={}, callouts=[])

# ---------------------------------------------------------------------
# UI rendering (details only)
# ---------------------------------------------------------------------

def render_details(result: ScenarioResult,
                   title: str = "Opportunity Cost — Lump Sum",
                   csv_name_prefix: str = "lump") -> None:
    st.subheader("Opportunity Cost of the Difference for Lump Sum Spending")
    st.markdown("**Thinking vs What-if difference invested across all historical windows**")
    st.caption("Assumes 0.20% (Global) and 0.05% (SP500) annual expenses.")

    if result.table is None or result.table.empty:
        st.info("No lump-sum opportunity cost to show.")
        return

    st.dataframe(result.table, use_container_width=True)
    st.download_button(
        "Download table (CSV)",
        data=(result.raw or pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        file_name=f"{csv_name_prefix}_min_median.csv",
        mime="text/csv",
    )