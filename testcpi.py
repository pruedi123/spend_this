

# testcpi.py — Streamlit helper to chain CPI deflators every 12 rows for 5 years
# Goal: from cpi_factors.xlsx in the SAME directory, grab the first row factor,
# then the factor 12 rows below it, and so on for 5 total values (years 0..4),
# show the five factors, their product, and the resulting sticker price on a base.

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st

# Try pandas + openpyxl for XLSX; if unavailable, we’ll show a friendly message
try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except Exception:
    pd = None  # type: ignore
    _HAS_PANDAS = False

APP_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
DEFAULT_XLSX = APP_DIR / "cpi_factors.xlsx"

st.set_page_config(page_title="CPI 12-step Chain (5 years)", layout="centered")
st.title("CPI 12‑Step Chain (5 Years)")
st.caption("Reads monthly **increase_factor** deflators from `cpi_factors.xlsx` in this folder and chains every 12 rows for five years.")

with st.sidebar:
    st.header("Inputs")
    base_price = st.number_input("Base car price ($)", min_value=0.0, value=35000.0, step=500.0, format="%0.2f")
    start_row_1_based = st.number_input("Simulation start row (1‑based)", min_value=1, value=1, step=1)
    file_override = st.text_input("Path override (optional)", value=str(DEFAULT_XLSX) if DEFAULT_XLSX.exists() else "")
    st.markdown("---")
    st.caption("If Excel reading fails (missing `openpyxl`), you can upload a CSV with a column named **increase_factor**.")
    csv_upload = st.file_uploader("Upload CSV with monthly increase_factor", type=["csv"])

# --- Helpers -----------------------------------------------------------------

def _auto_pick_col(columns: List[str]) -> str | None:
    for c in columns:
        s = str(c).strip().lower().replace(" ", "")
        if "increase" in s and "factor" in s:
            return c
    return None

@st.cache_data(show_spinner=False)
def load_deflators(xlsx_path: Path | None, csv_buf) -> Tuple[np.ndarray, str]:
    """Load monthly deflator series (increase_factor ~ < 1 values).
    Returns (inc_array, meta_str).
    Priority: uploaded CSV -> XLSX on disk -> error.
    """
    # 1) Uploaded CSV takes priority
    if csv_buf is not None:
        if not _HAS_PANDAS:
            raise RuntimeError("pandas not available; cannot parse CSV upload.")
        df = pd.read_csv(csv_buf)
        col = "increase_factor" if "increase_factor" in df.columns else _auto_pick_col(list(df.columns))
        if not col:
            raise KeyError("Uploaded CSV must include a column like 'increase_factor'.")
        inc = pd.to_numeric(df[col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        return inc, f"csv:upload col={col} n={inc.size}"

    # 2) XLSX on disk
    if xlsx_path and xlsx_path.exists():
        if not _HAS_PANDAS:
            raise RuntimeError("pandas/openpyxl required to read .xlsx. Install with: pip install pandas openpyxl")
        # Try likely sheets/columns
        sheet_candidates = ["increase_factors", "cpi_increase", "cpi_factors"]
        col_candidates = ["increase_factor", "increase factor", "increase", "12 mo increae", "12 mo factor"]
        last_err = None
        for sh in sheet_candidates:
            try:
                df = pd.read_excel(xlsx_path, sheet_name=sh)
            except Exception as e:
                last_err = e
                continue
            # choose column
            col = None
            for candidate in col_candidates:
                if candidate in df.columns:
                    col = candidate; break
            if not col:
                col = _auto_pick_col(list(df.columns))
            if not col:
                last_err = KeyError(f"No increase_factor-like column in sheet '{sh}'")
                continue
            # IMPORTANT: your workbook has two styles. We want the **deflator** (<1) series.
            # If user accidentally picks a growth (>1) series ('12 mo factor'), invert it.
            series = pd.to_numeric(df[col], errors="coerce")
            if series.dropna().median() > 1.05:  # looks like a growth multiplier
                series = 1.0 / series
                meta = f"xlsx:{xlsx_path.name}:{sh} col={col} (inverted) n={series.size}"
            else:
                meta = f"xlsx:{xlsx_path.name}:{sh} col={col} n={series.size}"
            inc = series.fillna(1.0).to_numpy(dtype=float)
            return inc, meta
        raise RuntimeError(f"Failed to read a usable sheet/column from {xlsx_path}: {last_err}")

    raise FileNotFoundError("No CPI provided. Upload CSV or place cpi_factors.xlsx next to this script.")

@st.cache_data(show_spinner=False)
def five_year_chain(inc: np.ndarray, start_row_1: int) -> Tuple[np.ndarray, float]:
    """Return the 5 deflators (rows: start, +12, +24, +36, +48) and their product."""
    if start_row_1 < 1:
        start_row_1 = 1
    start_idx = start_row_1 - 1
    idxs = start_idx + (np.arange(5) * 12)
    if idxs[-1] >= inc.size:
        raise IndexError("Not enough rows in CPI series for 5-year chain from this start row.")
    vals = inc[idxs].astype(float)
    # Guard NaN or non-positive
    vals = np.where(np.isfinite(vals) & (vals > 0), vals, 1.0)
    prod = float(np.prod(vals))
    return vals, prod

# --- Load CPI ----------------------------------------------------------------

xlsx_path = Path(file_override) if file_override.strip() else DEFAULT_XLSX
try:
    inc, meta = load_deflators(xlsx_path=xlsx_path, csv_buf=csv_upload)
    st.success(f"Loaded CPI deflator → {meta}")
except Exception as e:
    st.error(str(e))
    st.stop()

# --- Compute & Show ----------------------------------------------------------

try:
    vals, prod = five_year_chain(inc, int(start_row_1_based))
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Five 12‑month‑spaced factors")
fcols = st.columns(5)
for i, v in enumerate(vals):
    with fcols[i]:
        st.metric(label=f"Year {i+1}", value=f"{v:.6f}")

st.markdown("---")
st.metric(label="Product of 5 factors", value=f"{prod:.9f}")
st.metric(label="Sticker at year 5", value=f"${base_price * prod:,.5f}")

with st.expander("Show indices used"):
    start_idx = int(start_row_1_based) - 1
    st.code(f"rows: {start_idx}, {start_idx+12}, {start_idx+24}, {start_idx+36}, {start_idx+48}")

st.caption("Tip: If your first five are 0.9885, 0.9884, 0.9941, 0.9053, 0.8954, the product should be 0.787356322 and a $35,000 car becomes $27,557.47 in year 5.")