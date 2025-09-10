import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Global vs SPX 30-Year Sims", layout="wide")

# ---------------------------
# Functions
# ---------------------------

def load_factors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure all numeric cols are numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop empty cols
    df = df.dropna(axis=1, how="all")
    return df

def build_windows(df: pd.DataFrame, alloc_col: str, years: int, step: int = 12) -> pd.DataFrame:
    arr = df[alloc_col].values.astype(float)
    max_start = len(arr) - years * step
    rows = []
    if max_start < 0:
        return pd.DataFrame(columns=["start_index","factors","fv_multiple"])
    for s in range(0, max_start + 1):
        idxs = s + np.arange(years) * step
        if idxs[-1] >= len(arr):
            break
        window = arr[idxs]
        if np.isnan(window).any():
            continue
        fv = float(np.prod(window))  # product of factors
        rows.append((s, window, fv))
    return pd.DataFrame(rows, columns=["start_index","factors","fv_multiple"])

# ---------------------------
# UI
# ---------------------------

# ---------------------------
# UI
# ---------------------------

st.title("Spend This — Opportunity Cost Calculator")

# Inputs
current_age = st.number_input("Current Age", min_value=0, max_value=120, value=30)
retirement_age = st.number_input("Retirement Age", min_value=0, max_value=120, value=65)
lump_spend = st.number_input("Lump Spend ($)", min_value=0, value=10000, step=1000)

years = retirement_age - current_age
if years <= 0:
    st.error("Retirement age must be greater than current age.")
    st.stop()

dataset = st.radio("Choose dataset", ["Global", "SPX"], horizontal=True)
path = "global_factors.csv" if dataset == "Global" else "spx_factors.csv"

try:
    df = load_factors(path)
except Exception as e:
    st.error(f"Error loading file {path}: {e}")
    st.stop()

alloc = st.selectbox("Choose allocation column", [c for c in df.columns])

# Build simulations
sims = build_windows(df, alloc, years)

if sims.empty:
    st.warning(f"No {years}-year windows available in {dataset}.")
else:
    # Compute ending values for each sim (fv_multiple × lump spend)
    ending_values = sims["fv_multiple"] * lump_spend

    min_val = float(ending_values.min())
    med_val = float(ending_values.median())

    st.metric("Minimum Ending Value", f"${min_val:,.0f}")
    st.metric("Median Ending Value", f"${med_val:,.0f}")