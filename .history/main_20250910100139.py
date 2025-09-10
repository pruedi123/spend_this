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

st.title("Spend This — Opportunity Cost Calculator")

# Inputs
current_age = st.number_input("Current Age", min_value=0, max_value=120, value=30)
retirement_age = st.number_input("Retirement Age", min_value=0, max_value=120, value=65)
thinking_spend = st.number_input("Thinking of Spending ($)", min_value=0, value=15000, step=500)
whatif_spend = st.number_input("What if I Spend This Instead ($)", min_value=0, value=5000, step=500)

years = retirement_age - current_age
if years <= 0:
    st.error("Retirement age must be greater than current age.")
    st.stop()

# Always load both datasets
try:
    df_glob = load_factors("global_factors.csv")
    df_spx  = load_factors("spx_factors.csv")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Determine common allocation columns (exclude any non-numeric leftovers if present)
allocs_glob = [c for c in df_glob.columns]
allocs_spx  = [c for c in df_spx.columns]
common_allocs = sorted(
    set(allocs_glob).intersection(allocs_spx),
    key=lambda x: (999 if not str(x).upper().endswith("E") else int(str(x)[:-1]) if str(x)[:-1].isdigit() else 999, str(x))
)

if not common_allocs:
    st.error("No common allocation columns between global_factors.csv and spx_factors.csv.")
    st.stop()

# Compute spend difference
spend_diff = thinking_spend - whatif_spend
if spend_diff <= 0:
    st.info("No opportunity cost: the 'Thinking of Spending' amount must be greater than the 'What if I Spend This Instead' amount.")
    spend_diff = 0.0

# Build results table
rows = []
for alloc in common_allocs:
    # Global
    sims_g = build_windows(df_glob, alloc, years)
    if sims_g.empty:
        g_min = g_med = None
    else:
        end_g = sims_g["fv_multiple"] * spend_diff
        g_min = float(end_g.min())
        g_med = float(end_g.median())
    # SPX
    sims_s = build_windows(df_spx, alloc, years)
    if sims_s.empty:
        s_min = s_med = None
    else:
        end_s = sims_s["fv_multiple"] * spend_diff
        s_min = float(end_s.min())
        s_med = float(end_s.median())

    rows.append({
        "Allocation": alloc,
        "Global Min": (None if g_min is None else f"${g_min:,.0f}"),
        "Global Median": (None if g_med is None else f"${g_med:,.0f}"),
        "SPX Min": (None if s_min is None else f"${s_min:,.0f}"),
        "SPX Median": (None if s_med is None else f"${s_med:,.0f}"),
    })

result_df = pd.DataFrame(rows)

st.subheader("Opportunity Cost of the Difference — Min & Median by Allocation")
st.markdown("**Thinking vs What-if difference invested across all historical windows**")
st.dataframe(result_df, use_container_width=True)

# Also provide a raw numeric CSV (without $ formatting) for download
raw_rows = []
for alloc in common_allocs:
    sims_g = build_windows(df_glob, alloc, years)
    sims_s = build_windows(df_spx, alloc, years)
    if sims_g.empty:
        g_min = g_med = np.nan
    else:
        end_g = sims_g["fv_multiple"] * spend_diff
        g_min = float(end_g.min()); g_med = float(end_g.median())
    if sims_s.empty:
        s_min = s_med = np.nan
    else:
        end_s = sims_s["fv_multiple"] * spend_diff
        s_min = float(end_s.min()); s_med = float(end_s.median())
    raw_rows.append({
        "Allocation": alloc,
        "Global_Min": g_min,
        "Global_Median": g_med,
        "SPX_Min": s_min,
        "SPX_Median": s_med,
    })
raw_df = pd.DataFrame(raw_rows)
st.download_button(
    "Download table (CSV)",
    data=raw_df.to_csv(index=False).encode("utf-8"),
    file_name=f"spend_this_min_median_{years}y.csv",
    mime="text/csv"
)