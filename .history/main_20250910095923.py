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
thinking_spend = st.number_input("Thinking of Spending ($)", min_value=0, value=15000, step=500)
whatif_spend = st.number_input("What if I Spend This Instead ($)", min_value=0, value=5000, step=500)

years = retirement_age - current_age
if years <= 0:
    st.error("Retirement age must be greater than current age.")
    st.stop()

spend_diff = thinking_spend - whatif_spend
if spend_diff <= 0:
    st.info("No opportunity cost: the 'Thinking of Spending' amount must be greater than the 'What if I Spend This Instead' amount.")

dataset = st.selectbox("Choose dataset(s)", ["Global", "SPX", "Both"])

if dataset in ("Global", "SPX"):
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
        # Compute ending values for each sim (fv_multiple × spend_diff)
        ending_values = sims["fv_multiple"] * max(spend_diff, 0)

        min_val = float(ending_values.min())
        med_val = float(ending_values.median())

        st.subheader(f"{dataset} Results — Opportunity Cost of the Difference")
        st.metric("Minimum Ending Value of Difference", f"${min_val:,.0f}")
        st.metric("Median Ending Value of Difference", f"${med_val:,.0f}")

else:
    # BOTH
    try:
        df_glob = load_factors("global_factors.csv")
        df_spx  = load_factors("spx_factors.csv")
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

    colA, colB = st.columns(2)
    with colA:
        alloc_glob = st.selectbox("Global allocation column", [c for c in df_glob.columns], key="alloc_glob")
    with colB:
        alloc_spx = st.selectbox("SPX allocation column", [c for c in df_spx.columns], key="alloc_spx")

    # Build simulations for both
    sims_glob = build_windows(df_glob, alloc_glob, years)
    sims_spx  = build_windows(df_spx,  alloc_spx,  years)

    if sims_glob.empty and sims_spx.empty:
        st.warning(f"No {years}-year windows available in either dataset.")
    else:
        if not sims_glob.empty:
            end_glob = sims_glob["fv_multiple"] * max(spend_diff, 0)
            g_min = float(end_glob.min())
            g_med = float(end_glob.median())
        else:
            g_min = g_med = None

        if not sims_spx.empty:
            end_spx = sims_spx["fv_multiple"] * max(spend_diff, 0)
            s_min = float(end_spx.min())
            s_med = float(end_spx.median())
        else:
            s_min = s_med = None

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Global — Opportunity Cost of the Difference")
            if g_min is None:
                st.info(f"No {years}-year windows.")
            else:
                st.metric("Minimum Ending Value of Difference", f"${g_min:,.0f}")
                st.metric("Median Ending Value of Difference", f"${g_med:,.0f}")
        with col2:
            st.subheader("SPX — Opportunity Cost of the Difference")
            if s_min is None:
                st.info(f"No {years}-year windows.")
            else:
                st.metric("Minimum Ending Value of Difference", f"${s_min:,.0f}")
                st.metric("Median Ending Value of Difference", f"${s_med:,.0f}")