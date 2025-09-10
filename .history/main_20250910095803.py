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
        # Compute ending values separately for both scenarios
        end_thinking = sims["fv_multiple"] * thinking_spend
        end_whatif   = sims["fv_multiple"] * whatif_spend

        t_min = float(end_thinking.min())
        t_med = float(end_thinking.median())
        w_min = float(end_whatif.min())
        w_med = float(end_whatif.median())

        # Difference (thinking − what-if)
        d_min = t_min - w_min
        d_med = t_med - w_med

        st.subheader(f"{dataset} Results — **Compare Scenarios**")
        st.markdown(
            f"**Thinking:** ${thinking_spend:,.0f} • **What-if:** ${whatif_spend:,.0f} • Horizon: **{years} years**"
        )

        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown("**Thinking**")
            st.metric("Min Ending Value", f"**${t_min:,.0f}**")
            st.metric("Median Ending Value", f"**${t_med:,.0f}**")
        with colB:
            st.markdown("**What-if**")
            st.metric("Min Ending Value", f"**${w_min:,.0f}**")
            st.metric("Median Ending Value", f"**${w_med:,.0f}**")
        with colC:
            st.markdown("**Difference (Thinking − What-if)**")
            st.metric("Min Ending Value of Difference", f"**${d_min:,.0f}**")
            st.metric("Median Ending Value of Difference", f"**${d_med:,.0f}**")

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
            st.subheader("Global — **Compare Scenarios**")
            st.markdown(
                f"**Thinking:** ${thinking_spend:,.0f} • **What-if:** ${whatif_spend:,.0f} • Horizon: **{years} years**"
            )
            if sims_glob.empty:
                st.info(f"No {years}-year windows.")
            else:
                end_g_thinking = sims_glob["fv_multiple"] * thinking_spend
                end_g_whatif   = sims_glob["fv_multiple"] * whatif_spend
                g_t_min, g_t_med = float(end_g_thinking.min()), float(end_g_thinking.median())
                g_w_min, g_w_med = float(end_g_whatif.min()),   float(end_g_whatif.median())
                g_d_min, g_d_med = g_t_min - g_w_min, g_t_med - g_w_med

                gA, gB, gC = st.columns(3)
                with gA:
                    st.markdown("**Thinking**")
                    st.metric("Min Ending Value", f"**${g_t_min:,.0f}**")
                    st.metric("Median Ending Value", f"**${g_t_med:,.0f}**")
                with gB:
                    st.markdown("**What-if**")
                    st.metric("Min Ending Value", f"**${g_w_min:,.0f}**")
                    st.metric("Median Ending Value", f"**${g_w_med:,.0f}**")
                with gC:
                    st.markdown("**Difference (Thinking − What-if)**")
                    st.metric("Min Ending Value of Difference", f"**${g_d_min:,.0f}**")
                    st.metric("Median Ending Value of Difference", f"**${g_d_med:,.0f}**")

        with col2:
            st.subheader("SPX — **Compare Scenarios**")
            st.markdown(
                f"**Thinking:** ${thinking_spend:,.0f} • **What-if:** ${whatif_spend:,.0f} • Horizon: **{years} years**"
            )
            if sims_spx.empty:
                st.info(f"No {years}-year windows.")
            else:
                end_s_thinking = sims_spx["fv_multiple"] * thinking_spend
                end_s_whatif   = sims_spx["fv_multiple"] * whatif_spend
                s_t_min, s_t_med = float(end_s_thinking.min()), float(end_s_thinking.median())
                s_w_min, s_w_med = float(end_s_whatif.min()),   float(end_s_whatif.median())
                s_d_min, s_d_med = s_t_min - s_w_min, s_t_med - s_w_med

                sA, sB, sC = st.columns(3)
                with sA:
                    st.markdown("**Thinking**")
                    st.metric("Min Ending Value", f"**${s_t_min:,.0f}**")
                    st.metric("Median Ending Value", f"**${s_t_med:,.0f}**")
                with sB:
                    st.markdown("**What-if**")
                    st.metric("Min Ending Value", f"**${s_w_min:,.0f}**")
                    st.metric("Median Ending Value", f"**${s_w_med:,.0f}**")
                with sC:
                    st.markdown("**Difference (Thinking − What-if)**")
                    st.metric("Min Ending Value of Difference", f"**${s_d_min:,.0f}**")
                    st.metric("Median Ending Value of Difference", f"**${s_d_med:,.0f}**")