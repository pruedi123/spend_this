import streamlit as st
import pandas as pd
import numpy as np
import re

def _canonical_alloc_name(col: str) -> str | None:
    """
    Map header variants to canonical equity labels like '0E','10E',...,'100E'.
    Rules:
      - '100F' (100% fixed) -> '0E'
      - detect digits next to 'E' (case-insensitive): 'LBM 60E', 'spx60e' -> '60E'
      - fallback: trailing digits near known prefixes -> '<N>E'
      - accept exact 'NE' already
    """
    s_raw = str(col).strip()
    s = s_raw.lower().replace("_", "").replace(" ", "")
    # exact like '60e'
    m_exact = re.fullmatch(r"(\d{1,3})e", s)
    if m_exact:
        pct = max(0, min(100, int(m_exact.group(1))))
        return f"{pct}E"
    # 100F => 0E
    if s.endswith("100f"):
        return "0E"
    # find number followed by e at end: 'lbm60e'
    m = re.search(r"(\d{1,3})e$", s)
    if m:
        pct = max(0, min(100, int(m.group(1))))
        return f"{pct}E"
    # trailing digits with known prefixes: 'lbm60', 'spx60'
    m2 = re.search(r"(\d{1,3})$", s)
    if m2 and any(prefix in s for prefix in ("lbm", "spx", "glob", "global")):
        pct = max(0, min(100, int(m2.group(1))))
        return f"{pct}E"
    return None

def load_factors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure all numeric cols are numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Build a canonicalized frame: map columns to 0E..100E when possible
    canon_cols = {}
    for c in df.columns:
        canon = _canonical_alloc_name(c)
        if canon is None:
            continue
        # keep first occurrence if duplicates
        if canon not in canon_cols:
            canon_cols[canon] = df[c]
    out = pd.DataFrame(canon_cols)
    # Drop empty cols and sort columns by equity percent if present
    out = out.dropna(axis=1, how="all")
    if out.empty:
        # fallback: return numeric df as-is (last resort)
        df = df.dropna(axis=1, how="all")
        return df
    # sort like 0E..100E based on the number
    def _k(name: str):
        try:
            return int(str(name).replace("E",""))
        except Exception:
            return 999
    out = out.reindex(sorted(out.columns, key=_k), axis=1)
    return out

def build_windows(df: pd.DataFrame, alloc_col: str, years: int, step: int = 12, fee_mult_per_step: float = 1.0) -> pd.DataFrame:
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
        # Apply annual expense as a per-step multiplicative drag
        window = window * fee_mult_per_step
        fv = float(np.prod(window))  # product of adjusted factors
        rows.append((s, window, fv))
    return pd.DataFrame(rows, columns=["start_index","factors","fv_multiple"])

# ---------------------------
# UI
# ---------------------------

st.title("Spend This — Opportunity Cost Calculator")

# Inputs
current_age = st.number_input("Current Age", min_value=0, max_value=120, value=30)
retirement_age = st.number_input("Retirement Age", min_value=0, max_value=120, value=65)
retirement_years = st.slider("Number of Years in Retirement", 20, 35, 30, 1)
thinking_spend = st.number_input("Thinking of Spending ($)", min_value=0, value=15000, step=500)
whatif_spend = st.number_input("What if I Spend This Instead ($)", min_value=0, value=5000, step=500)

# Fixed annual expense drags: Global = 20 bps, SPX = 5 bps
_default_step = 12
fee_mult_per_step_glob = (1.0 - 0.0020) ** (_default_step / 12.0)  # 20 bps per year
fee_mult_per_step_spx  = (1.0 - 0.0005) ** (_default_step / 12.0)  # 5 bps per year

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

# Load withdrawals tables (for multiplying highest opportunity-cost results by withdrawal rules)
try:
    df_withdrawals = pd.read_csv("withdrawals.csv")
except Exception:
    df_withdrawals = None
try:
    df_withdrawals_spx = pd.read_csv("withdrawals_spx.csv")
except Exception:
    df_withdrawals_spx = None

# Determine common allocation columns (exclude any non-numeric leftovers if present)
allocs_glob = [c for c in df_glob.columns]
allocs_spx  = [c for c in df_spx.columns]
def _order_key(x: str):
    try:
        return int(str(x).replace("E",""))
    except Exception:
        return -1  # unknowns go to the end
# Sort descending so it starts with 100E, 90E, ..., 0E
common_allocs = sorted(set(allocs_glob).intersection(allocs_spx), key=_order_key, reverse=True)

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
    sims_g = build_windows(df_glob, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_glob)
    if sims_g.empty:
        g_min = g_med = None
    else:
        end_g = sims_g["fv_multiple"] * spend_diff
        g_min = float(end_g.min())
        g_med = float(end_g.median())
    # SPX
    sims_s = build_windows(df_spx, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_spx)
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
st.caption("This model assumes a fixed annual expense ratio of 0.20% for Global and 0.05% for SP500 portfolios.")
st.dataframe(result_df, use_container_width=True)

# Also provide a raw numeric CSV (without $ formatting) for download
raw_rows = []
for alloc in common_allocs:
    sims_g = build_windows(df_glob, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_glob)
    sims_s = build_windows(df_spx, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_spx)
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

# ===========================
# Withdrawal-adjusted results
# ===========================
def _lookup_withdrawal_row(df_w: pd.DataFrame | None, yrs: int):
    if df_w is None:
        return None
    try:
        row = df_w.loc[df_w["Years"] == yrs].iloc[0]
        return {
            "Min": float(row["Min"]),
            "Median": float(row["Median"]),
        }
    except Exception:
        return None

# Compute the highest (max) results across allocations for each dataset
global_best_min = float(np.nanmax(raw_df["Global_Min"].values)) if not raw_df.empty else np.nan
global_best_med = float(np.nanmax(raw_df["Global_Median"].values)) if not raw_df.empty else np.nan
spx_best_min    = float(np.nanmax(raw_df["SPX_Min"].values)) if not raw_df.empty else np.nan
spx_best_med    = float(np.nanmax(raw_df["SPX_Median"].values)) if not raw_df.empty else np.nan

w_g = _lookup_withdrawal_row(df_withdrawals, retirement_years)
w_s = _lookup_withdrawal_row(df_withdrawals_spx, retirement_years)

# Build separate tables for Global and SPX if we have the needed inputs
if np.isfinite(global_best_min) and w_g is not None:
    st.subheader("Withdrawal‑Adjusted Results — Global")
    g_table = pd.DataFrame([{
        "Retirement Years": retirement_years,
        "Highest Global Min (raw $)": global_best_min,
        "Highest Global Median (raw $)": global_best_med,
        "Withdrawal Min (from withdrawals.csv)": w_g["Min"],
        "Withdrawal Median (from withdrawals.csv)": w_g["Median"],
        "Global Min × Withdrawal Min": global_best_min * w_g["Min"],
        "Global Median × Withdrawal Median": global_best_med * w_g["Median"],
    }])
    # Display formatted
    g_fmt = g_table.copy()
    for col in ["Highest Global Min (raw $)", "Highest Global Median (raw $)", "Global Min × Withdrawal Min", "Global Median × Withdrawal Median"]:
        g_fmt[col] = g_fmt[col].map(lambda v: f"${v:,.0f}")
    g_fmt["Withdrawal Min (from withdrawals.csv)"] = g_fmt["Withdrawal Min (from withdrawals.csv)"].map(lambda v: f"{v:,.4f}")
    g_fmt["Withdrawal Median (from withdrawals.csv)"] = g_fmt["Withdrawal Median (from withdrawals.csv)"].map(lambda v: f"{v:,.4f}")
    st.dataframe(g_fmt, use_container_width=True)
    st.download_button(
        "Download Global withdrawal‑adjusted (CSV)",
        data=g_table.to_csv(index=False).encode("utf-8"),
        file_name=f"spend_this_withdrawal_adjusted_global_{retirement_years}y.csv",
        mime="text/csv"
    )

if np.isfinite(spx_best_min) and w_s is not None:
    st.subheader("Withdrawal‑Adjusted Results — SPX")
    s_table = pd.DataFrame([{
        "Retirement Years": retirement_years,
        "Highest SPX Min (raw $)": spx_best_min,
        "Highest SPX Median (raw $)": spx_best_med,
        "Withdrawal Min (from withdrawals_spx.csv)": w_s["Min"],
        "Withdrawal Median (from withdrawals_spx.csv)": w_s["Median"],
        "SPX Min × Withdrawal Min": spx_best_min * w_s["Min"],
        "SPX Median × Withdrawal Median": spx_best_med * w_s["Median"],
    }])
    s_fmt = s_table.copy()
    for col in ["Highest SPX Min (raw $)", "Highest SPX Median (raw $)", "SPX Min × Withdrawal Min", "SPX Median × Withdrawal Median"]:
        s_fmt[col] = s_fmt[col].map(lambda v: f"${v:,.0f}")
    s_fmt["Withdrawal Min (from withdrawals_spx.csv)"] = s_fmt["Withdrawal Min (from withdrawals_spx.csv)"].map(lambda v: f"{v:,.4f}")
    s_fmt["Withdrawal Median (from withdrawals_spx.csv)"] = s_fmt["Withdrawal Median (from withdrawals_spx.csv)"].map(lambda v: f"{v:,.4f}")
    st.dataframe(s_fmt, use_container_width=True)
    st.download_button(
        "Download SPX withdrawal‑adjusted (CSV)",
        data=s_table.to_csv(index=False).encode("utf-8"),
        file_name=f"spend_this_withdrawal_adjusted_spx_{retirement_years}y.csv",
        mime="text/csv"
    )