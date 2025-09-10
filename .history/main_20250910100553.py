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