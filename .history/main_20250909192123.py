import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re
import altair as alt

st.set_page_config(page_title="30-Year Returns — Global vs SPX", layout="wide")

# ---------------------------
# Minimal loaders
# ---------------------------

def _parse_alloc(col: str):
    s = str(col).strip().lower()
    if "100f" in s:
        return "0E"
    m = re.search(r"(\d{1,3})\s*e", s)
    if m:
        pct = max(0, min(100, int(m.group(1))))
        return f"{pct}E"
    m2 = re.search(r"(\d{1,3})$", s)
    if m2:
        pct = max(0, min(100, int(m2.group(1))))
        return f"{pct}E"
    return None

def _pick_date_col(df: pd.DataFrame) -> str | None:
    lc = {c: str(c).strip().lower() for c in df.columns}
    # exact preferences first
    for key in ["begin month","begin_month","begin date","begin_date","start month","start_month","start date","start_date","begin","start"]:
        for c in df.columns:
            if lc[c] == key:
                return c
    # fuzzy
    for c in df.columns:
        s = lc[c]
        if (("begin" in s) or ("start" in s)) and (("month" in s) or ("date" in s)):
            return c
    # standard
    for c in df.columns:
        if lc[c] in ("date","month","period","timestamp"):
            return c
    return None

def _normalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    date_col = _pick_date_col(df)
    if date_col is None:
        df = df.copy()
        df.insert(0, "date", pd.date_range("1900-01-01", periods=len(df), freq="MS"))
    else:
        df = df.rename(columns={date_col: "date"}).copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    alloc_map = {}
    for c in df.columns:
        if c == "date":
            continue
        lab = _parse_alloc(c)
        if lab:
            alloc_map[c] = lab
    if not alloc_map:
        raise ValueError("No allocation columns detected (expect names like 'LBM 60E', 'spx60e', '...100F').")

    out = df[["date"] + list(alloc_map.keys())].rename(columns=alloc_map).copy()
    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    out.attrs["dataset_label"] = label
    return out

@st.cache_data(show_spinner=False)
def load_factors(file_name: str, label: str) -> pd.DataFrame:
    p = Path(".") / file_name
    if not p.exists():
        raise FileNotFoundError(f"Missing {file_name} in app root.")
    if p.suffix.lower() in (".xlsx",".xls"):
        raw = pd.read_excel(p)
    else:
        raw = pd.read_csv(p)
    return _normalize(raw, label)

def alloc_options(df: pd.DataFrame):
    return [c for c in df.columns if c != "date"]

def default_60e_idx(opts):
    if "60E" in opts:
        return opts.index("60E")
    nums = [int(o.replace("E","")) for o in opts]
    return int(np.argmin(np.abs(np.array(nums) - 60)))

# ---------------------------
# Core math — annual (12-mo) windows
# ---------------------------

def annual_window(arr: np.ndarray, start_idx: int, years: int, step: int = 12) -> np.ndarray:
    idxs = start_idx + np.arange(years) * step
    if idxs[-1] >= len(arr):
        raise ValueError("Not enough data for this window.")
    w = arr[idxs].astype(float)
    if np.isnan(w).any():
        raise ValueError("Missing factor(s) in this window.")
    return w

def fv_of_1(factors: np.ndarray) -> float:
    # invest $1 at year 0; FV after all factors
    # equivalent to product of factors
    return float(np.prod(factors))

def distribution_fv(df: pd.DataFrame, alloc: str, years: int) -> pd.DataFrame:
    step = 12
    arr = df[alloc].values.astype(float)
    max_start = len(arr) - years * step
    rows = []
    if max_start < 0:
        return pd.DataFrame(columns=["start_date","fv"])
    for s in range(0, max_start + 1):
        try:
            w = annual_window(arr, s, years, step)
        except Exception:
            continue
        rows.append((df.loc[s, "date"], fv_of_1(w)))
    return pd.DataFrame(rows, columns=["start_date","fv"]).dropna()

# ---------------------------
# UI
# ---------------------------

st.title("30‑Year Returns — Global vs SPX (All Start Periods)")

YEARS = 30

try:
    df_glob = load_factors("global_factors.csv", "Global")
    df_spx  = load_factors("spx_factors.csv", "SP500")
except Exception as e:
    st.error(f"{type(e).__name__}: {e}")
    st.stop()

# Allocation selector (must exist in both)
common_allocs = sorted(set(alloc_options(df_glob)).intersection(alloc_options(df_spx)), key=lambda x: int(x.replace("E","")))
if not common_allocs:
    st.error("No common allocation columns between global_factors.csv and spx_factors.csv.")
    st.stop()

alloc = st.selectbox("Allocation (applies to both files)", common_allocs, index=default_60e_idx(common_allocs))

# Compute distributions
dist_glob = distribution_fv(df_glob, alloc, YEARS)
dist_spx  = distribution_fv(df_spx,  alloc, YEARS)

# Align on shared start_date range
min_start = max(dist_glob["start_date"].min(), dist_spx["start_date"].min()) if not dist_glob.empty and not dist_spx.empty else None
max_start = min(dist_glob["start_date"].max(), dist_spx["start_date"].max()) if not dist_glob.empty and not dist_spx.empty else None

if dist_glob.empty or dist_spx.empty or min_start is None or max_start is None or min_start > max_start:
    st.warning("Insufficient overlapping history to compute 30‑year windows for both datasets at this allocation.")
    st.stop()

dist_glob = dist_glob[(dist_glob["start_date"] >= min_start) & (dist_glob["start_date"] <= max_start)].reset_index(drop=True)
dist_spx  = dist_spx[(dist_spx["start_date"]  >= min_start) & (dist_spx["start_date"]  <= max_start)].reset_index(drop=True)

# Merge for table/metrics
merged = pd.DataFrame({
    "start_date": dist_glob["start_date"],
    "fv_global": dist_glob["fv"].values,
    "fv_spx":    dist_spx["fv"].values
})
merged["delta_spx_minus_global"] = merged["fv_spx"] - merged["fv_global"]

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Periods (overlap)", f"{len(merged):,}")
col2.metric("Median FV — Global", f"{np.median(merged['fv_global']):,.2f}×")
col3.metric("Median FV — SPX",    f"{np.median(merged['fv_spx']):,.2f}×")

# Charts
c1, c2 = st.columns(2)

with c1:
    st.subheader("Global (LBM)")
    min_all = pd.to_datetime(df_glob["date"]).min()
    max_all = pd.to_datetime(df_glob["date"]).max()
    chart_g = alt.Chart(dist_glob).mark_line().encode(
        x=alt.X("start_date:T", title="Start Date", scale=alt.Scale(domain=[min_all, max_all])),
        y=alt.Y("fv:Q", title="FV of $1 @ 30 years (×)"),
        tooltip=["start_date","fv"]
    ).properties(height=300)
    st.altair_chart(chart_g, use_container_width=True)

with c2:
    st.subheader("SP500 (SPX)")
    min_all_s = pd.to_datetime(df_spx["date"]).min()
    max_all_s = pd.to_datetime(df_spx["date"]).max()
    chart_s = alt.Chart(dist_spx).mark_line().encode(
        x=alt.X("start_date:T", title="Start Date", scale=alt.Scale(domain=[min_all_s, max_all_s])),
        y=alt.Y("fv:Q", title="FV of $1 @ 30 years (×)"),
        tooltip=["start_date","fv"]
    ).properties(height=300)
    st.altair_chart(chart_s, use_container_width=True)

st.markdown("### Combined Table (Overlapping Windows)")
st.dataframe(merged, use_container_width=True)

st.download_button(
    "Download Combined CSV",
    data=merged.to_csv(index=False).encode("utf-8"),
    file_name=f"30y_returns_{alloc}_global_vs_spx.csv",
    mime="text/csv",
)

# Show begin-month diagnostics
with st.expander("Data check (min dates & coverage)"):
    st.write("Global — min date:", pd.to_datetime(df_glob["date"]).min().date(), "max:", pd.to_datetime(df_glob["date"]).max().date())
    st.write("SPX    — min date:", pd.to_datetime(df_spx["date"]).min().date(),  "max:", pd.to_datetime(df_spx["date"]).max().date())
    st.write(f"Non‑null (Global, {alloc}):", int(df_glob[alloc].notna().sum()))
    st.write(f"Non‑null (SPX, {alloc}):",    int(df_spx[alloc].notna().sum()))
