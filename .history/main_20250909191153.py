import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re
import altair as alt

st.set_page_config(page_title="30-Year Opportunity Cost — Simple", layout="wide")

# ---------------------------
# Minimal factor loading
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
    # Build lowercase map once
    lc = {c: str(c).strip().lower() for c in df.columns}
    # Strong preferences
    preferred_exact = [
        "begin month", "begin_month", "begin date", "begin_date",
        "start month", "start_month", "start date", "start_date",
        "begin", "start"
    ]
    for key in preferred_exact:
        for c in df.columns:
            if lc[c] == key:
                return c
    # Fuzzy: contains begin/start and month/date
    for c in df.columns:
        s = lc[c]
        if (("begin" in s) or ("start" in s)) and (("month" in s) or ("date" in s)):
            return c
    # Standard fallbacks
    for c in df.columns:
        if lc[c] in ("date", "month", "period", "timestamp"):
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
    if p.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(p)
    else:
        raw = pd.read_csv(p)
    return _normalize(raw, label)


def alloc_options(df: pd.DataFrame):
    return [c for c in df.columns if c != "date"]


def default_60e_idx(opts):
    if "60E" in opts:
        return opts.index("60E")
    nums = [int(o.replace("E", "")) for o in opts]
    return int(np.argmin(np.abs(np.array(nums) - 60)))

# ---------------------------
# Core math (annual 12‑month step)
# ---------------------------

def annual_window(df: pd.DataFrame, alloc: str, start_idx: int, years: int, step: int = 12) -> np.ndarray:
    arr = df[alloc].values.astype(float)
    idxs = start_idx + np.arange(years) * step
    if idxs[-1] >= len(arr):
        raise ValueError("Not enough data for this window.")
    w = arr[idxs]
    if np.isnan(w).any():
        raise ValueError("Missing factor(s) in this window.")
    return w


def multipliers_to_end(factors: np.ndarray) -> np.ndarray:
    cp = np.cumprod(factors)
    prev = np.concatenate(([1.0], cp[:-1]))
    return cp[-1] / prev


def fv_of_lump_sum(amount: float, factors: np.ndarray) -> float:
    payments = np.zeros(len(factors))
    payments[0] = amount  # invest at year 0
    mult = multipliers_to_end(factors)
    return float(np.dot(payments, mult))


def distribution_fv(df: pd.DataFrame, alloc: str, years: int, amount: float) -> pd.DataFrame:
    step = 12
    arr = df[alloc].values.astype(float)
    max_start = len(arr) - years * step
    rows = []
    if max_start < 0:
        return pd.DataFrame(columns=["start_date", "fv"])
    for s in range(0, max_start + 1):
        try:
            f = annual_window(df, alloc, s, years, step)
        except Exception:
            continue
        fv = fv_of_lump_sum(amount, f)
        rows.append((df.loc[s, "date"], fv))
    return pd.DataFrame(rows, columns=["start_date", "fv"]).dropna()


def first_valid_window_start_date(df: pd.DataFrame, alloc: str, years: int, step: int = 12):
    arr = df[alloc].values.astype(float)
    max_start = len(arr) - years * step
    if max_start < 0:
        return None
    for s in range(0, max_start + 1):
        idxs = s + np.arange(years) * step
        if idxs[-1] >= len(arr):
            break
        w = arr[idxs]
        if not np.isnan(w).any():
            return df.loc[s, "date"]
    return None

# ---------------------------
# UI — simple and focused
# ---------------------------

st.title("Simple 30‑Year Future Value of Today's Spend")

with st.sidebar:
    dataset = st.radio("Dataset", ["Global (LBM)", "SP500 (SPX)"], index=0)
    if dataset == "Global (LBM)":
        factors = load_factors("global_factors.csv", "Global")
    else:
        factors = load_factors("spx_factors.csv", "SP500")

    opts = alloc_options(factors)
    alloc = st.selectbox(f"Allocation — {factors.attrs.get('dataset_label','')}", opts, index=default_60e_idx(opts))

    amount = st.slider("Amount to spend/invest today ($)", min_value=0, max_value=200_000, value=10_000, step=1_000)

    YEARS = 30
    st.caption("Horizon fixed at 30 years (annual 12‑month steps).")

# Compute immediately (no button) for simplicity
try:
    dist = distribution_fv(factors, alloc, YEARS, float(amount))
    if dist.empty:
        st.warning("Not enough history for a 30‑year horizon.")
    else:
        p10, p50, p90 = np.percentile(dist["fv"], [10, 50, 90])
        min_fv, max_fv = float(dist["fv"].min()), float(dist["fv"].max())
        count = len(dist)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Start periods", f"{count:,}")
        c2.metric("Median FV", f"${p50:,.0f}")
        c3.metric("P10 / P90", f"${p10:,.0f} / ${p90:,.0f}")
        c4.metric("Min / Max", f"${min_fv:,.0f} / ${max_fv:,.0f}")

        # Chart
        min_all = pd.to_datetime(factors["date"]).min()
        max_all = pd.to_datetime(factors["date"]).max()
        chart = alt.Chart(dist).mark_line().encode(
            x=alt.X("start_date:T", title="Start Date", scale=alt.Scale(domain=[min_all, max_all])),
            y=alt.Y("fv:Q", title=f"FV of ${amount:,} @ 30 years (real $)"),
            tooltip=["start_date", "fv"],
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)

        with st.expander("Data check (dates & coverage)", expanded=False):
            st.write("Dataset min date:", pd.to_datetime(factors["date"]).min().date())
            st.write("Dataset max date:", pd.to_datetime(factors["date"]).max().date())
            st.write(f"Non‑null observations for {alloc}:", int(factors[alloc].notna().sum()))
            fv_start = first_valid_window_start_date(factors, alloc, YEARS, step=12)
            st.write(f"First valid 30‑year window start for {alloc}:", pd.to_datetime(fv_start).date() if fv_start is not None else "None")

        # All simulations table (with multiples)
        dist_table = dist.copy()
        dist_table["multiple"] = dist_table["fv"] / float(amount)
        show_table = st.checkbox("Show all simulations (table)", value=True)
        if show_table:
            st.dataframe(dist_table.sort_values("start_date").reset_index(drop=True), use_container_width=True)

        st.download_button(
            "Download (CSV)",
            data=dist_table.to_csv(index=False).encode("utf-8"),
            file_name=f"fv_{int(amount)}_30y_{alloc}.csv",
            mime="text/csv",
        )
except Exception as e:
    st.error(f"{type(e).__name__}: {e}")
