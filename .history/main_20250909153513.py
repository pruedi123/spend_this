import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import re

st.set_page_config(page_title="Spend This, Not That", layout="wide")

# =============================================================================
# Utilities for loading & normalizing factor files from app root
# =============================================================================

def _parse_alloc(col: str):
    """
    Map column names to a canonical equity % label like '0E','10E',...,'100E'.
    Examples:
      'LBM 100E' -> '100E'
      'LBM 10E'  -> '10E'
      'LBM 100F' -> '0E'  (100% Fixed)
      'spx60e'   -> '60E'
      'Glob_0E'  -> '0E'
    """
    s = str(col).strip().lower()
    # 100F (100% fixed) => 0E
    if re.search(r'100\s*f', s):
        return '0E'
    # look for number followed by 'e'
    m = re.search(r'(\d{1,3})\s*e', s)
    if m:
        pct = int(m.group(1))
        pct = max(0, min(100, pct))
        return f'{pct}E'
    # fallback: trailing digits
    m2 = re.search(r'(\d{1,3})$', s)
    if m2:
        pct = int(m2.group(1))
        pct = max(0, min(100, pct))
        return f'{pct}E'
    return None


def _normalize_factors_df(df: pd.DataFrame, source_label: str):
    """
    - Ensure a 'date' column exists (fabricate monthly index if missing).
    - Keep only allocation columns, rename to canonical labels '0E'..'100E'.
    - Sort allocations by equity ascending (0E -> 100E).
    """
    cols = list(df.columns)
    # Find/ensure date column, preferring 'begin month' if present
    date_col = None
    # Prefer 'begin month' explicitly
    for c in cols:
        if str(c).strip().lower() == 'begin month':
            date_col = c
            break
    if date_col is None:
        # Fallback to other options
        for c in cols:
            cl = str(c).strip().lower()
            if cl in ('date', 'month', 'period', 'timestamp'):
                date_col = c
                break

    if date_col is None:
        df = df.copy()
        df.insert(0, 'date', pd.date_range('1900-01-01', periods=len(df), freq='MS'))
    else:
        df = df.rename(columns={date_col: 'date'}).copy()
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

    # Collect numeric allocation columns & map to canonical
    alloc_map = {}
    for c in df.columns:
        if c == 'date':
            continue
        lab = _parse_alloc(c)
        if lab is not None:
            alloc_map[c] = lab

    if not alloc_map:
        raise ValueError("No allocation columns detected (expected headers like 'LBM 100E', '...10E', '...100F', or 'spx60e').")

    keep = ['date'] + list(alloc_map.keys())
    out = df[keep].copy()
    out = out.rename(columns=alloc_map)

    # coerce to numeric
    for c in out.columns:
        if c != 'date':
            out[c] = pd.to_numeric(out[c], errors='coerce')

    # Do NOT drop rows globally; keep full history for correct start dates.
    # We'll handle NaNs per selected allocation at compute time.
    alloc_cols = [c for c in out.columns if c != 'date']
    out = out.sort_values('date').reset_index(drop=True)

    # sort allocations by equity percent (0E..100E)
    def _k(c):
        try:
            return int(c.replace('E', ''))
        except Exception:
            return 999
    ordered = sorted(alloc_cols, key=_k)
    out = out[['date'] + ordered]

    # attach label for UI
    out.attrs['dataset_label'] = source_label  # 'Global' or 'SP500'
    return out


def _inject_spx_dates(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """
    If loading global_factors.csv, align its rows with the begin month dates from spx_factors.csv.
    """
    if file_name.lower() != "global_factors.csv":
        return df
    spx_path = Path('.') / "spx_factors.csv"
    if not spx_path.exists():
        # If not found, fallback to default behavior
        return df
    # Load spx_factors.csv to get begin month dates
    if spx_path.suffix.lower() in ('.xlsx', '.xls'):
        spx_df = pd.read_excel(spx_path)
    else:
        spx_df = pd.read_csv(spx_path)
    # Try to find the begin month column
    date_col = None
    for c in spx_df.columns:
        if str(c).strip().lower() in ("begin month", "date", "month", "period", "timestamp"):
            date_col = c
            break
    if date_col is None:
        return df
    dates = pd.to_datetime(spx_df[date_col]).dt.tz_localize(None)
    # Align length
    if len(dates) != len(df):
        # If lengths don't match, fallback to default
        return df
    df = df.copy()
    df.insert(0, "date", dates)
    return df


@st.cache_data(show_spinner=False)
def load_factors_from_root(file_name: str, source_label: str):
    """
    Load CSV/Excel from project root (works on Streamlit Cloud if the file is in repo).
    Accepts wide formats with LBM/SPX columns; date optional.
    If loading global_factors.csv, align its dates with spx_factors.csv begin month.
    """
    p = Path('.') / file_name
    if not p.exists():
        raise FileNotFoundError(f"File not found in app root: {p}")
    if p.suffix.lower() in ('.xlsx', '.xls'):
        raw = pd.read_excel(p)
    else:
        raw = pd.read_csv(p)
    # If global and it lacks an explicit 'begin month', fallback to SPX date injection
    if source_label == "Global":
        lower_cols = [str(c).strip().lower() for c in raw.columns]
        if "begin month" not in lower_cols:
            raw = _inject_spx_dates(raw, file_name)
    return _normalize_factors_df(raw, source_label)


def find_alloc_options(df: pd.DataFrame):
    return [c for c in df.columns if c != 'date']


def default_60e_index(options):
    # pick '60E' when available; otherwise nearest
    if '60E' in options:
        return options.index('60E')
    # nearest by numeric distance
    def _num(x):
        try:
            return int(x.replace('E', ''))
        except Exception:
            return 0
    nums = [_num(o) for o in options]
    diffs = [abs(n - 60) for n in nums]
    return int(np.argmin(diffs)) if len(diffs) else 0


# =============================================================================
# Compounding helpers
# =============================================================================

def factors_window(df: pd.DataFrame, alloc: str, start_idx: int, months: int) -> np.ndarray:
    """Return a contiguous slice of monthly factors starting at start_idx of given length."""
    arr = df[alloc].values
    end_idx = start_idx + months
    if end_idx > len(arr):
        raise ValueError("Not enough data to cover the requested horizon.")
    return arr[start_idx:end_idx].astype(float)


def annual_factors_window(df: pd.DataFrame, alloc: str, start_idx: int, years: int, step_months: int = 12) -> np.ndarray:
    """
    Return an array of length `years` using ANNUAL 12-month-rolling factors,
    taking entries at start_idx, start_idx+12, start_idx+24, ...
    """
    arr = df[alloc].values.astype(float)
    idxs = start_idx + np.arange(years) * step_months
    if idxs[-1] >= len(arr):
        raise ValueError("Not enough data to cover the requested horizon with annual steps.")
    window = arr[idxs]
    if np.isnan(window).any():
        raise ValueError("Selected allocation has missing values in this period.")
    return window


def multipliers_to_end(factors: np.ndarray) -> np.ndarray:
    """Given an array of factors (length M), return vector m where m[t] = Π_{t..M-1} factors."""
    cp = np.cumprod(factors)
    prev = np.concatenate(([1.0], cp[:-1]))
    return cp[-1] / prev


def future_value_of_payments(payments: np.ndarray, factors: np.ndarray) -> float:
    """Sum_t payment[t] * Π_{t..end} factors. payments and factors must have the same length."""
    if len(payments) != len(factors):
        raise ValueError("payments and factors length mismatch")
    mult = multipliers_to_end(factors)
    return float(np.dot(payments, mult))


# =============================================================================
# Spending schedules
# =============================================================================

def schedule_one_time(months: int, at_month: int, amount: float) -> np.ndarray:
    p = np.zeros(months)
    if 0 <= at_month < months:
        p[at_month] = amount
    return p


def schedule_monthly(months: int, monthly_amount: float, start_month: int = 0, duration_months: int | None = None) -> np.ndarray:
    p = np.zeros(months)
    if duration_months is None:
        duration_months = months - start_month
    end = min(months, start_month + duration_months)
    p[start_month:end] = monthly_amount
    return p


def schedule_annual(months: int, annual_amount: float, start_month: int = 0, years: int | None = None) -> np.ndarray:
    p = np.zeros(months)
    total_years = years if years is not None else (months // 12)
    for y in range(total_years):
        m = start_month + 12 * y
        if m < months:
            p[m] = annual_amount
    return p


# YEAR-BASED schedules (ANNUAL step)
def schedule_one_time_years(years: int, at_year: int, amount: float) -> np.ndarray:
    p = np.zeros(years)
    if 0 <= at_year < years:
        p[at_year] = amount
    return p


def schedule_annual_years(years: int, annual_amount: float, start_year: int = 0, num_years: int | None = None) -> np.ndarray:
    p = np.zeros(years)
    total = num_years if num_years is not None else (years - start_year)
    for y in range(total):
        k = start_year + y
        if k < years:
            p[k] += annual_amount  # placed at END of year k
    return p


def schedule_monthly_to_annual(years: int, monthly_amount: float, start_year: int = 0, num_years: int | None = None) -> np.ndarray:
    """
    Aggregate monthly spending to annual (12 * monthly).
    Assumption: payment is treated as an END-OF-YEAR outflow (no within-year compounding).
    """
    return schedule_annual_years(years, monthly_amount * 12.0, start_year, num_years)


# =============================================================================
# Car depreciation & purchase modeling
# =============================================================================

def ddb_resale_value(purchase_price: float, years_held: int, useful_life_years: int = 10, residual_pct: float = 0.20) -> float:
    """
    Double-Declining Balance switching to Straight-Line down to a residual percent of original cost.
    Returns estimated resale/trade-in value at END of 'years_held'.
    """
    if years_held <= 0:
        return purchase_price

    salvage = purchase_price * residual_pct
    book = purchase_price
    rate = 2.0 / float(useful_life_years)

    for y in range(1, years_held + 1):
        ddb_dep = book * rate
        remaining_years = max(useful_life_years - (y - 1), 1)
        sl_dep = (book - salvage) / remaining_years
        dep = max(ddb_dep, sl_dep)
        book = max(book - dep, salvage)
    return float(book)


def build_car_purchase_events(
    base_price: float,
    interval_years: int,
    horizon_years: int,
    tax_rate: float = 0.07,
    fees: float = 1000.0,
    residual_pct: float = 0.20,
    useful_life_years: int = 10,
) -> pd.DataFrame:
    """
    Return a DataFrame of purchase events for a single pattern (ANNUAL timeline).
    Each event includes:
      - year (int)
      - new_price
      - tax_fees
      - trade_in
      - net_outflow (new_price + tax_fees - trade_in)
    Assumes buying NEW at t=0 (year 0), then every 'interval_years'.
    """
    events = []
    prev_purchase_price = None
    prev_purchase_year = None

    for y in range(0, horizon_years + 1, interval_years):
        if y > horizon_years:
            break
        new_price = float(base_price)
        tax_fees = new_price * float(tax_rate) + float(fees)

        # trade-in from previous car
        if prev_purchase_price is None:
            trade_in = 0.0
        else:
            years_held = max(y - prev_purchase_year, 0)
            trade_in = ddb_resale_value(prev_purchase_price, years_held, useful_life_years, residual_pct)

        net_outflow = new_price + tax_fees - trade_in
        events.append({
            "year": y,
            "new_price": new_price,
            "tax_fees": tax_fees,
            "trade_in": trade_in,
            "net_outflow": net_outflow,
        })

        prev_purchase_price = new_price
        prev_purchase_year = y

    return pd.DataFrame(events)


def car_opportunity_cost(
    df_factors: pd.DataFrame,
    alloc: str,
    start_idx: int,
    horizon_years: int,
    events: pd.DataFrame,
) -> float:
    """Compute FV-at-horizon of the car purchase net outflows given ANNUAL (12-month step) return factors."""
    years = horizon_years
    f = annual_factors_window(df_factors, alloc, start_idx, years, step_months=12)
    payments = np.zeros(years)
    for _, row in events.iterrows():
        y = int(row["year"])  # year offset from start
        if 0 <= y < years:
            payments[y] += float(row["net_outflow"])
    return future_value_of_payments(payments, f)


# =============================================================================
# Distributions across start periods
# =============================================================================

def rolling_opportunity_values_annual(
    df_factors: pd.DataFrame,
    alloc: str,
    horizon_years: int,
    payments_schedule_builder,
    step_months: int = 12,
) -> pd.DataFrame:
    """
    Run across all feasible start indices and compute FV-at-horizon for each start
    using ANNUAL (12-month step) factors.
    payments_schedule_builder(years: int) -> np.ndarray returns payments array per YEAR.
    """
    years = horizon_years
    arr = df_factors[alloc].values.astype(float)
    max_start = len(arr) - years * step_months
    results = []
    if max_start < 0:
        return pd.DataFrame(columns=["start_date", "fv"])

    for s in range(0, max_start + 1):
        try:
            f = annual_factors_window(df_factors, alloc, s, years, step_months=step_months)
        except Exception:
            continue
        payments = payments_schedule_builder(years)
        fv = future_value_of_payments(payments, f)
        results.append((df_factors.loc[s, "date"], fv))

    out = pd.DataFrame(results, columns=["start_date", "fv"]).dropna()
    return out


# =============================================================================
# UI
# =============================================================================

st.title("Spend This, Not That — Opportunity Cost Explorer")

with st.sidebar:
    st.subheader("Data source")
    dataset = st.radio(
        "Choose dataset",
        ["Global (LBM)", "SP500 (SPX)"],
        index=0,
        help="Reads CSVs from the app root: global_factors.csv or spx_factors.csv"
    )

    try:
        if dataset == "Global (LBM)":
            factors = load_factors_from_root("global_factors.csv", "Global")
        else:
            factors = load_factors_from_root("spx_factors.csv", "SP500")
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.stop()

    alloc_options = find_alloc_options(factors)
    alloc = st.selectbox(
        f"Allocation — {factors.attrs.get('dataset_label', '')}",
        alloc_options,
        index=default_60e_index(alloc_options)
    )

    st.markdown("---")
    st.subheader("Horizon & Start")
    horizon_years = st.slider("Horizon (years)", 5, 40, 30)

    mode = st.radio("Mode", ["Single start", "All start periods (distribution)"])

    if mode == "Single start":
        arr_len = len(factors)
        max_start = arr_len - horizon_years * 12
        if max_start < 0:
            st.error("Not enough history for that horizon (annual 12-month steps).")
            st.stop()
        idx = st.slider("Start index (0 = earliest)", 0, int(max_start), 0)
        start_date = factors.loc[idx, "date"].date()
        st.caption(f"Start date: {start_date}")
    else:
        idx = None


tab1, tab2 = st.tabs(["One-off / Recurring Spend", "Cars: Frugal vs Frequent Buyer"])

# ---- Tab 1: One-off / Recurring ----
with tab1:
    st.header("One-off / Recurring Spending")

    colA, colB, colC = st.columns(3)
    with colA:
        one_time_amount = st.number_input("One-time purchase ($)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
        one_time_year = st.number_input("When? (years from start)", min_value=0, value=0, step=1)
    with colB:
        monthly_amount = st.number_input("Monthly habit ($/mo)", min_value=0.0, value=0.0, step=10.0, format="%.2f")
        monthly_years = st.number_input("Monthly duration (years, 0 = through horizon)", min_value=0, value=0, step=1)
    with colC:
        annual_amount = st.number_input("Annual spend ($/yr)", min_value=0.0, value=0.0, step=100.0, format="%.2f")
        annual_years = st.number_input("Annual years (0 = through horizon)", min_value=0, value=0, step=1)

    run_btn = st.button("Compute Opportunity Cost")

    if run_btn:
        years = horizon_years

        def build_payments_years(Y: int) -> np.ndarray:
            p = np.zeros(Y)
            if one_time_amount > 0:
                p += schedule_one_time_years(Y, int(one_time_year), float(one_time_amount))
            if monthly_amount > 0:
                dur = None if monthly_years == 0 else int(monthly_years)
                p += schedule_monthly_to_annual(Y, float(monthly_amount), 0, dur)
            if annual_amount > 0:
                yrs = None if annual_years == 0 else int(annual_years)
                p += schedule_annual_years(Y, float(annual_amount), 0, yrs)
            return p

        if mode == "Single start":
            f = annual_factors_window(factors, alloc, idx, years, step_months=12)
            payments = build_payments_years(years)
            fv = future_value_of_payments(payments, f)

            st.subheader("Results (Single Start)")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Spent (real $)", f"${payments.sum():,.0f}")
            with m2:
                st.metric("Opportunity Cost @ Horizon (real $)", f"${fv:,.0f}")
            with m3:
                if payments.sum() > 0:
                    ratio = fv / payments.sum()
                    st.metric("Multiple of Outlay", f"{ratio:,.2f}×")
                else:
                    st.metric("Multiple of Outlay", "—")

            # Show payment schedule and chart (annual)
            pay_df = pd.DataFrame({"year_from_start": np.arange(years), "amount": payments})
            st.dataframe(pay_df[pay_df["amount"] > 0], use_container_width=True)
            chart = alt.Chart(pay_df[pay_df["amount"] > 0]).mark_bar().encode(
                x=alt.X("year_from_start:Q", title="Year from Start"), y=alt.Y("amount:Q"), tooltip=["year_from_start", "amount"]
            ).properties(height=240)
            st.altair_chart(chart, use_container_width=True)

            # download
            csv_bytes = pay_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download payment schedule (CSV)", data=csv_bytes, file_name="spend_schedule_annual.csv", mime="text/csv")

        else:
            dist_df = rolling_opportunity_values_annual(factors, alloc, horizon_years, build_payments_years)
            if len(dist_df) == 0:
                st.warning("Not enough history to compute the distribution for this horizon.")
            else:
                p10, p50, p90 = np.percentile(dist_df["fv"], [10, 50, 90])
                min_fv = np.min(dist_df["fv"])
                max_fv = np.max(dist_df["fv"])
                st.subheader("Results (All Start Periods)")
                m1, m2, m3 = st.columns(3)
                m1.metric("P10", f"${p10:,.0f}")
                m2.metric("Median", f"${p50:,.0f}")
                m3.metric("P90", f"${p90:,.0f}")
                n1, n2 = st.columns(2)
                n1.metric("Min", f"${min_fv:,.0f}")
                n2.metric("Max", f"${max_fv:,.0f}")

                dist_chart = alt.Chart(dist_df).mark_line().encode(
                    x=alt.X("start_date:T", title="Start Date"),
                    y=alt.Y("fv:Q", title="FV @ Horizon (real $)"),
                    tooltip=["start_date", "fv"]
                ).properties(height=280)
                st.altair_chart(dist_chart, use_container_width=True)

                st.dataframe(dist_df.tail(20), use_container_width=True)

                # download
                csv_bytes = dist_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download distribution (CSV)", data=csv_bytes, file_name="opportunity_cost_distribution_annual.csv", mime="text/csv")


# ---- Tab 2: Cars ----
with tab2:
    st.header("Cars: Frugal vs Frequent Buyer")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Frugal Owner")
        frugal_price = st.number_input("New car price ($)", min_value=0.0, value=30000.0, step=500.0)
        frugal_interval = st.selectbox("Replace every (years)", [8, 9, 10], index=2)
    with col2:
        st.subheader("Frequent Buyer")
        frequent_price_mode = st.radio("Price mode", ["Multiple of frugal", "Explicit price"], index=0)
        if frequent_price_mode == "Multiple of frugal":
            price_multiple = st.slider("Multiple", 1.0, 3.0, 2.0, 0.1)
            frequent_price = frugal_price * price_multiple
        else:
            frequent_price = st.number_input("New car price ($)", min_value=0.0, value=60000.0, step=500.0)
        frequent_interval = st.selectbox("Replace every (years)", [3, 4, 5], index=0)

    st.markdown("---")
    colX, colY, colZ, colW = st.columns(4)
    with colX:
        tax_rate = st.number_input("Sales tax rate", min_value=0.0, max_value=0.25, value=0.07, step=0.01)
    with colY:
        fees = st.number_input("Fees per purchase ($)", min_value=0.0, value=1000.0, step=50.0)
    with colZ:
        residual_pct = st.slider("Residual % of original (salvage)", 0.05, 0.40, 0.20, 0.01)
    with colW:
        useful_life = st.slider("Useful life (yrs)", 5, 15, 10, 1)

    run_cars = st.button("Compare Car Patterns")

    if run_cars:
        try:
            frugal_events = build_car_purchase_events(
                frugal_price, frugal_interval, horizon_years, tax_rate, fees, residual_pct, useful_life
            )
            frequent_events = build_car_purchase_events(
                frequent_price, frequent_interval, horizon_years, tax_rate, fees, residual_pct, useful_life
            )

            months = horizon_years * 12
            if mode == "Single start":
                fv_frugal = car_opportunity_cost(factors, alloc, idx, horizon_years, frugal_events)
                fv_freq = car_opportunity_cost(factors, alloc, idx, horizon_years, frequent_events)

                st.subheader("Results (Single Start)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Frugal: Total Outlay", f"${frugal_events['net_outflow'].sum():,.0f}")
                c2.metric("Frequent: Total Outlay", f"${frequent_events['net_outflow'].sum():,.0f}")
                c3.metric("Outlay Difference", f"${(frequent_events['net_outflow'].sum() - frugal_events['net_outflow'].sum()):,.0f}")

                d1, d2, d3 = st.columns(3)
                d1.metric("Frugal: Opportunity Cost @ Horizon", f"${fv_frugal:,.0f}")
                d2.metric("Frequent: Opportunity Cost @ Horizon", f"${fv_freq:,.0f}")
                d3.metric("Delta (Frequent − Frugal)", f"${(fv_freq - fv_frugal):,.0f}")

            else:
                # Distribution across all starts (annual step)
                arr = factors[alloc].values.astype(float)
                years = horizon_years
                max_start = len(arr) - years * 12
                fvs_frugal = []
                fvs_freq = []
                dates = []
                for s in range(0, max_start + 1):
                    f_slice = annual_factors_window(factors, alloc, s, years, step_months=12)

                    pay_fru = np.zeros(years)
                    for _, r in frugal_events.iterrows():
                        y = int(r["year"])
                        if 0 <= y < years:
                            pay_fru[y] += float(r["net_outflow"])

                    pay_freq = np.zeros(years)
                    for _, r in frequent_events.iterrows():
                        y = int(r["year"])
                        if 0 <= y < years:
                            pay_freq[y] += float(r["net_outflow"])

                    fvs_frugal.append(future_value_of_payments(pay_fru, f_slice))
                    fvs_freq.append(future_value_of_payments(pay_freq, f_slice))
                    dates.append(factors.loc[s, "date"])

                comp = pd.DataFrame({
                    "start_date": dates,
                    "fv_frugal": fvs_frugal,
                    "fv_frequent": fvs_freq,
                    "delta": np.array(fvs_freq) - np.array(fvs_frugal),
                })

                p10, p50, p90 = np.percentile(comp["delta"], [10, 50, 90])
                st.subheader("Results (All Start Periods)")
                e1, e2, e3 = st.columns(3)
                e1.metric("P10 Δ (Frequent − Frugal)", f"${p10:,.0f}")
                e2.metric("Median Δ", f"${p50:,.0f}")
                e3.metric("P90 Δ", f"${p90:,.0f}")

                line = alt.Chart(comp).mark_line().encode(
                    x=alt.X("start_date:T", title="Start Date"),
                    y=alt.Y("delta:Q", title="Δ FV @ Horizon (Frequent − Frugal)"),
                    tooltip=["start_date", "delta"]
                ).properties(height=280)
                st.altair_chart(line, use_container_width=True)

                # download
                csv_bytes = comp.to_csv(index=False).encode("utf-8")
                st.download_button("Download car comparison distribution (CSV)", data=csv_bytes, file_name="car_comparison_distribution_annual.csv", mime="text/csv")

            st.markdown("### Event Tables")
            st.caption("Values are in real dollars if your input factors are real returns.")
            st.dataframe(frugal_events, use_container_width=True)
            st.dataframe(frequent_events, use_container_width=True)

            # downloads
            st.download_button("Download frugal events (CSV)", data=frugal_events.to_csv(index=False).encode("utf-8"), file_name="frugal_events.csv", mime="text/csv")
            st.download_button("Download frequent events (CSV)", data=frequent_events.to_csv(index=False).encode("utf-8"), file_name="frequent_events.csv", mime="text/csv")

        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")


st.markdown("---")
st.caption(
    "Tip: Put 'global_factors.csv' and/or 'spx_factors.csv' in the app root. "
    "Each row should be a beginning-month ANNUAL (12-month) real return factor for each allocation. "
    "Monthly spends are aggregated to annual (12×monthly) and treated as end-of-year outflows by default. "
    "Opportunity cost is computed as the future value at your chosen horizon using annual 12-month steps."
)
