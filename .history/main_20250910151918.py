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

def annuity_fv_from_window(window: np.ndarray, contrib: float, timing: str = "end") -> float:
    """
    Future value of contributing `contrib` each year over `window` (array of per‑year factors).
    timing="end": deposit at end of each year (grows for remaining years only).
    timing="begin": deposit at beginning of each year (grows including the current year).
    """
    try:
        c = float(contrib)
    except Exception:
        return 0.0
    if c <= 0:
        return 0.0
    f = np.asarray(window, dtype=float)
    n = f.size
    if n == 0:
        return 0.0
    # Precompute suffix products
    rev_cum = np.cumprod(f[::-1])
    suffix_including = rev_cum[::-1]                # product f[i:]  (for beginning-of-year)
    # For end-of-year, exclude the current year's factor: f[i+1:]
    suffix_excluding = np.concatenate([suffix_including[1:], [1.0]])
    if timing == "begin":
        return float(c * np.sum(suffix_including))
    else:
        return float(c * np.sum(suffix_excluding))


# --- Auto Strategy helpers ---
def pmt(principal: float, apr_pct: float, years_term: int) -> float:
    """Monthly payment for an amortizing loan."""
    P = float(max(0.0, principal))
    n = int(max(1, years_term)) * 12
    r = float(apr_pct) / 100.0 / 12.0
    if P <= 0:
        return 0.0
    if r == 0.0:
        return P / n
    return P * r / (1.0 - (1.0 + r) ** (-n))


def residual_value(price: float, years_held: int, d1: float, d2_5: float, d6_10: float, d11p: float) -> float:
    """Simple slider-driven depreciation model. d* are percents (e.g., 20 for 20%)."""
    value = float(max(0.0, price))
    y = int(max(0, years_held))
    for k in range(1, y + 1):
        if k == 1:
            value *= (1.0 - d1 / 100.0)
        elif 2 <= k <= 5:
            value *= (1.0 - d2_5 / 100.0)
        elif 6 <= k <= 10:
            value *= (1.0 - d6_10 / 100.0)
        else:
            value *= (1.0 - d11p / 100.0)
        if value <= 0:
            return 0.0
    return max(0.0, value)


def variable_annuity_fv_from_window(window: np.ndarray, contrib_series: np.ndarray, timing: str = "end") -> float:
    """
    Future value of a *variable* annual contribution series across a per-year factor window.
    `window`: array of length N with per-year growth factors.
    `contrib_series`: array-like of length N with annual contributions (dollars/year).
    `timing`: 'end' -> each year's contribution grows for remaining years only; 'begin' -> grows including current year.
    """
    f = np.asarray(window, dtype=float)
    c = np.asarray(contrib_series, dtype=float)
    if f.size == 0 or c.size == 0:
        return 0.0
    N = min(f.size, c.size)
    f = f[:N]
    c = c[:N]
    # suffix products of factors
    rev_cum = np.cumprod(f[::-1])
    suffix_including = rev_cum[::-1]         # product f[i:]
    suffix_excluding = np.concatenate([suffix_including[1:], [1.0]])  # product f[i+1:]
    if timing == "begin":
        weights = suffix_including
    else:
        weights = suffix_excluding
    return float(np.sum(c * weights))

# ---------------------------
# UI
# ---------------------------

st.title("Spend This — Opportunity Cost Calculator")

# Inputs
with st.sidebar:
    st.header("Inputs")
    current_age = st.number_input("Current Age", min_value=0, max_value=120, value=30)
    retirement_age = st.number_input("Retirement Age", min_value=0, max_value=120, value=65)
    retirement_years = st.slider("Number of Years in Retirement", 20, 35, 30, 1)
    thinking_spend = st.number_input("Thinking of Spending ($)", min_value=0, value=15000, step=500)
    whatif_spend = st.number_input("What if I Spend This Instead ($)", min_value=0, value=5000, step=500)
    # Show lump-sum difference
    lump_diff = max(0, thinking_spend - whatif_spend)
    st.caption(f"**Lump-sum difference:** ${lump_diff:,.0f}")
    st.markdown("---")
    st.subheader("Annual Habits (Recurring Savings)")
    num_habits = st.selectbox("Number of habits", [1, 2, 3, 4, 5], index=0)
    habits = []
    for i in range(1, num_habits + 1):
        st.markdown(f"**Habit {i}**")
        d = st.number_input(f"Daily Spend ($) — Habit {i}", min_value=0.0, value=8.0, step=0.5, key=f"daily_spend_{i}")
        f = st.number_input(f"Frugal Alternative ($) — Habit {i}", min_value=0.0, value=1.0, step=0.5, key=f"frugal_daily_{i}")
        dpw = st.slider(f"Days per Week — Habit {i}", 1, 7, 5, key=f"days_per_week_{i}")
        wpy = st.slider(f"Weeks per Year — Habit {i}", 1, 52, 52, key=f"weeks_per_year_{i}")
        daily_diff_i = max(0.0, d - f)
        annual_contrib_disp_i = max(0.0, daily_diff_i * float(dpw) * float(wpy))
        st.caption(f"Daily difference: ${daily_diff_i:,.2f}/day")
        st.caption(f"Annual investable amount: ${annual_contrib_disp_i:,.0f}/yr")
        habits.append({
            "daily": d, "frugal": f, "dpw": dpw, "wpy": wpy,
            "daily_diff": daily_diff_i, "annual": annual_contrib_disp_i
        })

    st.markdown("---")
    st.subheader("Auto Purchase Strategy")
    frugal_price = st.number_input("Frugal car price ($)", min_value=0, value=30000, step=1000)
    frugal_replace = st.slider("Frugal replacement frequency (years)", 3, 15, 10)

    non_price = st.number_input("Non‑frugal car price ($)", min_value=0, value=100000, step=1000)
    non_down = st.number_input("Non‑frugal down payment ($)", min_value=0, value=30000, step=1000)
    non_rate = st.number_input("Finance rate (APR %)", min_value=0.0, max_value=25.0, value=5.0, step=0.25)
    non_term = st.slider("Finance term (years)", 1, 10, 5)
    non_replace = st.slider("Non‑frugal replacement frequency (years)", 2, 15, 5)

    st.caption("Depreciation model (percent per year)")
    dep_y1 = st.slider("Year 1", 0, 50, 20)
    dep_y2_5 = st.slider("Years 2–5", 0, 40, 15)
    dep_y6_10 = st.slider("Years 6–10", 0, 30, 10)
    dep_y11p = st.slider("Years 11+", 0, 25, 7)

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
    # Coerce numeric in case values are strings or percentages
    for _col in ("Min", "Median"):
        if _col in df_withdrawals.columns:
            df_withdrawals[_col] = pd.to_numeric(df_withdrawals[_col], errors="coerce")
except Exception:
    df_withdrawals = None
try:
    df_withdrawals_spx = pd.read_csv("withdrawals_spx.csv")
    for _col in ("Min", "Median"):
        if _col in df_withdrawals_spx.columns:
            df_withdrawals_spx[_col] = pd.to_numeric(df_withdrawals_spx[_col], errors="coerce")
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

# Annual contributions from each habit
annual_contribs = [h["annual"] for h in habits] if 'habits' in locals() else []

# --- Auto Purchase Strategy: build non‑frugal payment schedule and frugal contributions ---
financed = max(0.0, float(non_price) - float(non_down))
monthly_pmt = pmt(principal=financed, apr_pct=float(non_rate), years_term=int(non_term))
annual_pmt = monthly_pmt * 12.0

# Contribution vector for frugal investor (length = years) — they invest what the non‑frugal pays
auto_contribs = np.zeros(int(years), dtype=float)
starts_non = []
if years > 0 and annual_pmt > 0:
    s = 0
    while s < years:
        starts_non.append(int(s))
        end_y = min(int(years), int(s + non_term))
        auto_contribs[s:end_y] += annual_pmt  # overlaps sum
        s += int(non_replace)

# Count vehicles purchased until retirement
num_cars_frugal = 0 if years <= 0 else sum(1 for t in range(0, int(years), int(frugal_replace)))
num_cars_non = 0 if years <= 0 else sum(1 for t in range(0, int(years), int(non_replace)))

# Residual values at retirement (age at retirement of last vehicle)
last_frugal_start = max([t for t in range(0, int(years), int(frugal_replace))], default=None)
last_non_start = max([t for t in range(0, int(years), int(non_replace))], default=None)
frugal_age_at_ret = 0 if last_frugal_start is None else int(years) - int(last_frugal_start)
non_age_at_ret = 0 if last_non_start is None else int(years) - int(last_non_start)
frugal_residual = residual_value(frugal_price, frugal_age_at_ret, dep_y1, dep_y2_5, dep_y6_10, dep_y11p)
non_residual = residual_value(non_price, non_age_at_ret, dep_y1, dep_y2_5, dep_y6_10, dep_y11p)

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
        "Global Minimum Ending Value": (None if g_min is None else f"${g_min:,.0f}"),
        "SPX Mininimum Ending Value": (None if s_min is None else f"${s_min:,.0f}"),
        "Global Median Ending Value": (None if g_med is None else f"${g_med:,.0f}"),
        "SPX Median Ending Value": (None if s_med is None else f"${s_med:,.0f}"),
    })

result_df = pd.DataFrame(rows)
result_df = result_df[[
    "Allocation",
    "Global Minimum Ending Value",
    "SPX Mininimum Ending Value",
    "Global Median Ending Value",
    "SPX Median Ending Value",
]]

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
        "Global Minimum Ending Value": g_min,
        "SPX Mininimum Ending Value": s_min,
        "Global Median Ending Value": g_med,
        "SPX Median Ending Value": s_med,
    })
raw_df = pd.DataFrame(raw_rows)
raw_df = raw_df[[
    "Allocation",
    "Global Minimum Ending Value",
    "SPX Mininimum Ending Value",
    "Global Median Ending Value",
    "SPX Median Ending Value",
]]
st.download_button(
    "Download table (CSV)",
    data=raw_df.to_csv(index=False).encode("utf-8"),
    file_name=f"spend_this_min_median_{years}y.csv",
    mime="text/csv"
)

#
# ==============================================
# Opportunity Cost — Annual Habit Only (per habit)
# ==============================================
raw_rows_annual_list = []
for idx, annual_contrib in enumerate(annual_contribs, start=1):
    rows_annual = []
    raw_rows_annual = []
    for alloc in common_allocs:
        # Global
        sims_g = build_windows(df_glob, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_glob)
        if sims_g.empty:
            g_min = g_med = np.nan
        else:
            ann_g = np.array([annuity_fv_from_window(w, annual_contrib, timing="end") for w in sims_g["factors"].values])
            g_min = float(np.nanmin(ann_g))
            g_med = float(np.nanmedian(ann_g))
        # SPX
        sims_s = build_windows(df_spx, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_spx)
        if sims_s.empty:
            s_min = s_med = np.nan
        else:
            ann_s = np.array([annuity_fv_from_window(w, annual_contrib, timing="end") for w in sims_s["factors"].values])
            s_min = float(np.nanmin(ann_s))
            s_med = float(np.nanmedian(ann_s))
        rows_annual.append({
            "Allocation": alloc,
            "Global Minimum Ending Value": (None if np.isnan(g_min) else f"${g_min:,.0f}"),
            "SPX Mininimum Ending Value": (None if np.isnan(s_min) else f"${s_min:,.0f}"),
            "Global Median Ending Value": (None if np.isnan(g_med) else f"${g_med:,.0f}"),
            "SPX Median Ending Value": (None if np.isnan(s_med) else f"${s_med:,.0f}"),
        })
        raw_rows_annual.append({
            "Allocation": alloc,
            "Global Minimum Ending Value": g_min,
            "SPX Mininimum Ending Value": s_min,
            "Global Median Ending Value": g_med,
            "SPX Median Ending Value": s_med,
        })
    raw_rows_annual_list.append(raw_rows_annual)

    result_annual_df = pd.DataFrame(rows_annual)
    if not result_annual_df.empty:
        result_annual_df = result_annual_df[[
            "Allocation",
            "Global Minimum Ending Value",
            "SPX Mininimum Ending Value",
            "Global Median Ending Value",
            "SPX Median Ending Value",
        ]]
        st.subheader(f"Opportunity Cost — Annual Habit {idx} Only (Min & Median by Allocation)")
        # Use the captured inputs for this habit in the caption
        h = habits[idx - 1] if idx - 1 < len(habits) else {"daily":0,"frugal":0,"dpw":0,"wpy":0,"annual":0}
        st.caption(
            f"Annual investment = (${h['daily']:,.2f} − ${h['frugal']:,.2f}) × {int(h['dpw'])} × {int(h['wpy'])} "
            f"= **${h['annual']:,.0f} per year**, net of the same expenses (Global 20 bps; SPX 5 bps)."
        )
        st.dataframe(result_annual_df, use_container_width=True)

        raw_annual_df = pd.DataFrame(raw_rows_annual)[[
            "Allocation",
            "Global Minimum Ending Value",
            "SPX Mininimum Ending Value",
            "Global Median Ending Value",
            "SPX Median Ending Value",
        ]]
        st.download_button(
            f"Download annual habit {idx} table (CSV)",
            data=raw_annual_df.to_csv(index=False).encode("utf-8"),
            file_name=f"spend_this_annual_only_habit{idx}_{years}y.csv",
            mime="text/csv"
        )
    else:
        st.info(f"No results to display for the annual habit {idx} section.")

# ==============================================
# Opportunity Cost — Auto Payments Invested (Min & Median by Allocation)
# ==============================================
if years > 0 and np.any(auto_contribs > 0):
    rows_auto = []
    raw_rows_auto = []
    for alloc in common_allocs:
        # Global
        sims_g = build_windows(df_glob, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_glob)
        if sims_g.empty:
            g_min = g_med = np.nan
        else:
            fv_g = np.array([variable_annuity_fv_from_window(w, auto_contribs, timing="end") for w in sims_g["factors"].values])
            g_min = float(np.nanmin(fv_g)); g_med = float(np.nanmedian(fv_g))
        # SPX
        sims_s = build_windows(df_spx, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_spx)
        if sims_s.empty:
            s_min = s_med = np.nan
        else:
            fv_s = np.array([variable_annuity_fv_from_window(w, auto_contribs, timing="end") for w in sims_s["factors"].values])
            s_min = float(np.nanmin(fv_s)); s_med = float(np.nanmedian(fv_s))
        rows_auto.append({
            "Allocation": alloc,
            "Global Minimum Ending Value": (None if np.isnan(g_min) else f"${g_min:,.0f}"),
            "SPX Mininimum Ending Value": (None if np.isnan(s_min) else f"${s_min:,.0f}"),
            "Global Median Ending Value": (None if np.isnan(g_med) else f"${g_med:,.0f}"),
            "SPX Median Ending Value": (None if np.isnan(s_med) else f"${s_med:,.0f}"),
        })
        raw_rows_auto.append({
            "Allocation": alloc,
            "Global Minimum Ending Value": g_min,
            "SPX Mininimum Ending Value": s_min,
            "Global Median Ending Value": g_med,
            "SPX Median Ending Value": s_med,
        })
    result_auto_df = pd.DataFrame(rows_auto)[[
        "Allocation",
        "Global Minimum Ending Value",
        "SPX Mininimum Ending Value",
        "Global Median Ending Value",
        "SPX Median Ending Value",
    ]]
    st.subheader("Opportunity Cost — Auto Payments Invested (Min & Median by Allocation)")
    st.caption(
        f"Frugal invests the non‑frugal payment (monthly pmt ${monthly_pmt:,.0f}, annual ${annual_pmt:,.0f}); "
        f"Non‑frugal finances ${financed:,.0f} at {non_rate:.2f}% for {int(non_term)} years; "
        f"replacement: frugal every {int(frugal_replace)}y, non‑frugal every {int(non_replace)}y."
    )
    st.dataframe(result_auto_df, use_container_width=True)

    raw_auto_df = pd.DataFrame(raw_rows_auto)[[
        "Allocation",
        "Global Minimum Ending Value",
        "SPX Mininimum Ending Value",
        "Global Median Ending Value",
        "SPX Median Ending Value",
    ]]
    st.download_button(
        "Download auto payments invested table (CSV)",
        data=raw_auto_df.to_csv(index=False).encode("utf-8"),
        file_name=f"spend_this_auto_payments_invested_{years}y.csv",
        mime="text/csv"
    )

    # Residual value summary
    resid_df = pd.DataFrame([
        {"Buyer": "Frugal", "Vehicles until retirement": int(num_cars_frugal), "Residual Value at retirement": frugal_residual},
        {"Buyer": "Non‑frugal", "Vehicles until retirement": int(num_cars_non), "Residual Value at retirement": non_residual},
    ])
    resid_disp = resid_df.copy()
    resid_disp["Residual Value at retirement"] = resid_disp["Residual Value at retirement"].map(lambda v: f"${v:,.0f}")
    st.subheader("Auto Residual Value Summary (Informational)")
    st.dataframe(resid_disp, use_container_width=True)
else:
    st.info("Auto Purchase Strategy: No positive financed amount or contributions to invest under current settings.")

# ===========================
# Median Withdrawal Tables
# ===========================
def _lookup_median_withdrawal(df_w: pd.DataFrame | None, yrs: int):
    if df_w is None:
        return None
    try:
        return float(df_w.loc[df_w["Years"] == yrs, "Median"].iloc[0])
    except Exception:
        return None

# --- Median rates (assumed 60/40 withdrawal strategy)
median_withdrawal_g = _lookup_median_withdrawal(df_withdrawals, int(retirement_years))
median_withdrawal_s = _lookup_median_withdrawal(df_withdrawals_spx, int(retirement_years))

# ---------- Lump Sum median withdrawal table ----------
lump_rows = []
if median_withdrawal_g is not None:
    base_g = float(np.nanmax(raw_df["Global Median Ending Value"].values)) if not raw_df.empty else np.nan
    if median_withdrawal_g <= 1.0 and np.isfinite(base_g):
        ann_g = median_withdrawal_g * base_g
    else:
        ann_g = median_withdrawal_g
    lump_rows.append({
        "Portfolio": "Global",
        "Years": int(retirement_years),
        "Annual Retirement Income (Historically)": ann_g,
        "Total Median Retirement Income": ann_g * int(retirement_years),
    })
if median_withdrawal_s is not None:
    base_s = float(np.nanmax(raw_df["SPX Median Ending Value"].values)) if not raw_df.empty else np.nan
    if median_withdrawal_s <= 1.0 and np.isfinite(base_s):
        ann_s = median_withdrawal_s * base_s
    else:
        ann_s = median_withdrawal_s
    lump_rows.append({
        "Portfolio": "SPX",
        "Years": int(retirement_years),
        "Annual Retirement Income (Historically)": ann_s,
        "Total Median Retirement Income": ann_s * int(retirement_years),
    })
if lump_rows:
    lump_df = pd.DataFrame(lump_rows)
    lump_disp = lump_df.copy()
    lump_disp["Annual Retirement Income (Historically)"] = lump_disp["Annual Retirement Income (Historically)"].map(lambda v: f"${v:,.0f}")
    lump_disp["Total Median Retirement Income"] = lump_disp["Total Median Retirement Income"].map(lambda v: f"${v:,.0f}")
    st.subheader("Median Withdrawal — Lump Sum")
    st.caption("Ending portfolio uses the highest median ending value across allocations; withdrawals assume a 60% Equity / 40% Fixed portfolio. Expenses: Global 20 bps; SPX 5 bps.")
    st.dataframe(lump_disp, use_container_width=True)
    st.download_button(
        "Download median withdrawal — Lump Sum (CSV)",
        data=lump_df.to_csv(index=False).encode("utf-8"),
        file_name=f"median_withdrawal_lumpsum_{years}y.csv",
        mime="text/csv"
    )
    # Note: spending vs frugal impact (for each portfolio row)
    try:
        for _, r in lump_df.iterrows():
            total_val = float(r["Total Median Retirement Income"])
            p = str(r.get("Portfolio", "")).strip().lower()
            portfolio_label = "the Global portfolio" if p == "global" else "the S&P 500 (SPX) portfolio"
            _msg = (
                f"Spending ${thinking_spend:,.0f} instead of ${whatif_spend:,.0f} "
                f"cost you ${total_val:,.0f} in lifetime retirement income by investing in {portfolio_label}."
            )
            st.markdown(
                f"<div style='white-space: normal; word-break: normal; overflow-wrap: break-word;'><strong>{_msg}</strong></div>",
                unsafe_allow_html=True
            )
    except Exception:
        pass
else:
    st.info("Median withdrawals not found for Lump Sum scenario.")


# ---------- Annual Habit Only median withdrawal tables (per habit) ----------
for idx, raw_rows_annual in enumerate(raw_rows_annual_list, start=1):
    annual_rows = []
    if median_withdrawal_g is not None and len(raw_rows_annual) > 0:
        try:
            base_g_a = float(np.nanmax([row.get("Global Median Ending Value", np.nan) for row in raw_rows_annual]))
        except Exception:
            base_g_a = np.nan
        if median_withdrawal_g <= 1.0 and np.isfinite(base_g_a):
            ann_g_a = median_withdrawal_g * base_g_a
        else:
            ann_g_a = median_withdrawal_g
        annual_rows.append({
            "Portfolio": "Global",
            "Years": int(retirement_years),
            "Annual Retirement Income (Historically)": ann_g_a,
            "Total Median Retirement Income": ann_g_a * int(retirement_years),
        })
    if median_withdrawal_s is not None and len(raw_rows_annual) > 0:
        try:
            base_s_a = float(np.nanmax([row.get("SPX Median Ending Value", np.nan) for row in raw_rows_annual]))
        except Exception:
            base_s_a = np.nan
        if median_withdrawal_s <= 1.0 and np.isfinite(base_s_a):
            ann_s_a = median_withdrawal_s * base_s_a
        else:
            ann_s_a = median_withdrawal_s
        annual_rows.append({
            "Portfolio": "SPX",
            "Years": int(retirement_years),
            "Annual Retirement Income (Historically)": ann_s_a,
            "Total Median Retirement Income": ann_s_a * int(retirement_years),
        })
    if annual_rows:
        annual_df = pd.DataFrame(annual_rows)
        annual_disp = annual_df.copy()
        annual_disp["Annual Retirement Income (Historically)"] = annual_disp["Annual Retirement Income (Historically)"].map(lambda v: f"${v:,.0f}")
        annual_disp["Total Median Retirement Income"] = annual_disp["Total Median Retirement Income"].map(lambda v: f"${v:,.0f}")
        st.subheader(f"Median Withdrawal — Annual Habit {idx} Only")
        st.caption("Ending portfolio uses the highest median ending value (annual contributions only) across allocations; withdrawals assume a 60% Equity / 40% Fixed portfolio. Expenses: Global 20 bps; SPX 5 bps.")
        st.dataframe(annual_disp, use_container_width=True)
        st.download_button(
            f"Download median withdrawal — Annual Habit {idx} Only (CSV)",
            data=annual_df.to_csv(index=False).encode("utf-8"),
            file_name=f"median_withdrawal_annual_only_habit{idx}_{years}y.csv",
            mime="text/csv"
        )
        # Note: spending vs frugal impact for this habit (for each portfolio row)
        try:
            for _, r in annual_df.iterrows():
                total_val = float(r["Total Median Retirement Income"])
                p = str(r.get("Portfolio", "")).strip().lower()
                portfolio_label = "the Global portfolio" if p == "global" else "the S&P 500 (SPX) portfolio"
                _msg_h = (
                    f"Spending ${h['daily']:,.2f} instead of ${h['frugal']:,.2f} "
                    f"cost you ${total_val:,.0f} in lifetime retirement income by investing in {portfolio_label}."
                )
                st.markdown(
                    f"<div style='white-space: normal; word-break: normal; overflow-wrap: break-word;'><strong>{_msg_h}</strong></div>",
                    unsafe_allow_html=True
                )
        except Exception:
            pass

    else:
        st.info(f"Median withdrawals not found for Annual Habit {idx} scenario.")


# ---------- Grand Total median withdrawal (Lump + All Annual Habits) ----------
try:
    years_int = int(retirement_years)

    # Recompute Lump Sum annual income for Global & SPX to avoid scope issues
    annual_lump_g = 0.0
    if median_withdrawal_g is not None and not raw_df.empty:
        base_g = float(np.nanmax(raw_df["Global Median Ending Value"].values))
        annual_lump_g = (median_withdrawal_g * base_g) if (median_withdrawal_g <= 1.0 and np.isfinite(base_g)) else float(median_withdrawal_g)

    annual_lump_s = 0.0
    if median_withdrawal_s is not None and not raw_df.empty:
        base_s = float(np.nanmax(raw_df["SPX Median Ending Value"].values))
        annual_lump_s = (median_withdrawal_s * base_s) if (median_withdrawal_s <= 1.0 and np.isfinite(base_s)) else float(median_withdrawal_s)

    # Sum Annual Habit incomes across all habits for Global & SPX
    habits_total_g = 0.0
    habits_total_s = 0.0
    for raw_rows_annual in raw_rows_annual_list:
        # Global
        if median_withdrawal_g is not None and len(raw_rows_annual) > 0:
            try:
                base_g_a = float(np.nanmax([row.get("Global Median Ending Value", np.nan) for row in raw_rows_annual]))
            except Exception:
                base_g_a = np.nan
            if median_withdrawal_g <= 1.0 and np.isfinite(base_g_a):
                habits_total_g += median_withdrawal_g * base_g_a
            else:
                habits_total_g += float(median_withdrawal_g)
        # SPX
        if median_withdrawal_s is not None and len(raw_rows_annual) > 0:
            try:
                base_s_a = float(np.nanmax([row.get("SPX Median Ending Value", np.nan) for row in raw_rows_annual]))
            except Exception:
                base_s_a = np.nan
            if median_withdrawal_s <= 1.0 and np.isfinite(base_s_a):
                habits_total_s += median_withdrawal_s * base_s_a
            else:
                habits_total_s += float(median_withdrawal_s)

    # Build Grand Total table
    grand_rows = [
        {
            "Portfolio": "Global",
            "Years": years_int,
            "Annual Retirement Income — Lump Sum": annual_lump_g,
            "Annual Retirement Income — Habits Total": habits_total_g,
            "Annual Retirement Income — Grand Total": annual_lump_g + habits_total_g,
            "Total — Lump Sum": annual_lump_g * years_int,
            "Total — Habits Total": habits_total_g * years_int,
            "Total — Grand Total": (annual_lump_g + habits_total_g) * years_int,
        },
        {
            "Portfolio": "SPX",
            "Years": years_int,
            "Annual Retirement Income — Lump Sum": annual_lump_s,
            "Annual Retirement Income — Habits Total": habits_total_s,
            "Annual Retirement Income — Grand Total": annual_lump_s + habits_total_s,
            "Total — Lump Sum": annual_lump_s * years_int,
            "Total — Habits Total": habits_total_s * years_int,
            "Total — Grand Total": (annual_lump_s + habits_total_s) * years_int,
        },
    ]

    grand_df = pd.DataFrame(grand_rows)
    # Display with currency formatting
    grand_disp = grand_df.copy()
    for col in [
        "Annual Retirement Income — Lump Sum", "Annual Retirement Income — Habits Total", "Annual Retirement Income — Grand Total",
        "Total — Lump Sum", "Total — Habits Total", "Total — Grand Total",
    ]:
        grand_disp[col] = grand_disp[col].map(lambda v: f"${v:,.0f}")

    st.subheader("Median Withdrawal — Grand Total (Lump + All Annual Habits)")
    st.dataframe(grand_disp, use_container_width=True)
    st.download_button(
        "Download median withdrawal — Grand Total (CSV)",
        data=grand_df.to_csv(index=False).encode("utf-8"),
        file_name=f"median_withdrawal_grand_total_{years}y.csv",
        mime="text/csv"
    )
except Exception:
    pass