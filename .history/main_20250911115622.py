import streamlit as st
import pandas as pd
import numpy as np
import re

# ---------------------------
# Helpers: factor loading & windows
# ---------------------------

def _canonical_alloc_name(col: str) -> str | None:
    s_raw = str(col).strip()
    s = s_raw.lower().replace("_", "").replace(" ", "")
    m_exact = re.fullmatch(r"(\d{1,3})e", s)
    if m_exact:
        pct = max(0, min(100, int(m_exact.group(1))))
        return f"{pct}E"
    if s.endswith("100f"):
        return "0E"
    m = re.search(r"(\d{1,3})e$", s)
    if m:
        pct = max(0, min(100, int(m.group(1))))
        return f"{pct}E"
    m2 = re.search(r"(\d{1,3})$", s)
    if m2 and any(prefix in s for prefix in ("lbm", "spx", "glob", "global")):
        pct = max(0, min(100, int(m2.group(1))))
        return f"{pct}E"
    return None

def load_factors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    canon_cols = {}
    for c in df.columns:
        canon = _canonical_alloc_name(c)
        if canon is None:
            continue
        if canon not in canon_cols:
            canon_cols[canon] = df[c]
    out = pd.DataFrame(canon_cols)
    out = out.dropna(axis=1, how="all")
    if out.empty:
        return df.dropna(axis=1, how="all")
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
        window = window * fee_mult_per_step
        fv = float(np.prod(window))
        rows.append((s, window, fv))
    return pd.DataFrame(rows, columns=["start_index","factors","fv_multiple"])

# ---------------------------
# Helpers: annuities / autos
# ---------------------------

def annuity_fv_from_window(window: np.ndarray, contrib: float, timing: str = "end") -> float:
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
    rev_cum = np.cumprod(f[::-1])
    suffix_including = rev_cum[::-1]
    suffix_excluding = np.concatenate([suffix_including[1:], [1.0]])
    if timing == "begin":
        return float(c * np.sum(suffix_including))
    else:
        return float(c * np.sum(suffix_excluding))

def pmt(principal: float, apr_pct: float, years_term: int) -> float:
    P = float(max(0.0, principal))
    n = int(max(1, years_term)) * 12
    r = float(apr_pct) / 100.0 / 12.0
    if P <= 0:
        return 0.0
    if r == 0.0:
        return P / n
    return P * r / (1.0 - (1.0 + r) ** (-n))

def residual_value(price: float, years_held: int, d1: float, d2_5: float, d6_10: float, d11p: float) -> float:
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
    f = np.asarray(window, dtype=float)
    c = np.asarray(contrib_series, dtype=float)
    if f.size == 0 or c.size == 0:
        return 0.0
    N = min(f.size, c.size)
    f = f[:N]
    c = c[:N]
    rev_cum = np.cumprod(f[::-1])
    suffix_including = rev_cum[::-1]
    suffix_excluding = np.concatenate([suffix_including[1:], [1.0]])
    weights = suffix_including if (timing == "begin") else suffix_excluding
    return float(np.sum(c * weights))

def build_payment_vector(price: float, initial_down: float, apr_pct: float, years_term: int, replace_freq: int,
                         horizon_years: int, d1: float, d2_5: float, d6_10: float, d11p: float,
                         apply_residual: bool) -> tuple[np.ndarray, int, int | None]:
    """Return (annual_payment_vector, num_cars, last_start_year)."""
    Y = int(max(0, horizon_years))
    vec = np.zeros(Y, dtype=float)
    if Y == 0:
        return vec, 0, None
    t = 0
    num_cars = 0
    last_start = None
    down_next = float(max(0.0, initial_down))
    while t < Y:
        num_cars += 1
        last_start = t
        financed_amt = max(0.0, float(price) - down_next)
        if financed_amt > 0 and years_term > 0:
            ann_pmt = pmt(financed_amt, apr_pct, years_term) * 12.0
            end_y = min(Y, t + int(years_term))
            vec[t:end_y] += ann_pmt
        hold = min(int(replace_freq), Y - t)
        res = residual_value(price, hold, d1, d2_5, d6_10, d11p) if apply_residual else 0.0
        t += int(replace_freq)
        down_next = float(max(0.0, res))
    return vec, num_cars, last_start

# ---------------------------
# UI
# ---------------------------

st.title("Spend This — Opportunity Cost Calculator")

with st.sidebar:
    st.header("Inputs")
    current_age = st.number_input("Current Age", min_value=0, max_value=120, value=30)
    retirement_age = st.number_input("Retirement Age", min_value=0, max_value=120, value=65)
    retirement_years = st.slider("Number of Years in Retirement", 20, 35, 30, 1)

    thinking_spend = st.number_input("Thinking of Spending ($)", min_value=0, value=15000, step=500)
    whatif_spend = st.number_input("What if I Spend This Instead ($)", min_value=0, value=5000, step=500)
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
        habits.append({"daily": d, "frugal": f, "dpw": dpw, "wpy": wpy, "daily_diff": daily_diff_i, "annual": annual_contrib_disp_i})

    st.markdown("---")
    st.subheader("Auto Purchase Strategy")
    frugal_price = st.number_input("Frugal car price ($)", min_value=0, value=30000, step=1000)
    frugal_replace = st.slider("Frugal replacement frequency (years)", 3, 15, 10)
    # NEW: frugal financing + residual-to-down-payment
    frugal_down = st.number_input("Frugal down payment ($)", min_value=0, value=30000, step=1000)
    frugal_rate = st.number_input("Frugal finance rate (APR %)", min_value=0.0, max_value=25.0, value=5.0, step=0.25)
    frugal_term = st.slider("Frugal finance term (years)", 1, 10, 5)
    apply_residual_dp = st.checkbox("Use residual value as next down payment", value=True)

    non_price = st.number_input("Non-frugal car price ($)", min_value=0, value=100000, step=1000)
    non_down = st.number_input("Non-frugal down payment ($)", min_value=0, value=30000, step=1000)
    non_rate = st.number_input("Finance rate (APR %)", min_value=0.0, max_value=25.0, value=5.0, step=0.25)
    non_term = st.slider("Finance term (years)", 1, 10, 5)
    _financed_preview = max(0.0, float(non_price) - float(non_down))
    _monthly_preview = pmt(principal=_financed_preview, apr_pct=float(non_rate), years_term=int(non_term))
    _annual_preview = _monthly_preview * 12.0
    st.markdown(f"**Annual Payment:** ${_annual_preview:,.0f}")
    non_replace = st.slider("Non-frugal replacement frequency (years)", 2, 15, 5)

    st.caption("Depreciation model (percent per year)")
    dep_y1 = st.slider("Year 1", 0, 50, 20)
    dep_y2_5 = st.slider("Years 2–5", 0, 40, 15)
    dep_y6_10 = st.slider("Years 6–10", 0, 30, 10)
    dep_y11p = st.slider("Years 11+", 0, 25, 7)

    st.markdown("---")
    st.subheader("Housing Strategy")
    house_spender_price = st.number_input("Spender home price ($)", min_value=0, value=500000, step=10000)
    house_frugal_price = st.number_input("Frugal home price ($)", min_value=0, value=300000, step=10000)

    # Down payment controls
    house_down_pct = st.slider("Down payment % (applies if overrides are 0)", 0, 100, 20)
    house_spender_down_amt = st.number_input("Spender down payment override ($)", min_value=0, value=0, step=5000)
    house_frugal_down_amt = st.number_input("Frugal down payment override ($)", min_value=0, value=0, step=5000)

    # Mortgage controls
    house_apr = st.number_input("Mortgage APR (%)", min_value=0.0, max_value=25.0, value=5.0, step=0.25)
    house_term = st.slider("Mortgage term (years)", 5, 40, 30)

    # Preview annual payments for both buyers using current settings
    _dp_spender_prev = float(house_spender_down_amt) if float(house_spender_down_amt) > 0 else (house_down_pct / 100.0) * float(house_spender_price)
    _dp_frugal_prev  = float(house_frugal_down_amt)  if float(house_frugal_down_amt)  > 0 else (house_down_pct / 100.0) * float(house_frugal_price)
    _p_spender_prev  = max(0.0, float(house_spender_price) - _dp_spender_prev)
    _p_frugal_prev   = max(0.0, float(house_frugal_price) - _dp_frugal_prev)
    _mo_spender_prev = pmt(_p_spender_prev, float(house_apr), int(house_term))
    _mo_frugal_prev  = pmt(_p_frugal_prev, float(house_apr), int(house_term))
    _sp_ann = _mo_spender_prev * 12.0
    _fr_ann = _mo_frugal_prev * 12.0
    _diff_ann = max(0.0, _sp_ann - _fr_ann)
    st.caption(f"**Annual payments:**")
    st.caption(f"Spender ${_sp_ann:,.0f}")
    st.caption(f"Frugal ${_fr_ann:,.0f}")
    st.caption(f"Difference ${_diff_ann:,.0f}")
    house_tax_rate = st.slider("Property tax rate (%)", 0.0, 3.0, 2.5, 0.1)

# ---------------------------
# Global constants
# ---------------------------
_default_step = 12
fee_mult_per_step_glob = (1.0 - 0.0020) ** (_default_step / 12.0)
fee_mult_per_step_spx  = (1.0 - 0.0005) ** (_default_step / 12.0)

years = retirement_age - current_age
if years <= 0:
    st.error("Retirement age must be greater than current age.")
    st.stop()

# Load datasets
try:
    df_glob = load_factors("global_factors.csv")
    df_spx  = load_factors("spx_factors.csv")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Withdrawals
try:
    df_withdrawals = pd.read_csv("withdrawals.csv")
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

# Columns / allocations
allocs_glob = [c for c in df_glob.columns]
allocs_spx  = [c for c in df_spx.columns]
def _order_key(x: str):
    try:
        return int(str(x).replace("E",""))
    except Exception:
        return -1
common_allocs = sorted(set(allocs_glob).intersection(allocs_spx), key=_order_key, reverse=True)
if not common_allocs:
    st.error("No common allocation columns between global_factors.csv and spx_factors.csv.")
    st.stop()

# Lump-sum spending difference
spend_diff = thinking_spend - whatif_spend
if spend_diff <= 0:
    st.info("No opportunity cost: the 'Thinking of Spending' amount must be greater than the 'What if I Spend This Instead' amount.")
    spend_diff = 0.0

# Section flags
has_lump = spend_diff > 0

# Annual contributions from each habit
annual_contribs = [h["annual"] for h in habits] if 'habits' in locals() else []
has_habits = any(float(c) > 0 for c in annual_contribs) if annual_contribs else False

# ---------------------------
# Housing Strategy: build mortgage payment vectors and invest differences
# ---------------------------
dp_spender = float(house_spender_down_amt) if float(house_spender_down_amt) > 0 else (house_down_pct / 100.0) * float(house_spender_price)
dp_frugal  = float(house_frugal_down_amt)  if float(house_frugal_down_amt)  > 0 else (house_down_pct / 100.0) * float(house_frugal_price)
dp_spender = min(dp_spender, float(house_spender_price))
dp_frugal  = min(dp_frugal,  float(house_frugal_price))

loan_spender = max(0.0, float(house_spender_price) - dp_spender)
loan_frugal  = max(0.0, float(house_frugal_price)  - dp_frugal)

mo_spender = pmt(loan_spender, float(house_apr), int(house_term))
mo_frugal  = pmt(loan_frugal,  float(house_apr), int(house_term))
an_spender = mo_spender * 12.0
an_frugal  = mo_frugal  * 12.0

house_spender_vec = np.zeros(int(years), dtype=float)
house_frugal_vec  = np.zeros(int(years), dtype=float)
if int(years) > 0:
    end_sp = min(int(years), int(house_term))
    end_fr = min(int(years), int(house_term))
    if end_sp > 0:
        house_spender_vec[:end_sp] = an_spender
    if end_fr > 0:
        house_frugal_vec[:end_fr] = an_frugal

# Payment difference each year (invested) + property tax difference each year
tax_delta_annual = max(0.0, (float(house_spender_price) - float(house_frugal_price)) * (float(house_tax_rate) / 100.0))
if int(years) > 0:
    housing_contribs = np.maximum(0.0, house_spender_vec - house_frugal_vec) + np.full(int(years), tax_delta_annual, dtype=float)
else:
    housing_contribs = np.array([], dtype=float)
# Down payment difference invested at year 0 (beginning-of-year)
dp_diff = max(0.0, dp_spender - dp_frugal)

has_housing = bool(np.any(housing_contribs > 0) or dp_diff > 0)

# ---------------------------
# Auto Purchase Strategy: payment streams & residuals (DIFFERENCE-based)
# ---------------------------
financed_first = max(0.0, float(non_price) - float(non_down))
monthly_pmt_first = pmt(principal=financed_first, apr_pct=float(non_rate), years_term=int(non_term))
annual_pmt_first = monthly_pmt_first * 12.0
monthly_pmt = monthly_pmt_first
annual_pmt = annual_pmt_first

non_vec, num_cars_non, last_non_start = build_payment_vector(
    price=float(non_price), initial_down=float(non_down), apr_pct=float(non_rate), years_term=int(non_term),
    replace_freq=int(non_replace), horizon_years=int(years),
    d1=dep_y1, d2_5=dep_y2_5, d6_10=dep_y6_10, d11p=dep_y11p,
    apply_residual=bool(apply_residual_dp)
)

frugal_vec, num_cars_frugal, last_frugal_start = build_payment_vector(
    price=float(frugal_price), initial_down=float(frugal_down), apr_pct=float(frugal_rate), years_term=int(frugal_term),
    replace_freq=int(frugal_replace), horizon_years=int(years),
    d1=dep_y1, d2_5=dep_y2_5, d6_10=dep_y6_10, d11p=dep_y11p,
    apply_residual=bool(apply_residual_dp)
)

# Frugal invests the difference each year
auto_contribs = np.maximum(0.0, non_vec - frugal_vec)
has_auto = bool(np.any(auto_contribs > 0))

# Residuals at retirement
frugal_age_at_ret = 0 if last_frugal_start is None else int(years) - int(last_frugal_start)
non_age_at_ret = 0 if last_non_start is None else int(years) - int(last_non_start)
frugal_residual = residual_value(frugal_price, frugal_age_at_ret, dep_y1, dep_y2_5, dep_y6_10, dep_y11p)
non_residual = residual_value(non_price, non_age_at_ret, dep_y1, dep_y2_5, dep_y6_10, dep_y11p)

# ---------------------------
# Opportunity Cost — Lump Sum (Min & Median by Allocation)
# ---------------------------
if has_lump:
    rows = []
    for alloc in common_allocs:
        sims_g = build_windows(df_glob, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_glob)
        if sims_g.empty:
            g_min = g_med = None
        else:
            end_g = sims_g["fv_multiple"] * spend_diff
            g_min = float(end_g.min()); g_med = float(end_g.median())
        sims_s = build_windows(df_spx, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_spx)
        if sims_s.empty:
            s_min = s_med = None
        else:
            end_s = sims_s["fv_multiple"] * spend_diff
            s_min = float(end_s.min()); s_med = float(end_s.median())
        rows.append({
            "Allocation": alloc,
            "Global Minimum Ending Value": (None if g_min is None else f"${g_min:,.0f}"),
            "SPX Mininimum Ending Value": (None if s_min is None else f"${s_min:,.0f}"),
            "Global Median Ending Value": (None if g_med is None else f"${g_med:,.0f}"),
            "SPX Median Ending Value": (None if s_med is None else f"${s_med:,.0f}"),
        })
    result_df = pd.DataFrame(rows)[["Allocation","Global Minimum Ending Value","SPX Mininimum Ending Value","Global Median Ending Value","SPX Median Ending Value"]]
    st.subheader("Opportunity Cost of the Difference for Lump Sum Spending")
    st.markdown("**Thinking vs What-if difference invested across all historical windows**")
    st.caption("This model assumes a fixed annual expense ratio of 0.20% for Global and 0.05% for SP500 portfolios.")
    st.dataframe(result_df, use_container_width=True)

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
    raw_df = pd.DataFrame(raw_rows)[["Allocation","Global Minimum Ending Value","SPX Mininimum Ending Value","Global Median Ending Value","SPX Median Ending Value"]]
    st.download_button("Download table (CSV)", data=raw_df.to_csv(index=False).encode("utf-8"), file_name=f"spend_this_min_median_{years}y.csv", mime="text/csv")
else:
    # Ensure raw_df exists for downstream guards
    raw_df = pd.DataFrame()

# ---------------------------
# Opportunity Cost — Annual Habit Only (per habit)
# ---------------------------
raw_rows_annual_list = []
for idx, annual_contrib in enumerate(annual_contribs, start=1):
    if float(annual_contrib) <= 0:
        continue
    rows_annual = []
    raw_rows_annual = []
    for alloc in common_allocs:
        sims_g = build_windows(df_glob, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_glob)
        if sims_g.empty:
            g_min = g_med = np.nan
        else:
            ann_g = np.array([annuity_fv_from_window(w, annual_contrib, timing="end") for w in sims_g["factors"].values])
            g_min = float(np.nanmin(ann_g)); g_med = float(np.nanmedian(ann_g))
        sims_s = build_windows(df_spx, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_spx)
        if sims_s.empty:
            s_min = s_med = np.nan
        else:
            ann_s = np.array([annuity_fv_from_window(w, annual_contrib, timing="end") for w in sims_s["factors"].values])
            s_min = float(np.nanmin(ann_s)); s_med = float(np.nanmedian(ann_s))
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
    result_annual_df = pd.DataFrame(rows_annual)[["Allocation","Global Minimum Ending Value","SPX Mininimum Ending Value","Global Median Ending Value","SPX Median Ending Value"]]
    if not result_annual_df.empty:
        st.subheader(f"Opportunity Cost — Annual Habit {idx} Only (Min & Median by Allocation)")
        h = habits[idx - 1] if idx - 1 < len(habits) else {"daily":0,"frugal":0,"dpw":0,"wpy":0,"annual":0}
        st.caption(
            f"Annual investment = (${h['daily']:,.2f} − ${h['frugal']:,.2f}) × {int(h['dpw'])} × {int(h['wpy'])} = **${h['annual']:,.0f} per year**, net of the same expenses (Global 20 bps; SPX 5 bps)."
        )
        st.dataframe(result_annual_df, use_container_width=True)
        raw_annual_df = pd.DataFrame(raw_rows_annual)[["Allocation","Global Minimum Ending Value","SPX Mininimum Ending Value","Global Median Ending Value","SPX Median Ending Value"]]
        st.download_button(f"Download annual habit {idx} table (CSV)", data=raw_annual_df.to_csv(index=False).encode("utf-8"), file_name=f"spend_this_annual_only_habit{idx}_{years}y.csv", mime="text/csv")
    else:
        st.info(f"No results to display for the annual habit {idx} section.")

# ---------------------------
# Frugal Investment Contribution Schedules (Year-by-Year)
# ---------------------------
if int(years) > 0:
    _n_years = int(years)
    year_idx = np.arange(1, _n_years + 1)

    # Auto payments schedule (both buyers) and invested difference
    if has_auto:
        auto_sched_df = pd.DataFrame({
            "Year": year_idx,
            "Non-frugal Payment ($/yr)": non_vec[:_n_years] if non_vec.size >= _n_years else np.zeros(_n_years, dtype=float),
            "Frugal Payment ($/yr)": frugal_vec[:_n_years] if frugal_vec.size >= _n_years else np.zeros(_n_years, dtype=float),
            "Invested Difference ($/yr)": auto_contribs[:_n_years] if auto_contribs.size >= _n_years else np.zeros(_n_years, dtype=float),
        })
        auto_sched_disp = auto_sched_df.copy()
        for col in ["Non-frugal Payment ($/yr)", "Frugal Payment ($/yr)", "Invested Difference ($/yr)"]:
            auto_sched_disp[col] = auto_sched_disp[col].map(lambda v: f"${v:,.0f}")
        st.subheader("Frugal Contributions — Auto Payments (Year by Year)")
        st.dataframe(auto_sched_disp, use_container_width=True)
        st.download_button("Download auto payments schedule (CSV)", data=auto_sched_df.to_csv(index=False).encode("utf-8"), file_name=f"frugal_auto_payments_schedule_{_n_years}y.csv", mime="text/csv")

    # Housing year-by-year schedule
    if has_housing:
        house_sched_df = pd.DataFrame({
            "Year": year_idx,
            "Spender Mortgage Payment ($/yr)": house_spender_vec[:_n_years],
            "Frugal Mortgage Payment ($/yr)": house_frugal_vec[:_n_years],
            "Property Tax Difference ($/yr)": np.full(_n_years, tax_delta_annual),
            "Invested Difference ($/yr)": housing_contribs[:_n_years],
            "Down Payment Difference ($)": np.concatenate(([dp_diff], np.zeros(max(0, _n_years-1))))
        })
        house_sched_disp = house_sched_df.copy()
        for col in ["Spender Mortgage Payment ($/yr)", "Frugal Mortgage Payment ($/yr)", "Property Tax Difference ($/yr)", "Invested Difference ($/yr)", "Down Payment Difference ($)"]:
            house_sched_disp[col] = house_sched_disp[col].map(lambda v: f"${v:,.0f}")
        st.subheader("Frugal Contributions — Housing (Year by Year)")
        st.dataframe(house_sched_disp, use_container_width=True)
        st.download_button("Download housing schedule (CSV)", data=house_sched_df.to_csv(index=False).encode("utf-8"), file_name=f"frugal_housing_schedule_{_n_years}y.csv", mime="text/csv")

    # Annual habits schedules (one column per habit) + total
    if has_habits:
        habit_cols = {}
        habit_col_names = []
        for i, h in enumerate(habits, start=1):
            col_name = f"Habit {i} ($/yr)"
            habit_col_names.append(col_name)
            habit_cols[col_name] = np.full(_n_years, float(h.get("annual", 0.0)))
        habits_sched_df = pd.DataFrame({"Year": year_idx, **habit_cols})
        if habit_col_names:
            habits_sched_df["Total Habits ($/yr)"] = habits_sched_df[habit_col_names].sum(axis=1)
        habits_sched_disp = habits_sched_df.copy()
        for cn in habit_col_names + (["Total Habits ($/yr)"] if habit_col_names else []):
            habits_sched_disp[cn] = habits_sched_disp[cn].map(lambda v: f"${v:,.0f}")
        st.subheader("Frugal Contributions — Annual Habits (Year by Year)")
        st.dataframe(habits_sched_disp, use_container_width=True)
        st.download_button("Download habits schedule (CSV)", data=habits_sched_df.to_csv(index=False).encode("utf-8"), file_name=f"frugal_habits_schedule_{_n_years}y.csv", mime="text/csv")

        # Combined only if at least one of auto/habits is present
        total_habits_vec = habits_sched_df["Total Habits ($/yr)"].values if habit_col_names else np.zeros(_n_years, dtype=float)
        if has_auto or has_habits:
            combined_df = pd.DataFrame({
                "Year": year_idx,
                "Auto Payment Invested ($/yr)": (auto_sched_df["Invested Difference ($/yr)"].values if has_auto else np.zeros(_n_years, dtype=float)),
                "Total Habits ($/yr)": total_habits_vec,
            })
            combined_df["Total Frugal Contributions ($/yr)"] = combined_df["Auto Payment Invested ($/yr)"] + combined_df["Total Habits ($/yr)"]
            combined_disp = combined_df.copy()
            for cn in ["Auto Payment Invested ($/yr)", "Total Habits ($/yr)", "Total Frugal Contributions ($/yr)"]:
                combined_disp[cn] = combined_disp[cn].map(lambda v: f"${v:,.0f}")
            st.subheader("Frugal Contributions — Combined (Year by Year)")
            st.dataframe(combined_disp, use_container_width=True)
            st.download_button("Download combined frugal contributions (CSV)", data=combined_df.to_csv(index=False).encode("utf-8"), file_name=f"frugal_contributions_combined_{_n_years}y.csv", mime="text/csv")

 # ==============================================
# Opportunity Cost — Auto Payments Invested (Min & Median by Allocation)
# ==============================================
if years > 0 and np.any(auto_contribs > 0):
    rows_auto = []
    raw_rows_auto = []
    for alloc in common_allocs:
        sims_g = build_windows(df_glob, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_glob)
        if sims_g.empty:
            g_min = g_med = np.nan
        else:
            fv_g = np.array([variable_annuity_fv_from_window(w, auto_contribs, timing="end") for w in sims_g["factors"].values])
            g_min = float(np.nanmin(fv_g)); g_med = float(np.nanmedian(fv_g))
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
        f"Frugal invests the **difference** between non-frugal and frugal payments; first-cycle non-frugal annual payment ${annual_pmt:,.0f}. "
        f"Non-frugal finances ${financed_first:,.0f} at {non_rate:.2f}% for {int(non_term)} years; replacement: frugal every {int(frugal_replace)}y, non-frugal every {int(non_replace)}y."
    )
    st.dataframe(result_auto_df, use_container_width=True)

    raw_auto_df = pd.DataFrame(raw_rows_auto)[[
        "Allocation",
        "Global Minimum Ending Value",
        "SPX Mininimum Ending Value",
        "Global Median Ending Value",
        "SPX Median Ending Value",
    ]]
    st.download_button("Download auto payments invested table (CSV)", data=raw_auto_df.to_csv(index=False).encode("utf-8"), file_name=f"spend_this_auto_payments_invested_{years}y.csv", mime="text/csv")

    # Residual value summary
    resid_df = pd.DataFrame([
        {"Buyer": "Frugal", "Vehicles until retirement": int(num_cars_frugal), "Residual Value at retirement": frugal_residual},
        {"Buyer": "Non-frugal", "Vehicles until retirement": int(num_cars_non), "Residual Value at retirement": non_residual},
    ])
    resid_disp = resid_df.copy()
    resid_disp["Residual Value at retirement"] = resid_disp["Residual Value at retirement"].map(lambda v: f"${v:,.0f}")
    st.subheader("Auto Residual Value Summary (Informational)")
    st.dataframe(resid_disp, use_container_width=True)
else:
    st.info("Auto Purchase Strategy: No positive financed amount or contributions to invest under current settings.")

# ==============================================
# Opportunity Cost — Housing Payments Invested (Min & Median by Allocation)
# ==============================================
if years > 0 and has_housing:
    rows_house = []
    raw_rows_house = []

    # A series with dp_diff at year 0
    dp_series = np.zeros(int(years), dtype=float)
    if int(years) > 0 and dp_diff > 0:
        dp_series[0] = dp_diff

    for alloc in common_allocs:
        sims_g = build_windows(df_glob, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_glob)
        if sims_g.empty:
            g_min = g_med = np.nan
        else:
            fv_pay_g = np.array([variable_annuity_fv_from_window(w, housing_contribs, timing="end") for w in sims_g["factors"].values])
            fv_dp_g  = np.array([variable_annuity_fv_from_window(w, dp_series, timing="begin") for w in sims_g["factors"].values])
            fv_g = fv_pay_g + fv_dp_g
            g_min = float(np.nanmin(fv_g)); g_med = float(np.nanmedian(fv_g))

        sims_s = build_windows(df_spx, alloc, years, step=12, fee_mult_per_step=fee_mult_per_step_spx)
        if sims_s.empty:
            s_min = s_med = np.nan
        else:
            fv_pay_s = np.array([variable_annuity_fv_from_window(w, housing_contribs, timing="end") for w in sims_s["factors"].values])
            fv_dp_s  = np.array([variable_annuity_fv_from_window(w, dp_series, timing="begin") for w in sims_s["factors"].values])
            fv_s = fv_pay_s + fv_dp_s
            s_min = float(np.nanmin(fv_s)); s_med = float(np.nanmedian(fv_s))

        rows_house.append({
            "Allocation": alloc,
            "Global Minimum Ending Value": (None if np.isnan(g_min) else f"${g_min:,.0f}"),
            "SPX Mininimum Ending Value": (None if np.isnan(s_min) else f"${s_min:,.0f}"),
            "Global Median Ending Value": (None if np.isnan(g_med) else f"${g_med:,.0f}"),
            "SPX Median Ending Value": (None if np.isnan(s_med) else f"${s_med:,.0f}"),
        })
        raw_rows_house.append({
            "Allocation": alloc,
            "Global Minimum Ending Value": g_min,
            "SPX Mininimum Ending Value": s_min,
            "Global Median Ending Value": g_med,
            "SPX Median Ending Value": s_med,
        })

    result_house_df = pd.DataFrame(rows_house)[[
        "Allocation",
        "Global Minimum Ending Value",
        "SPX Mininimum Ending Value",
        "Global Median Ending Value",
        "SPX Median Ending Value",
    ]]
    st.subheader("Opportunity Cost — Housing Payments Invested (Min & Median by Allocation)")
    st.caption(
        f"Housing: invest the **difference** between Spender and Frugal mortgage payments, plus the **down payment difference** at year 0. "
        f"APR {house_apr:.2f}% • term {int(house_term)} years • DP% {house_down_pct}% (overrides applied if non-zero) • "
        f"property tax rate {house_tax_rate:.1f}%."
    )
    st.dataframe(result_house_df, use_container_width=True)

    raw_house_df = pd.DataFrame(raw_rows_house)[[
        "Allocation",
        "Global Minimum Ending Value",
        "SPX Mininimum Ending Value",
        "Global Median Ending Value",
        "SPX Median Ending Value",
    ]]
    st.download_button("Download housing payments invested table (CSV)", data=raw_house_df.to_csv(index=False).encode("utf-8"), file_name=f"spend_this_housing_payments_invested_{years}y.csv", mime="text/csv")

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

median_withdrawal_g = _lookup_median_withdrawal(df_withdrawals, int(retirement_years))
median_withdrawal_s = _lookup_median_withdrawal(df_withdrawals_spx, int(retirement_years))

# Lump Sum median withdrawal
lump_rows = []
if has_lump and not raw_df.empty:
    if median_withdrawal_g is not None:
        base_g = float(np.nanmax(raw_df["Global Median Ending Value"].values)) if not raw_df.empty else np.nan
        ann_g = median_withdrawal_g * base_g if (median_withdrawal_g <= 1.0 and np.isfinite(base_g)) else median_withdrawal_g
        lump_rows.append({"Portfolio": "Global", "Years": int(retirement_years), "Annual Retirement Income (Historically)": ann_g, "Total Median Retirement Income": ann_g * int(retirement_years)})
    if median_withdrawal_s is not None:
        base_s = float(np.nanmax(raw_df["SPX Median Ending Value"].values)) if not raw_df.empty else np.nan
        ann_s = median_withdrawal_s * base_s if (median_withdrawal_s <= 1.0 and np.isfinite(base_s)) else median_withdrawal_s
        lump_rows.append({"Portfolio": "SPX", "Years": int(retirement_years), "Annual Retirement Income (Historically)": ann_s, "Total Median Retirement Income": ann_s * int(retirement_years)})
    if has_lump and not raw_df.empty and lump_rows:
        lump_df = pd.DataFrame(lump_rows)
        lump_disp = lump_df.copy()
        lump_disp["Annual Retirement Income (Historically)"] = lump_disp["Annual Retirement Income (Historically)"].map(lambda v: f"${v:,.0f}")
        lump_disp["Total Median Retirement Income"] = lump_disp["Total Median Retirement Income"].map(lambda v: f"${v:,.0f}")
        st.subheader("Median Withdrawal — Lump Sum")
        st.caption("Ending portfolio uses the highest median ending value across allocations; withdrawals assume a 60% Equity / 40% Fixed portfolio. Expenses: Global 20 bps; SPX 5 bps.")
        st.dataframe(lump_disp, use_container_width=True)
        st.download_button("Download median withdrawal — Lump Sum (CSV)", data=lump_df.to_csv(index=False).encode("utf-8"), file_name=f"median_withdrawal_lumpsum_{years}y.csv", mime="text/csv")
        try:
            for _, r in lump_df.iterrows():
                total_val = float(r["Total Median Retirement Income"])
                p = str(r.get("Portfolio", "")).strip().lower()
                portfolio_label = "the Global portfolio" if p == "global" else "the S&P 500 (SPX) portfolio"
                _msg = (f"Spending ${thinking_spend:,.0f} instead of ${whatif_spend:,.0f} cost you ${total_val:,.0f} in lifetime retirement income by investing in {portfolio_label}.")
                st.markdown(f"<div style='white-space: normal; word-break: normal; overflow-wrap: break-word;'><strong>{_msg}</strong></div>", unsafe_allow_html=True)
        except Exception:
            pass
    else:
        pass

# Annual Habit Only median withdrawal tables (per habit)
for idx, raw_rows_annual in enumerate(raw_rows_annual_list, start=1):
    annual_rows = []
    if median_withdrawal_g is not None and len(raw_rows_annual) > 0:
        try:
            base_g_a = float(np.nanmax([row.get("Global Median Ending Value", np.nan) for row in raw_rows_annual]))
        except Exception:
            base_g_a = np.nan
        ann_g_a = median_withdrawal_g * base_g_a if (median_withdrawal_g <= 1.0 and np.isfinite(base_g_a)) else median_withdrawal_g
        annual_rows.append({"Portfolio": "Global", "Years": int(retirement_years), "Annual Retirement Income (Historically)": ann_g_a, "Total Median Retirement Income": ann_g_a * int(retirement_years)})
    if median_withdrawal_s is not None and len(raw_rows_annual) > 0:
        try:
            base_s_a = float(np.nanmax([row.get("SPX Median Ending Value", np.nan) for row in raw_rows_annual]))
        except Exception:
            base_s_a = np.nan
        ann_s_a = median_withdrawal_s * base_s_a if (median_withdrawal_s <= 1.0 and np.isfinite(base_s_a)) else median_withdrawal_s
        annual_rows.append({"Portfolio": "SPX", "Years": int(retirement_years), "Annual Retirement Income (Historically)": ann_s_a, "Total Median Retirement Income": ann_s_a * int(retirement_years)})
    if annual_rows:
        annual_df = pd.DataFrame(annual_rows)
        annual_disp = annual_df.copy()
        annual_disp["Annual Retirement Income (Historically)"] = annual_disp["Annual Retirement Income (Historically)"].map(lambda v: f"${v:,.0f}")
        annual_disp["Total Median Retirement Income"] = annual_disp["Total Median Retirement Income"].map(lambda v: f"${v:,.0f}")
        st.subheader(f"Median Withdrawal — Annual Habit {idx} Only")
        st.caption("Ending portfolio uses the highest median ending value (annual contributions only) across allocations; withdrawals assume a 60% Equity / 40% Fixed portfolio. Expenses: Global 20 bps; SPX 5 bps.")
        st.dataframe(annual_disp, use_container_width=True)
        st.download_button(f"Download median withdrawal — Annual Habit {idx} Only (CSV)", data=annual_df.to_csv(index=False).encode("utf-8"), file_name=f"median_withdrawal_annual_only_habit{idx}_{years}y.csv", mime="text/csv")
        try:
            h = habits[idx - 1] if idx - 1 < len(habits) else {"daily":0,"frugal":0}
            for _, r in annual_df.iterrows():
                total_val = float(r["Total Median Retirement Income"])
                p = str(r.get("Portfolio", "")).strip().lower()
                portfolio_label = "the Global portfolio" if p == "global" else "the S&P 500 (SPX) portfolio"
                _msg_h = (f"Spending ${h['daily']:,.2f} instead of ${h['frugal']:,.2f} cost you ${total_val:,.0f} in lifetime retirement income by investing in {portfolio_label}.")
                st.markdown(f"<div style='white-space: normal; word-break: normal; overflow-wrap: break-word;'><strong>{_msg_h}</strong></div>", unsafe_allow_html=True)
        except Exception:
            pass
    else:
        st.info(f"Median withdrawals not found for Annual Habit {idx} scenario.")

 # Auto Payments Invested — median withdrawal
if 'raw_auto_df' in locals() and isinstance(raw_auto_df, pd.DataFrame) and not raw_auto_df.empty:
    auto_rows = []
    yrs_int = int(retirement_years)
    if median_withdrawal_g is not None:
        try:
            base_g_auto = float(np.nanmax(raw_auto_df["Global Median Ending Value"].values))
        except Exception:
            base_g_auto = np.nan
        ann_g_auto = median_withdrawal_g * base_g_auto if (median_withdrawal_g <= 1.0 and np.isfinite(base_g_auto)) else float(median_withdrawal_g)
        auto_rows.append({"Portfolio": "Global", "Years": yrs_int, "Annual Retirement Income (Historically)": ann_g_auto, "Total Median Retirement Income": ann_g_auto * yrs_int})
    if median_withdrawal_s is not None:
        try:
            base_s_auto = float(np.nanmax(raw_auto_df["SPX Median Ending Value"].values))
        except Exception:
            base_s_auto = np.nan
        ann_s_auto = median_withdrawal_s * base_s_auto if (median_withdrawal_s <= 1.0 and np.isfinite(base_s_auto)) else float(median_withdrawal_s)
        auto_rows.append({"Portfolio": "SPX", "Years": yrs_int, "Annual Retirement Income (Historically)": ann_s_auto, "Total Median Retirement Income": ann_s_auto * yrs_int})
    if auto_rows:
        auto_med_df = pd.DataFrame(auto_rows)
        auto_med_disp = auto_med_df.copy()
        auto_med_disp["Annual Retirement Income (Historically)"] = auto_med_disp["Annual Retirement Income (Historically)"].map(lambda v: f"${v:,.0f}")
        auto_med_disp["Total Median Retirement Income"] = auto_med_disp["Total Median Retirement Income"].map(lambda v: f"${v:,.0f}")
        st.subheader("Median Withdrawal — Auto Payments Invested")
        st.caption("Ending portfolio uses the highest median ending value (auto payments invested) across allocations; withdrawals assume a 60/40 portfolio. Expenses: Global 20 bps; SPX 5 bps.")
        st.dataframe(auto_med_disp, use_container_width=True)
        st.download_button("Download median withdrawal — Auto Payments Invested (CSV)", data=auto_med_df.to_csv(index=False).encode("utf-8"), file_name=f"median_withdrawal_auto_payments_invested_{years}y.csv", mime="text/csv")

# Housing — median withdrawal table
if 'raw_house_df' in locals() and isinstance(raw_house_df, pd.DataFrame) and not raw_house_df.empty:
    housing_rows = []
    yrs_int = int(retirement_years)
    if median_withdrawal_g is not None:
        try:
            base_g_house = float(np.nanmax(raw_house_df["Global Median Ending Value"].values))
        except Exception:
            base_g_house = np.nan
        ann_g_house = median_withdrawal_g * base_g_house if (median_withdrawal_g <= 1.0 and np.isfinite(base_g_house)) else float(median_withdrawal_g)
        housing_rows.append({"Portfolio": "Global", "Years": yrs_int, "Annual Retirement Income (Historically)": ann_g_house, "Total Median Retirement Income": ann_g_house * yrs_int})
    if median_withdrawal_s is not None:
        try:
            base_s_house = float(np.nanmax(raw_house_df["SPX Median Ending Value"].values))
        except Exception:
            base_s_house = np.nan
        ann_s_house = median_withdrawal_s * base_s_house if (median_withdrawal_s <= 1.0 and np.isfinite(base_s_house)) else float(median_withdrawal_s)
        housing_rows.append({"Portfolio": "SPX", "Years": yrs_int, "Annual Retirement Income (Historically)": ann_s_house, "Total Median Retirement Income": ann_s_house * yrs_int})
    if housing_rows:
        housing_df = pd.DataFrame(housing_rows)
        housing_disp = housing_df.copy()
        housing_disp["Annual Retirement Income (Historically)"] = housing_disp["Annual Retirement Income (Historically)"].map(lambda v: f"${v:,.0f}")
        housing_disp["Total Median Retirement Income"] = housing_disp["Total Median Retirement Income"].map(lambda v: f"${v:,.0f}")
        st.subheader("Median Withdrawal — Housing")
        st.caption("Housing difference stream includes payment differences (yearly, end-of-year), property tax difference each year, and down payment difference (year 0, beginning-of-year). Withdrawals assume a 60/40 portfolio; expenses: Global 20 bps; SPX 5 bps.")
        st.dataframe(housing_disp, use_container_width=True)
        st.download_button("Download median withdrawal — Housing (CSV)", data=housing_df.to_csv(index=False).encode("utf-8"), file_name=f"median_withdrawal_housing_{years}y.csv", mime="text/csv")
        try:
            for _, r in housing_df.iterrows():
                total_val = float(r["Total Median Retirement Income"])
                p = str(r.get("Portfolio", "")).strip().lower()
                portfolio_label = "the Global portfolio" if p == "global" else "the S&P 500 (SPX) portfolio"
                _msg_housing = (
                    f"Choosing a ${house_spender_price:,.0f} home instead of ${house_frugal_price:,.0f} home "
                    f"cost you ${total_val:,.0f} in lifetime retirement income by investing in {portfolio_label}."
                )
                st.markdown(
                    f"<div style='white-space: normal; word-break: normal; overflow-wrap: break-word;'><strong>{_msg_housing}</strong></div>",
                    unsafe_allow_html=True
                )
        except Exception:
            pass

 # Grand Total median withdrawal (Lump + All Annual Habits)
try:
    years_int = int(retirement_years)
    annual_lump_g = 0.0
    if median_withdrawal_g is not None and not raw_df.empty:
        base_g = float(np.nanmax(raw_df["Global Median Ending Value"].values))
        annual_lump_g = (median_withdrawal_g * base_g) if (median_withdrawal_g <= 1.0 and np.isfinite(base_g)) else float(median_withdrawal_g)
    annual_lump_s = 0.0
    if median_withdrawal_s is not None and not raw_df.empty:
        base_s = float(np.nanmax(raw_df["SPX Median Ending Value"].values))
        annual_lump_s = (median_withdrawal_s * base_s) if (median_withdrawal_s <= 1.0 and np.isfinite(base_s)) else float(median_withdrawal_s)

    habits_total_g = 0.0
    habits_total_s = 0.0
    for raw_rows_annual in raw_rows_annual_list:
        if median_withdrawal_g is not None and len(raw_rows_annual) > 0:
            try:
                base_g_a = float(np.nanmax([row.get("Global Median Ending Value", np.nan) for row in raw_rows_annual]))
            except Exception:
                base_g_a = np.nan
            habits_total_g += (median_withdrawal_g * base_g_a) if (median_withdrawal_g <= 1.0 and np.isfinite(base_g_a)) else float(median_withdrawal_g)
        if median_withdrawal_s is not None and len(raw_rows_annual) > 0:
            try:
                base_s_a = float(np.nanmax([row.get("SPX Median Ending Value", np.nan) for row in raw_rows_annual]))
            except Exception:
                base_s_a = np.nan
            habits_total_s += (median_withdrawal_s * base_s_a) if (median_withdrawal_s <= 1.0 and np.isfinite(base_s_a)) else float(median_withdrawal_s)

    # Auto (difference) annual incomes
    auto_annual_g = 0.0
    if 'raw_auto_df' in locals() and isinstance(raw_auto_df, pd.DataFrame) and not raw_auto_df.empty and (median_withdrawal_g is not None):
        try:
            base_g_auto = float(np.nanmax(raw_auto_df["Global Median Ending Value"].values))
        except Exception:
            base_g_auto = np.nan
        if median_withdrawal_g <= 1.0 and np.isfinite(base_g_auto):
            auto_annual_g = median_withdrawal_g * base_g_auto
        else:
            auto_annual_g = float(median_withdrawal_g)

    auto_annual_s = 0.0
    if 'raw_auto_df' in locals() and isinstance(raw_auto_df, pd.DataFrame) and not raw_auto_df.empty and (median_withdrawal_s is not None):
        try:
            base_s_auto = float(np.nanmax(raw_auto_df["SPX Median Ending Value"].values))
        except Exception:
            base_s_auto = np.nan
        if median_withdrawal_s <= 1.0 and np.isfinite(base_s_auto):
            auto_annual_s = median_withdrawal_s * base_s_auto
        else:
            auto_annual_s = float(median_withdrawal_s)

    # Housing annual incomes
    housing_annual_g = 0.0
    if 'raw_house_df' in locals() and isinstance(raw_house_df, pd.DataFrame) and not raw_house_df.empty and (median_withdrawal_g is not None):
        try:
            base_g_house = float(np.nanmax(raw_house_df["Global Median Ending Value"].values))
        except Exception:
            base_g_house = np.nan
        housing_annual_g = (median_withdrawal_g * base_g_house) if (median_withdrawal_g <= 1.0 and np.isfinite(base_g_house)) else float(median_withdrawal_g)

    housing_annual_s = 0.0
    if 'raw_house_df' in locals() and isinstance(raw_house_df, pd.DataFrame) and not raw_house_df.empty and (median_withdrawal_s is not None):
        try:
            base_s_house = float(np.nanmax(raw_house_df["SPX Median Ending Value"].values))
        except Exception:
            base_s_house = np.nan
        housing_annual_s = (median_withdrawal_s * base_s_house) if (median_withdrawal_s <= 1.0 and np.isfinite(base_s_house)) else float(median_withdrawal_s)

    # Skip grand total if all components are zero
    if not any([
        annual_lump_g, habits_total_g, auto_annual_g, housing_annual_g,
        annual_lump_s, habits_total_s, auto_annual_s, housing_annual_s,
    ]):
        raise RuntimeError("No components to summarize")

    grand_rows = [
        {
            "Portfolio": "Global",
            "Years": years_int,
            "Annual Retirement Income — Lump Sum": annual_lump_g,
            "Annual Retirement Income — Habits Total": habits_total_g,
            "Annual Retirement Income — Auto": auto_annual_g,
            "Annual Retirement Income — Housing": housing_annual_g,
            "Annual Retirement Income — Grand Total": annual_lump_g + habits_total_g + auto_annual_g + housing_annual_g,
            "Total — Lump Sum": annual_lump_g * years_int,
            "Total — Habits Total": habits_total_g * years_int,
            "Total — Auto": auto_annual_g * years_int,
            "Total — Housing": housing_annual_g * years_int,
            "Total — Grand Total": (annual_lump_g + habits_total_g + auto_annual_g + housing_annual_g) * years_int,
        },
        {
            "Portfolio": "SPX",
            "Years": years_int,
            "Annual Retirement Income — Lump Sum": annual_lump_s,
            "Annual Retirement Income — Habits Total": habits_total_s,
            "Annual Retirement Income — Auto": auto_annual_s,
            "Annual Retirement Income — Housing": housing_annual_s,
            "Annual Retirement Income — Grand Total": annual_lump_s + habits_total_s + auto_annual_s + housing_annual_s,
            "Total — Lump Sum": annual_lump_s * years_int,
            "Total — Habits Total": habits_total_s * years_int,
            "Total — Auto": auto_annual_s * years_int,
            "Total — Housing": housing_annual_s * years_int,
            "Total — Grand Total": (annual_lump_s + habits_total_s + auto_annual_s + housing_annual_s) * years_int,
        },
    ]

    grand_df = pd.DataFrame(grand_rows)
    # Dynamically drop columns that are entirely zero/NaN (except identifiers)
    id_cols = ["Portfolio", "Years"]
    numeric_cols = [c for c in grand_df.columns if c not in id_cols]
    nz_mask = grand_df[numeric_cols].fillna(0).astype(float).abs().sum(axis=0) > 0
    keep_cols = id_cols + [c for c in numeric_cols if nz_mask.get(c, False)]
    grand_df = grand_df[keep_cols]

    # Format for display only for kept numeric columns
    grand_disp = grand_df.copy()
    for col in [c for c in keep_cols if c not in id_cols]:
        grand_disp[col] = grand_disp[col].map(lambda v: f"${v:,.0f}")

    st.subheader("Median Withdrawal — Grand Total (Lump + All Annual Habits + Auto + Housing)")
    st.dataframe(grand_disp, use_container_width=True)
    st.download_button(
        "Download median withdrawal — Grand Total (CSV)",
        data=grand_df.to_csv(index=False).encode("utf-8"),
        file_name=f"median_withdrawal_grand_total_{years}y.csv",
        mime="text/csv"
    )
except Exception:
    pass