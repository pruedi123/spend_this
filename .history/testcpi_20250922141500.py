import numpy as np
import pandas as pd
from pathlib import Path

def build_purchase_table(increase_factor: np.ndarray,
                         base_price: float,
                         start_row_1_based: int = 1,
                         freq_years: int = 5,
                         horizon_years: int = 35) -> pd.DataFrame:
    """
    increase_factor: 1D np.array of monthly deflators (e.g., 0.988506, 0.971591, ...)
    base_price:      today's dollars sticker price (e.g., 35000)
    start_row_1_based: simulation start row (1 = first CPI row)
    freq_years:      replace every N years (default 5)
    horizon_years:   length of simulation (default 35)

    Returns a table with one row per purchase showing:
      - purchase index (0,1,2,...), calendar year offset, CPI level, nominal sticker
    """
    inc = np.asarray(increase_factor, dtype=float)
    # guard bad values: NaN/nonpositive treated as 1.0 (no change)
    inc = np.where(np.isfinite(inc) & (inc > 0), inc, 1.0)

    start_idx = max(0, int(start_row_1_based) - 1)  # convert 1-based -> 0-based
    years_at = list(range(0, horizon_years, freq_years))  # 0,5,10,15,20,25,30

    rows = []
    # price level starts at 1.0 at t=0; for year t, multiply 1/inc at indices start_idx + (k-1)*12 for k=1..t
    for k, t in enumerate(years_at):
        # compute product of 1/inc at 12-month steps
        if t == 0:
            level = 1.0
        else:
            idxs = start_idx + (np.arange(0, t) * 12)
            if idxs[-1] >= inc.size:
                # Not enough CPI data—stop
                break
            level = float(np.prod(1.0 / inc[idxs]))
        price_t = base_price * level
        rows.append({"purchase": k, "year": t, "cpi_level": level, "sticker_price": price_t})

    return pd.DataFrame(rows)

# --- EXAMPLE USAGE -----------------------------------------------------------

# Option A: load monthly increase_factor from a CSV (no openpyxl needed)
#   Expect a column named 'increase_factor' (or adjust the column name below)
increase_csv_candidates = ["increase_factors.csv", "cpi_increase.csv"]
increase = None
for p in increase_csv_candidates:
    if Path(p).exists():
        df_inc = pd.read_csv(p)
        # try several likely column names
        col = None
        for c in df_inc.columns:
            s = str(c).strip().lower().replace(" ", "")
            if "increase" in s and "factor" in s:
                col = c; break
        if col is None:  # last-ditch fallback
            col = "increase_factor"
        increase = pd.to_numeric(df_inc[col], errors="coerce").fillna(1.0).to_numpy()
        break

# Option B: if you already have the monthly array in memory, just set `increase = your_array`

if increase is None:
    # Fallback dummy series so the snippet runs; replace with real data
    # (e.g., ~1% inflation per year → monthly deflator ~ 1.01**(1/12) ≈ 1.00083)
    increase = np.full(600, 1.00083)  # 50 years of monthly deflators

base_price = 35000.0
freq_years = 5
horizon_years = 35

# Simulation 1 — begin at row 1
sim1 = build_purchase_table(increase, base_price, start_row_1_based=1,
                            freq_years=freq_years, horizon_years=horizon_years)

# Simulation 2 — begin at row 2
sim2 = build_purchase_table(increase, base_price, start_row_1_based=2,
                            freq_years=freq_years, horizon_years=horizon_years)

print("=== Simulation 1 (start row 1) ===")
print(sim1.to_string(index=False, formatters={
    "cpi_level": lambda v: f"{v:.6f}",
    "sticker_price": lambda v: f"${v:,.0f}",
}))

print("\n=== Simulation 2 (start row 2) ===")
print(sim2.to_string(index=False, formatters={
    "cpi_level": lambda v: f"{v:.6f}",
    "sticker_price": lambda v: f"${v:,.0f}",
}))