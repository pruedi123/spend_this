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
      - CPI level computed by multiplying the deflators
    """
    inc = np.asarray(increase_factor, dtype=float)
    # guard bad values: NaN/nonpositive treated as 1.0 (no change)
    inc = np.where(np.isfinite(inc) & (inc > 0), inc, 1.0)

    start_idx = max(0, int(start_row_1_based) - 1)  # convert 1-based -> 0-based
    years_at = list(range(0, horizon_years, freq_years))  # 0,5,10,15,20,25,30

    rows = []
    # price level starts at 1.0 at t=0; for year t, multiply deflators at indices start_idx + j*12 for j=0..t-1
    for k, t in enumerate(years_at):
        # compute product of deflators at 12-month steps
        if t == 0:
            level = 1.0
        else:
            idxs = start_idx + (np.arange(0, t) * 12)
            if idxs[-1] >= inc.size:
                # Not enough CPI data—stop
                break
            level = float(np.prod(inc[idxs]))
        price_t = base_price * level
        rows.append({"purchase": k, "year": t, "cpi_level": level, "sticker_price": price_t})

    return pd.DataFrame(rows)

# --- EXAMPLE USAGE -----------------------------------------------------------

# Deterministic same-directory loader (no Excel dependency)
here = Path(__file__).parent if '__file__' in globals() else Path.cwd()
preferred = here / "increase_factors.csv"
fallback  = here / "cpi_increase.csv"
increase = None
used_path = None
used_col = None

for p in (preferred, fallback):
    if p.exists():
        df_inc = pd.read_csv(p)
        # choose column: prefer exact 'increase_factor', else auto-detect a similar name
        col = "increase_factor" if "increase_factor" in df_inc.columns else None
        if col is None:
            for c in df_inc.columns:
                s = str(c).strip().lower().replace(" ", "")
                if "increase" in s and "factor" in s:
                    col = c
                    break
        if col is None:
            raise KeyError(f"No 'increase_factor' column found in {p.name}")
        increase = pd.to_numeric(df_inc[col], errors="coerce").fillna(1.0).to_numpy()
        used_path = str(p)
        used_col = col
        break

if increase is None:
    raise FileNotFoundError(
        "Could not find increase_factors.csv or cpi_increase.csv in the same directory as this script."
    )
else:
    print(f"Loaded CPI deflator from: {used_path} (column: {used_col}, n={increase.size})")

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

# --- Quick check with user-specified example factors ---
example_five = np.array([0.9885, 0.9884, 0.9941, 0.9053, 0.8954], dtype=float)
prod_five = float(np.prod(example_five))
price_example = 35000.0 * prod_five
print("\nExpected 5-year cumulative deflator (user example):", prod_five)
print("Expected sticker at year 5:", f"${price_example:,.5f}")