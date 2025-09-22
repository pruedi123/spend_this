import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys

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

def load_increase_series(csv_path: str | None = None,
                         xlsx_path: str | None = None,
                         sheet: str = "increase_factors",
                         col: str = "increase_factor") -> tuple[np.ndarray, str, str]:
    """Load monthly increase_factor deflators.
    Returns: (increase_array, source_meta, used_col)
    """
    # 1) CSV explicit path
    if csv_path:
        df = pd.read_csv(csv_path)
        use_col = col if col in df.columns else None
        if use_col is None:
            for c in df.columns:
                s = str(c).strip().lower().replace(" ", "")
                if "increase" in s and "factor" in s:
                    use_col = c
                    break
        if use_col is None:
            raise KeyError(f"No '{col}'-like column found in CSV: {csv_path}")
        inc = pd.to_numeric(df[use_col], errors="coerce").fillna(1.0).to_numpy()
        return inc, f"csv:{csv_path}", str(use_col)

    # 2) XLSX explicit path (requires openpyxl)
    if xlsx_path:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sheet)
        except Exception as e:
            raise RuntimeError(f"Failed to read XLSX (need openpyxl?): {e}")
        use_col = col if col in df.columns else None
        if use_col is None:
            for c in df.columns:
                s = str(c).strip().lower().replace(" ", "")
                if "increase" in s and "factor" in s:
                    use_col = c
                    break
        if use_col is None:
            raise KeyError(f"No '{col}'-like column found in XLSX sheet '{sheet}'")
        inc = pd.to_numeric(df[use_col], errors="coerce").fillna(1.0).to_numpy()
        return inc, f"xlsx:{xlsx_path}:{sheet}", str(use_col)

    # 3) Same-directory CSVs
    here = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    for p in (here / "increase_factors.csv", here / "cpi_increase.csv"):
        if p.exists():
            df = pd.read_csv(p)
            use_col = col if col in df.columns else None
            if use_col is None:
                for c in df.columns:
                    s = str(c).strip().lower().replace(" ", "")
                    if "increase" in s and "factor" in s:
                        use_col = c
                        break
            if use_col is None:
                continue
            inc = pd.to_numeric(df[use_col], errors="coerce").fillna(1.0).to_numpy()
            return inc, f"csv:{p}", str(use_col)

    raise FileNotFoundError("Provide --csv path/to/increase_factors.csv or --xlsx path/to/cpi_factors.xlsx (with --sheet/--col), or place increase_factors.csv next to this script.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Car purchase CPI chaining demo")
    ap.add_argument("--csv", type=str, default=None, help="Path to CSV containing monthly increase_factor column")
    ap.add_argument("--xlsx", type=str, default=None, help="Path to XLSX workbook (requires openpyxl)")
    ap.add_argument("--sheet", type=str, default="increase_factors", help="Sheet name for XLSX (default: increase_factors)")
    ap.add_argument("--col", type=str, default="increase_factor", help="Column name for deflator (default: increase_factor)")
    ap.add_argument("--base", type=float, default=35000.0, help="Base sticker price in today's dollars")
    ap.add_argument("--freq", type=int, default=5, help="Replacement frequency in years (default 5)")
    ap.add_argument("--horizon", type=int, default=35, help="Horizon in years (default 35)")
    ap.add_argument("--start", type=int, default=1, help="Simulation start row (1-based; default 1)")
    ap.add_argument("--start2", type=int, default=2, help="Second simulation start row (default 2)")
    args = ap.parse_args()

    try:
        increase, meta, used_col = load_increase_series(csv_path=args.csv, xlsx_path=args.xlsx, sheet=args.sheet, col=args.col)
    except Exception as e:
        print(f"Failed to load CPI series: {e}", file=sys.stderr)
        sys.exit(1)

    base_price = float(args.base)
    freq_years = int(args.freq)
    horizon_years = int(args.horizon)

    sim1 = build_purchase_table(increase, base_price, start_row_1_based=int(args.start),
                                freq_years=freq_years, horizon_years=horizon_years)
    sim2 = build_purchase_table(increase, base_price, start_row_1_based=int(args.start2),
                                freq_years=freq_years, horizon_years=horizon_years)

    print(f"Loaded CPI deflator from: {meta} (column: {used_col}, n={increase.size})")

    print("\n=== Simulation 1 (start row {:d}) ===".format(int(args.start)))
    print(sim1.to_string(index=False, formatters={
        "cpi_level": lambda v: f"{v:.6f}",
        "sticker_price": lambda v: f"${v:,.0f}",
    }))

    print("\n=== Simulation 2 (start row {:d}) ===".format(int(args.start2)))
    print(sim2.to_string(index=False, formatters={
        "cpi_level": lambda v: f"{v:.6f}",
        "sticker_price": lambda v: f"${v:,.0f}",
    }))

    # Quick check with user's five factors
    example_five = np.array([0.9885, 0.9884, 0.9941, 0.9053, 0.8954], dtype=float)
    prod_five = float(np.prod(example_five))
    price_example = base_price * prod_five
    print("\nCheck (user example, 5-year product):", prod_five)
    print("Sticker at 5 years on base:", f"${price_example:,.5f}")