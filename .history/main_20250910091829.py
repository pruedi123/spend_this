import pandas as pd
import numpy as np
import re

def _canonical_alloc_name(col: str) -> str | None:
    """
    Map various header styles to a canonical equity label like '0E','10E',...,'100E'.
    Rules:
      - '100F' (100% fixed) -> '0E'
      - detect digits next to 'E' (case-insensitive): 'LBM 60E', 'spx60e' -> '60E'
      - fallback: trailing digits -> '<N>E' if looks like equity pct
    Returns None if it doesn't look like an allocation column.
    """
    s = str(col).strip().lower().replace("_", "").replace(" ", "")
    # 100F => 0E
    if s.endswith("100f"):
        return "0E"
    # find number followed by 'e'
    m = re.search(r"(\d{1,3})e$", s)
    if m:
        pct = max(0, min(100, int(m.group(1))))
        return f"{pct}E"
    # sometimes it's like 'lbm60' without E -> treat as equity pct
    m2 = re.search(r"(\d{1,3})$", s)
    if m2 and ("lbm" in s or "spx" in s or "glob" in s or "global" in s):
        pct = max(0, min(100, int(m2.group(1))))
        return f"{pct}E"
    return None

def load_factors_csv(path: str) -> pd.DataFrame:
    """
    Load a factors CSV and return a DataFrame with:
      - 'date' column from 'Begin Month' (strict)
      - allocation columns like '0E','10E',...,'100E' (numbers may be embedded)
    """
    df = pd.read_csv(path)
    # Strict: require Begin Month
    if "Begin Month" not in df.columns and "begin month" not in [c.lower() for c in df.columns]:
        raise ValueError("CSV must contain a 'Begin Month' column.")

    # Normalize date
    date_col = "Begin Month" if "Begin Month" in df.columns else \
               [c for c in df.columns if c.lower() == "begin month"][0]
    df = df.rename(columns={date_col: "date"}).copy()
    df["date"] = pd.to_datetime(df["date"])

    # Keep date + numeric allocations, renamed to canonical labels when possible
    out = pd.DataFrame({"date": df["date"]})
    for c in df.columns:
        if c == "date":
            continue
        val = pd.to_numeric(df[c], errors="coerce")
        canon = _canonical_alloc_name(c)
        if canon is None:
            # keep original header if it's numeric series but unknown naming
            out[c] = val
        else:
            out[canon] = val

    # Sort, reset index
    out = out.sort_values("date").reset_index(drop=True)
    # Drop columns that are entirely NaN (non-series)
    out = out.dropna(axis=1, how="all")
    if set(out.columns) == {"date"}:
        raise ValueError("No allocation columns detected after normalization.")
    return out

def thirty_year_windows(df: pd.DataFrame, alloc_col: str, years: int = 30, step: int = 12) -> pd.DataFrame:
    """
    Build 30-year windows for every possible begin month.

    Returns a DataFrame with:
      - start_date
      - end_date
      - factors: numpy array of length 30
      - fv_multiple: product of the 30 factors (FV of $1)
    Skips windows that don't have exactly 30 clean values.
    """
    if alloc_col not in df.columns:
        raise ValueError(f"Allocation column '{alloc_col}' not found. Available: {', '.join([c for c in df.columns if c != 'date'])}")

    arr = df[alloc_col].values.astype(float)
    dates = df["date"].values
    needed = years * step
    max_start = len(arr) - needed
    rows = []
    if max_start < 0:
        return pd.DataFrame(columns=["start_date","end_date","factors","fv_multiple"])

    for s in range(0, max_start + 1):
        idxs = s + np.arange(years) * step   # s, s+12, s+24, ..., s+12*(years-1)
        window = arr[idxs]
        if np.isnan(window).any():
            continue
        fv = float(np.prod(window))  # FV multiple of $1
        start_date = pd.to_datetime(dates[s])
        end_date   = pd.to_datetime(dates[idxs[-1]])
        rows.append((start_date, end_date, window, fv))

    return pd.DataFrame(rows, columns=["start_date","end_date","factors","fv_multiple"])

# ---------- Example usage ----------
if __name__ == "__main__":
    # Choose one file and one allocation column present in that file
    df_global = load_factors_csv("global_factors.csv")
    # pick 60E if present; otherwise nearest to 60E; otherwise first non-date column
    alloc_candidates = [c for c in df_global.columns if c != "date"]
    if "60E" in alloc_candidates:
        alloc_pick = "60E"
    else:
        # find columns like 'NE'
        nums = []
        for c in alloc_candidates:
            if c.endswith("E"):
                try:
                    nums.append((abs(int(c[:-1]) - 60), c))
                except Exception:
                    continue
        alloc_pick = sorted(nums)[0][1] if nums else alloc_candidates[0]
    print("Using allocation column:", alloc_pick)
    result = thirty_year_windows(df_global, alloc_col=alloc_pick, years=30, step=12)

    # Peek
    print(result.head(3))
    # Example: first window’s 30 factors
    if not result.empty:
        first_factors = result.loc[0, "factors"]
        print("First window:", result.loc[0, "start_date"].date(), "→", result.loc[0, "end_date"].date())
        print("30 factors:", first_factors)
        print("FV multiple of $1:", result.loc[0, "fv_multiple"])