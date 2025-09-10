import pandas as pd
import numpy as np

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

    # Keep date + any numeric allocation-like columns
    out = pd.DataFrame({"date": df["date"]})
    for c in df.columns:
        if c == "date":
            continue
        # keep numeric columns only
        out[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort, reset index
    out = out.sort_values("date").reset_index(drop=True)
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
        raise ValueError(f"Allocation column '{alloc_col}' not found.")

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
    # e.g., '60E' or the exact column name in your CSV for the 60% equity series
    result = thirty_year_windows(df_global, alloc_col="60E", years=30, step=12)

    # Peek
    print(result.head(3))
    # Example: first window’s 30 factors
    if not result.empty:
        first_factors = result.loc[0, "factors"]
        print("First window:", result.loc[0, "start_date"].date(), "→", result.loc[0, "end_date"].date())
        print("30 factors:", first_factors)
        print("FV multiple of $1:", result.loc[0, "fv_multiple"])