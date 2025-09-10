import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Global vs SPX 30-Year Sims", layout="wide")

# ---------------------------
# Functions
# ---------------------------

def load_factors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure all numeric cols are numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop empty cols
    df = df.dropna(axis=1, how="all")
    return df

def build_windows(df: pd.DataFrame, alloc_col: str, years: int, step: int = 12) -> pd.DataFrame:
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
        fv = float(np.prod(window))  # product of factors
        rows.append((s, window, fv))
    return pd.DataFrame(rows, columns=["start_index","factors","fv_multiple"])

# ---------------------------
# UI
# ---------------------------

st.title("All Available Multi-Year Simulations")

dataset = st.radio("Choose dataset", ["Global", "SPX"], horizontal=True)
if dataset == "Global":
    path = "global_factors.csv"
else:
    path = "spx_factors.csv"

try:
    df = load_factors(path)
except Exception as e:
    st.error(f"Error loading file {path}: {e}")
    st.stop()

alloc = st.selectbox("Choose allocation column", [c for c in df.columns])
years = st.slider("Years to simulate", 5, 40, 30)

sims = build_windows(df, alloc, years)

st.write(f"Total {len(sims)} {years}-year simulations built from {dataset}.")

if not sims.empty:
    # Expand factors into columns for display
    expanded = pd.DataFrame(sims["factors"].tolist(), index=sims["start_index"])
    expanded.index.name = "start_index"
    expanded["fv_multiple"] = sims["fv_multiple"].values
    st.dataframe(expanded.head(20), use_container_width=True)
    st.download_button("Download all sims (CSV)",
                       data=expanded.to_csv().encode("utf-8"),
                       file_name=f"{dataset}_{alloc}_{years}y_sims.csv",
                       mime="text/csv")