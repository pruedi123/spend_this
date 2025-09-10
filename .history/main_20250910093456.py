import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="30-Year Factor Sims", layout="wide")

# ---------------------------
# Functions
# ---------------------------

def load_factors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # drop non-numeric columns if any
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(axis=1, how="all")

def build_windows(df: pd.DataFrame, alloc_col: str, years: int, step: int = 12) -> pd.DataFrame:
    arr = df[alloc_col].values.astype(float)
    max_start = len(arr) - years * step
    rows = []
    if max_start < 0:
        return pd.DataFrame(columns=["start_index","factors"])
    for s in range(0, max_start + 1):
        idxs = s + np.arange(years) * step
        if idxs[-1] >= len(arr):
            break
        window = arr[idxs]
        if np.isnan(window).any():
            continue
        rows.append((s, window))
    return pd.DataFrame(rows, columns=["start_index","factors"])

# ---------------------------
# UI
# ---------------------------

st.title("Factor Simulations")

path = st.text_input("CSV file path", "spx_factors.csv")

try:
    df = load_factors(path)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

alloc = st.selectbox("Choose allocation column", [c for c in df.columns])
years = st.slider("Years to simulate", 5, 40, 30)

sims = build_windows(df, alloc, years)

st.write(f"Total {len(sims)} simulations built for {years} years.")

if not sims.empty:
    # Expand the factor arrays into columns
    expanded = pd.DataFrame(sims["factors"].tolist(), index=sims["start_index"])
    expanded.index.name = "start_index"
    st.dataframe(expanded.head(20), use_container_width=True)
    st.download_button("Download all sims (CSV)",
                       data=expanded.to_csv().encode("utf-8"),
                       file_name=f"{alloc}_{years}y_sims.csv",
                       mime="text/csv")