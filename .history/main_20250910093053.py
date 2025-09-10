import streamlit as st
import pandas as pd

st.set_page_config(page_title="First 10 Begin Months", layout="centered")

def read_begin_month(csv_path: str) -> pd.Series:
    """
    Load a CSV and return a parsed datetime Series for the 'Begin Month' column.
    Accepts case-insensitive / underscore variants like 'begin month' or 'Begin_Month'.
    """
    df = pd.read_csv(csv_path)
    # find the column case-insensitively
    candidates = [c for c in df.columns if str(c).strip().lower().replace("_", " ") == "begin month"]
    if not candidates:
        raise ValueError(
            f"'{csv_path}' must contain a 'Begin Month' column "
            "(case-insensitive; underscores allowed, e.g., 'Begin_Month')."
        )
    s = pd.to_datetime(df[candidates[0]], errors="raise")
    return s.sort_values().reset_index(drop=True)

st.title("First 10 Begin Months — Global vs SPX")

col1, col2 = st.columns(2)
with col1:
    global_path = st.text_input("Path to global_factors.csv", "global_factors.csv")
with col2:
    spx_path = st.text_input("Path to spx_factors.csv", "spx_factors.csv")

try:
    global_begin = read_begin_month(global_path)
    st.subheader("Global (first 10)")
    st.write(global_begin.head(10).dt.date.tolist())
    st.caption(f"Earliest: {global_begin.min().date()}  •  Latest: {global_begin.max().date()}  •  Total rows: {len(global_begin):,}")
except Exception as e:
    st.error(f"Global error: {e}")

try:
    spx_begin = read_begin_month(spx_path)
    st.subheader("SPX (first 10)")
    st.write(spx_begin.head(10).dt.date.tolist())
    st.caption(f"Earliest: {spx_begin.min().date()}  •  Latest: {spx_begin.max().date()}  •  Total rows: {len(spx_begin):,}")
except Exception as e:
    st.error(f"SPX error: {e}")