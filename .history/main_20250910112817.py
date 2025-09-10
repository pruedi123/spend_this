# ===========================
# Median Withdrawal Summary
# ===========================
def _lookup_withdrawal_row(df_w: pd.DataFrame | None, yrs: int):
    if df_w is None:
        return None
    try:
        row = df_w.loc[df_w["Years"] == yrs].iloc[0]
        return {
            "Min": float(row.get("Min")),
            "Median": float(row.get("Median")),
        }
    except Exception:
        return None

# Use the Global withdrawals table for the historical median income stream
w_g = _lookup_withdrawal_row(df_withdrawals, retirement_years)
if w_g is not None and w_g.get("Median") is not None:
    annual_income = float(w_g["Median"])
    total_income = annual_income * int(retirement_years)
    summary_df = pd.DataFrame([{
        "Number of Retirement Years": int(retirement_years),
        "Annual Income Stream (Historically)": annual_income,
        "Total Median (Historical) Retirement Income": total_income,
    }])
    # Pretty formatting for display
    summary_fmt = summary_df.copy()
    summary_fmt["Annual Income Stream (Historically)"] = summary_fmt["Annual Income Stream (Historically)"].map(lambda v: f"${v:,.0f}")
    summary_fmt["Total Median (Historical) Retirement Income"] = summary_fmt["Total Median (Historical) Retirement Income"].map(lambda v: f"${v:,.0f}")
    st.subheader("Median Withdrawal Summary")
    st.dataframe(summary_fmt, use_container_width=True)
else:
    st.info("No median withdrawal found in withdrawals.csv for the selected retirement years.")