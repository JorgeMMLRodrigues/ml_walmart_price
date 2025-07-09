import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def filter_df(
    df: pd.DataFrame,
    store_choice: str,
    dept_choice:  str,
    limit_window: str
) -> pd.DataFrame:
    """
    Returns a slice of *df* based on the sidebar selections.
    """
    df_sel = df.copy()

    # Store / Dept filters
    if store_choice != "All Stores":
        df_sel = df_sel[df_sel["store"] == int(store_choice)]
    if dept_choice != "All Departments":
        df_sel = df_sel[df_sel["dept"] == int(dept_choice)]

    # Time-window trim
    if limit_window in {"Last 52 Weeks", "Last 104 Weeks"} and not df_sel.empty:
        weeks = 52 if limit_window == "Last 52 Weeks" else 104
        cutoff = df_sel["date"].max() - pd.Timedelta(weeks=weeks)
        df_sel = df_sel[df_sel["date"] >= cutoff]

    return df_sel


@st.cache_data(show_spinner=False)
def aggregate_timeseries(
    df: pd.DataFrame,
    date_col: str = "date",
    cols: dict[str, str] = None
) -> pd.DataFrame:
    """
    Given df_sel and a dict mapping output names → column names to sum,
    returns a long-form DataFrame with columns [date, Series, Sales].
    """
    if cols is None:
        cols = {"Total": "weekly_sales"}
    df_agg = (
        df
        .groupby(date_col)
        .agg({v: "sum" for v in cols.values()})
        .rename(columns={v: k for k, v in cols.items()})
        .reset_index()
        .melt(id_vars=date_col, var_name="Series", value_name="Sales")
    )
    return df_agg

