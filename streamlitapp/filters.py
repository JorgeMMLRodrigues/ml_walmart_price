
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def filter_df(
    df: pd.DataFrame,
    store_choice: str,
    dept_choice: str,
    rf_only: bool,
    limit_window: str
) -> pd.DataFrame:
    # …existing logic…
    return df

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
        .rename(columns={v: k for k,v in cols.items()})
        .reset_index()
        .melt(id_vars=date_col, var_name="Series", value_name="Sales")
    )
    return df_agg
