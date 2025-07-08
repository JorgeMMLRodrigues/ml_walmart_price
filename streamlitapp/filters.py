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


@st.cache_data(show_spinner=False)
def compute_rf_trends(
    df: pd.DataFrame,
    date_col: str = "date",
    actual_col: str = "weekly_sales",
    rf_col: str    = "rf_02_predicted_weekly_sales",
    metric: str    = "total"
) -> dict:
    """
    Returns everything needed for Actual vs. RF plots:

      • ts_act, ts_rf  → pd.Series indexed by date (the transformed series)
      • df_pred        → combined DataFrame with columns [date, Series, value]
      • delta_act, delta_rf  → the start→end change (or latest for YOY)
      • y_title        → axis label for plotting

    metric: "total", "yoy_pct", or "yoy_diff"
    """
    # raw aggregates
    ts_act_raw = df.groupby(date_col)[actual_col].sum().sort_index()
    ts_rf_raw  = df.groupby(date_col)[rf_col].sum().sort_index()

    # 52-week slice sums
    cutoff     = df[date_col].max() - pd.Timedelta(weeks=52)
    df_last52  = df[df[date_col] >= cutoff]
    last52_act = df_last52[actual_col].sum()
    last52_rf  = df_last52[rf_col].sum()

    # transform per metric
    if metric == "total":
        ts_act, ts_rf = ts_act_raw, ts_rf_raw
        y_title       = "Total Weekly Sales"
    elif metric == "yoy_pct":
        ts_act = ts_act_raw.pct_change(52) * 100
        pred_pct = (last52_rf - last52_act) / last52_act * 100 if last52_act else float("nan")
        ts_rf = ts_rf_raw.pct_change(52) * 100
        y_title = "YOY % Change"
    else:  # metric == "yoy_diff"
        ts_act = ts_act_raw.diff(52)
        pred_diff = last52_rf - last52_act
        ts_rf = ts_rf_raw.diff(52)
        y_title = "YOY $ Change"

    # drop NaNs
    ts_act = ts_act.dropna()
    ts_rf  = ts_rf.dropna()

    # deltas
    if metric == "yoy_pct":
        delta_act = ts_act.iloc[-1]
        delta_rf  = pred_pct
    else:
        delta_act = ts_act.iloc[-1] - ts_act.iloc[0]
        delta_rf  = ts_rf.iloc[-1]  - ts_rf.iloc[0]

    # combined DF for plotting
    df_pred = pd.concat([
        ts_act.rename("value").reset_index().assign(Series="Actual"),
        ts_rf .rename("value").reset_index().assign(Series="RandomForest")
    ], ignore_index=True)

    return {
        "ts_act":     ts_act,
        "ts_rf":      ts_rf,
        "df_pred":    df_pred,
        "delta_act":  delta_act,
        "delta_rf":   delta_rf,
        "y_title":    y_title,
    }
