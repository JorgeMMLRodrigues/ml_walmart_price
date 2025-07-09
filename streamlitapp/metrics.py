import pandas as pd
from typing import Tuple, Dict, List
import streamlit as st
import numpy as np


def total_sales(
    df: pd.DataFrame,
    col: str = "weekly_sales"
) -> float:
    """
    Sum up the specified column.
    """
    return float(df[col].sum())

@st.cache_data
def load_rf_summary(path: str) -> Tuple[float, float]:
    df = pd.read_csv(path)
    return float(df.at[0, "mae"]), float(df.at[0, "r2"])


def span_stats(
    df: pd.DataFrame,
    date_col: str = "date",
    group_cols: List[str] = ["store", "dept"]
) -> Dict[str, object]:
    start, end = df[date_col].min(), df[date_col].max()
    counts = {}
    for col in group_cols:
        counts[col] = (
            df.loc[df[date_col] == start, col].nunique(),
            df.loc[df[date_col] == end,   col].nunique(),
        )
    return {"start_date": start, "end_date": end, "counts": counts}


def week_span(
    df: pd.DataFrame,
    date_col: str = "date",
    limit_window: str | None = None
) -> int:
    if limit_window == "Last 52 Weeks":
        return 52
    if limit_window == "Last 104 Weeks":
        return 104
    return int(df[date_col].nunique())


def build_ranking_table(
    df: pd.DataFrame,
    group_col: str,
    window_weeks: int = 52,
    date_col: str = "date"
) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    def _yoy(sub: pd.DataFrame) -> tuple[float, float]:
        # need at least two windows of data
        if sub[date_col].nunique() < 2 * window_weeks:
            return np.nan, np.nan

        end = sub[date_col].max()
        # sum over the most recent window_weeks
        latest = sub.loc[
            sub[date_col] > end - pd.Timedelta(weeks=window_weeks),
            "weekly_sales"
        ].sum()
        # sum over the prior window_weeks
        prev = sub.loc[
            (sub[date_col] <= end - pd.Timedelta(weeks=window_weeks)) &
            (sub[date_col] >  end - pd.Timedelta(weeks=2 * window_weeks)),
            "weekly_sales"
        ].sum()

        diff = latest - prev
        pct  = (diff / prev * 100) if prev else np.nan
        return diff, pct

    # total sales by group
    total = df.groupby(group_col)["weekly_sales"].sum().rename("total")

    # YOY by group
    yoy = (
        df
        .groupby(group_col)
        .apply(lambda g: pd.Series(_yoy(g), index=["yoy_diff", "yoy_pct"]))
    )

    return pd.concat([total, yoy], axis=1).reset_index()

