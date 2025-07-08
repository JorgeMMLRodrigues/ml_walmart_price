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
    date_col: str = "date"
) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    def _yoy(sub: pd.DataFrame) -> Tuple[float, float]:
        if sub[date_col].nunique() < 104:
            return np.nan, np.nan
        end = sub[date_col].max()
        latest = sub[sub[date_col] > end - pd.Timedelta(weeks=52)]["weekly_sales"].sum()
        prev   = sub[(sub[date_col] <= end - pd.Timedelta(weeks=52)) &
                     (sub[date_col] > end - pd.Timedelta(weeks=104))]["weekly_sales"].sum()
        diff = latest - prev
        pct = (diff / prev * 100) if prev else np.nan
        return diff, pct

    total = df.groupby(group_col)["weekly_sales"].sum().rename("total")
    yoy = df.groupby(group_col).apply(lambda g: pd.Series(_yoy(g), index=["yoy_diff", "yoy_pct"]))
    return pd.concat([total, yoy], axis=1).reset_index()

# 2) Stitching helper
@st.cache_data
def merge_for_rf_comparison(
    ts_actual: pd.Series,
    ts_pred:   pd.Series,
    warmup_weeks: int = 52
) -> pd.DataFrame:
    """
    Prepend the last `warmup_weeks` of actuals to the RF predictions,
    then compute total vs. total and differences.
    """
    pred_start = ts_pred.index.min()
    cutoff_actual = ts_actual[ts_actual.index < pred_start]
    warmup_idx    = cutoff_actual.iloc[-warmup_weeks:].index
    warmup_series = ts_actual.loc[warmup_idx]

    pred_full = pd.concat([
        warmup_series.rename("Pred_Warmup"),
        ts_pred.rename("Predicted")
    ])

    orig_actual    = ts_actual.reindex(pred_full.index).rename("Actual")
    total_act      = orig_actual.sum(skipna=True)
    total_pred     = pred_full.sum(skipna=True)
    abs_diff       = total_pred - total_act
    pct_diff       = total_act and (abs_diff / total_act * 100)

    df = pd.DataFrame({
        "date":        pred_full.index,
        "Actual":      orig_actual.values,
        "Pred_Warmup": pred_full.where(pred_full.index.isin(warmup_idx)).values,
        "Predicted":   pred_full.where(pred_full.index >= pred_start).values,
    }).reset_index(drop=True)

    df.attrs.update({
        "total_actual":         total_act,
        "total_predicted_full": total_pred,
        "abs_diff":             abs_diff,
        "pct_diff":             pct_diff or 0.0
    })
    return df