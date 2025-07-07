import pandas as pd
from typing import Union, Tuple, Dict
import streamlit as st


def total_sales(df: pd.DataFrame) -> Union[int, float]:
    """
    Sum up the weekly_sales column.
    """
    return df["weekly_sales"].sum()


@st.cache_data
def load_rf_summary(path: str) -> tuple[float, float]:
    """
    Read your RandomForest summary CSV once and return (mae, r2).
    """
    df = pd.read_csv(path)
    return float(df.at[0, "mae"]), float(df.at[0, "r2"])

def span_stats(
    df: pd.DataFrame,
    date_col: str = "date",
    group_cols: list[str] = ["store","dept"]
) -> dict:
    """
    For the min and max of date_col, returns a dict:
      {
        'start_date': Timestamp,
        'end_date':   Timestamp,
        'counts': {
            'store': (start_count, end_count),
            'dept':  (start_count, end_count)
        }
      }
    """
    start, end = df[date_col].min(), df[date_col].max()
    out = {"start_date": start, "end_date": end, "counts": {}}
    for g in group_cols:
        c0 = df[df[date_col] == start][g].nunique()
        c1 = df[df[date_col] == end][g].nunique()
        out["counts"][g] = (c0, c1)
    return out

def week_span(df: pd.DataFrame, date_col: str = "date") -> int:
    return df[date_col].nunique()