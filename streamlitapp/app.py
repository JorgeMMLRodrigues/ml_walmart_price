import streamlit as st
import pandas as pd

from metrics import (total_sales, 
                     load_rf_summary, 
                     span_stats, 
                     week_span,
                     build_ranking_table,
                     merge_for_rf_comparison
)

from data_loader import load_sales

from ui_components import (
     human_formatter,
     header,
     footer,
     show_metrics_generic,
     sidebar_controls,
     plot_timeseries,
     show_rf_metrics,
     plot_actual_vs_rf,
     show_ranking_grid,
 )
import altair as alt
from filters import filter_df, aggregate_timeseries, compute_rf_trends
import numpy as np


# ------------------------------------
# Page config
# ------------------------------------
st.set_page_config(page_title="Walmart Sales Dashboard", layout="wide")

# ------------------------------------
# Load data
# ------------------------------------
df = load_sales("df_wm_store_sales_predictions.csv")

y1 = int(df["date"].dt.year.min())
y2 = int(df["date"].dt.year.max())

# ------------------------------------
# Sidebar controls
# ------------------------------------
stores = ["All Stores"] + [str(s) for s in sorted(df["store"].unique())]
depts  = ["All Departments"] + [str(d) for d in sorted(df["dept"].unique())]
controls = sidebar_controls(stores, depts)
store_choice = controls["store"]
dept_choice  = controls["dept"]
limit_window = controls["window"]
metric_label = controls["ranking"]           # ← use sidebar choice
metric_map   = {
    "Total Sales":  "total",
    "YOY % Growth": "yoy_pct",
    "YOY $ Growth": "yoy_diff",
}
metric_col = metric_map[metric_label]
limit_window = controls["window"]
# ------------------------------------
# Data filtering
# ------------------------------------
df_sel = filter_df(df, store_choice, dept_choice, limit_window)

# ------------------------------------
# Choose full vs filtered for charts
# ------------------------------------
if store_choice == "All Stores" and dept_choice == "All Departments":
    df_graph = df.copy()
else:
    df_graph = df_sel

# ------------------------------------
# Compute metrics
# ------------------------------------

# Get start/end dates and counts in one call
stats = span_stats(df_sel)
stores_start, stores_end = stats["counts"]["store"]
depts_start, depts_end   = stats["counts"]["dept"]

span_start  = stats["start_date"].date()
span_end    = stats["end_date"].date()
total_weeks = week_span(df_sel, date_col="date", limit_window=limit_window)

# ------------------------------------
# Render UI
# ------------------------------------
header(
    "images/Walmart_logo.svg",
    title="🏪 Walmart Weekly Sales Dashboard",
    logo_width=150,
    logo_style={"padding": "2rem 0 1rem 0", "background": None},
    title_style={"value_size": 40, "value_color": "#333", "padding": "0 0 2rem 0"},
)

basic_metrics = [
    (f"🏪 Stores ({y1}→{y2})", f"{stores_start} → {stores_end}"),
    (f"🗂 Depts ({y1}→{y2})", f"{depts_start} → {depts_end}"),
    ("📆 Weeks Selected", str(total_weeks)),
    {
        "styled": True,
        "props": {
            "width": "100%", "display": "block", "text_align": "center",
            "margin": "0", "padding": "0",
            "label": "📅 Date Span", "label_size": 18, "label_color": "#888",
            "label_weight": "normal", "label_margin": "0 0 .25rem 0",
            "value": f"{span_start} → {span_end}",
            "value_size": 20, "value_color": "#000",
            "value_weight": "600", "value_margin": "0",
        }
    },
    ("💰 Total Sales", human_formatter(total_sales(df_sel)))
]

show_metrics_generic(basic_metrics)

ts_total = aggregate_timeseries(df_sel, cols={"Actual":"weekly_sales"})
# total sales
plot_timeseries(
    ts_total,
    value_col="Sales",
    date_col="date",
    title="📈 Total Sales Over Time"
)
# ──────────────────────────────────────────────────────────────
# Ranking grid: best / worst stores or depts
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🏆 Ranking")

# decide what to group by
if store_choice == "All Stores":
    if dept_choice == "All Departments":
        group_col, label_name = "store", "Store"
    else:
        group_col, label_name = "store", f"Store (Dept {dept_choice})"
else:
    if dept_choice == "All Departments":
        group_col, label_name = "dept", f"Dept in Store {store_choice}"
    else:
        group_col, label_name = None, f"Dept {dept_choice} in Store {store_choice}"

# build the ranking table if needed
if group_col:
    rank_tbl = build_ranking_table(df_sel, group_col)

# now render, re-using metric_label & metric_col from the sidebar
if group_col is None:
    # single slice → show one metric
    total_val = df_sel["weekly_sales"].sum()
    tmp       = df_sel.assign(__all__=0)
    yoy       = build_ranking_table(tmp, "__all__").iloc[0]

    val_map = {
        "total":    total_val,
        "yoy_pct":  yoy["yoy_pct"],
        "yoy_diff": yoy["yoy_diff"],
    }
    val = val_map[metric_col]

    st.metric(
        f"{label_name} – {metric_label}",
        human_formatter(val,
                        is_money=(metric_col!="yoy_pct"),
                        is_percent=(metric_col=="yoy_pct"))
    )
else:
    # full Best/Worst grid
    show_ranking_grid(
        df_rank   = rank_tbl[[group_col, metric_col]]
                         .rename(columns={group_col: label_name}),
        metric_col= metric_col,
        label_col = label_name,
        top_n     = 10,
        title     = f"Ranked by {metric_label}"
    )

# ──────────────────────────────────────────────────────────────
#  Trend lines + change metric for Best & Worst groups
# ──────────────────────────────────────────────────────────────
if group_col is not None:

    # Identify best & worst keys
    best_key  = rank_tbl.nlargest(1, metric_col)[group_col].iloc[0]
    worst_key = rank_tbl.nsmallest(1, metric_col)[group_col].iloc[0]

    # Slice and (for YOY) cap at 104 weeks
    df_best  = df_sel[df_sel[group_col] == best_key].copy()
    df_worst = df_sel[df_sel[group_col] == worst_key].copy()
    if metric_col in ("yoy_pct", "yoy_diff"):
        cutoff = df_sel["date"].max() - pd.Timedelta(weeks=104)
        df_best  = df_best [df_best ["date"] >= cutoff]
        df_worst = df_worst[df_worst["date"] >= cutoff]

    # Aggregate weekly_sales by date
    ts_best_raw  = df_best.groupby("date")["weekly_sales"].sum().sort_index().rename("value")
    ts_worst_raw = df_worst.groupby("date")["weekly_sales"].sum().sort_index().rename("value")

    # Transform based on metric
    if metric_col == "total":
        ts_best, ts_worst = ts_best_raw, ts_worst_raw
        y_axis_label     = "Total Weekly Sales"
    elif metric_col == "yoy_pct":
        ts_best  = ts_best_raw.pct_change(52) * 100
        ts_worst = ts_worst_raw.pct_change(52) * 100
        y_axis_label     = "YOY % Change"
    else:  # "yoy_diff"
        ts_best  = ts_best_raw.diff(52)
        ts_worst = ts_worst_raw.diff(52)
        y_axis_label     = "YOY $ Change"

    # Flags for formatting
    money_flag   = metric_col != "yoy_pct"
    percent_flag = metric_col == "yoy_pct"

    # Build combined DataFrame
    df_trend = pd.concat([
        ts_best .reset_index().assign(Series="Best"),
        ts_worst.reset_index().assign(Series="Worst"),
    ], ignore_index=True)

    col1, col2 = st.columns(2, gap="large")

    # — Best panel —
    with col1:
        st.subheader(f"📈 Trend for Best: {best_key}")

        # Show only the delta (latest value) as metric
        end_val   = ts_best.iloc[-1]
        if metric_col in ("yoy_pct", "yoy_diff"):
            delta_val = end_val
        else:
            delta_val = end_val - ts_best.iloc[0]

        st.metric(
            label=f"{metric_label} Change",
            value="",  # hide the primary value
            delta=human_formatter(delta_val, is_money=money_flag, is_percent=percent_flag),
        )

        # Draw the lines
        base_best = alt.Chart(df_trend[df_trend["Series"] == "Best"])
        main_line = base_best.mark_line(color="green").encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title=y_axis_label),
            tooltip=["date:T", "value:Q"],
        )
        trend_line = (
            base_best
              .transform_loess("date", "value", bandwidth=0.3)
              .mark_line(color="lightgrey", strokeDash=[4,4], size=1)
              .encode(x="date:T", y="value:Q")
        )
        st.altair_chart((main_line + trend_line).interactive(),
                        use_container_width=True)

    # — Worst panel —
    with col2:
        st.subheader(f"📉 Trend for Worst: {worst_key}")

        end_val   = ts_worst.iloc[-1]
        if metric_col in ("yoy_pct", "yoy_diff"):
            delta_val = end_val
        else:
            delta_val = end_val - ts_worst.iloc[0]

        st.metric(
            label=f"{metric_label} Change",
            value="",
            delta=human_formatter(delta_val, is_money=money_flag, is_percent=percent_flag),
        )

        base_worst = alt.Chart(df_trend[df_trend["Series"] == "Worst"])
        main_line  = base_worst.mark_line(color="red").encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title=y_axis_label),
            tooltip=["date:T", "value:Q"],
        )
        trend_line = (
            base_worst
              .transform_loess("date", "value", bandwidth=0.3)
              .mark_line(color="lightgrey", strokeDash=[4,4], size=1)
              .encode(x="date:T", y="value:Q")
        )
        st.altair_chart((main_line + trend_line).interactive(),
                        use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest comparisons
# ─────────────────────────────────────────────────────────────────────────────
st.title("Random Forest Sales Forecast Comparison")

# --- RF summary metrics
mae, r2 = load_rf_summary("RandomForest02_summary.csv")
show_metrics_generic([
    ("MAE", f"{mae:.2f}"),
    ("R²",  f"{r2:.3f}"),
])

# --- Compute trends on the filtered data
tr = compute_rf_trends(
    df_sel,
    date_col="date",
    actual_col="weekly_sales",
    rf_col="rf_02_predicted_weekly_sales",
    metric=metric_col
)

# --- Merge warm-up + out-of-sample predictions
df_overlay = merge_for_rf_comparison(
    ts_actual   = tr["ts_act"],
    ts_pred     = tr["ts_rf"],
    warmup_weeks=52
)

# --- Plot the stitched series
st.subheader("📈 Actual vs Forecast Overlay")
plot_actual_vs_rf(
    df          = df_overlay,
    date_col    = "date",
    actual_col  = "Actual",
    warmup_col  = "Pred_Warmup",
    pred_col    = "Predicted",
    title       = "Actual Sales vs Forecast"
)

# --- Compute totals apples→apples
# 1) Actual over your *entire* filtered dataset
actual_total = df_sel["weekly_sales"].sum(skipna=True)

# 2) Predicted-full = historical actuals up to forecast start (warm-up)
#    plus RF predictions thereafter
pred_full = df_overlay["Pred_Warmup"].fillna(0) + df_overlay["Predicted"].fillna(0)
pred_total = pred_full.sum(skipna=True)

# 3) Differences
abs_diff = pred_total - actual_total
pct_diff = (abs_diff / actual_total * 100) if actual_total else float("nan")

# --- Display with human_formatter
show_metrics_generic([
    ("Actual (full period)",    human_formatter(actual_total, is_money=True)),
    ("Predicted (full period)", human_formatter(pred_total,   is_money=True)),
    ("Absolute difference",     human_formatter(abs_diff,      is_money=True)),
    ("% difference",            human_formatter(pct_diff,      is_percent=True)),
])


# --- Footer
footer()


