import streamlit as st
import pandas as pd
from metrics import total_sales, load_rf_summary, span_stats, week_span
from data_loader import load_sales
from ui_components import (
     human_formatter,
     header,
     footer,
     show_metrics_generic,
     sidebar_controls,
     plot_timeseries,
     show_rf_metrics,
     plot_actual_vs_rf
 )
import altair as alt
from filters import filter_df, aggregate_timeseries


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
rf_only       = controls["rf_only"]

# ------------------------------------
# Data filtering
# ------------------------------------
df_sel = filter_df(df, store_choice, dept_choice, rf_only, limit_window)

# ------------------------------------
# Compute metrics
# ------------------------------------

# Get start/end dates and counts in one call
stats = span_stats(df_sel)
stores_start, stores_end = stats["counts"]["store"]
depts_start, depts_end   = stats["counts"]["dept"]

span_start  = stats["start_date"].date()
span_end    = stats["end_date"].date()
total_weeks = week_span(df_sel)

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

# ------------------------------------
# Actual vs Random Forest Predictions
# ------------------------------------
mae, r2 = load_rf_summary("RandomForest02_summary.csv")
show_rf_metrics(mae, r2)

# pass df_sel (wide form) + column names
plot_actual_vs_rf(
    df_sel,
    date_col="date",
    actual_col="weekly_sales",
    rf_col="rf_02_predicted_weekly_sales",
    title="📊 Actual vs RF Predictions"
)

# ------------------------------------
# Last 104 Weeks Sales
# ------------------------------------
df_104 = df_sel[df_sel["date"] >= (df_sel["date"].max() - pd.Timedelta(weeks=104))]
ts_104 = aggregate_timeseries(df_104, cols={"Sales (104w)":"weekly_sales"})
plot_timeseries(
    ts_total,
    value_col="Sales",
    date_col="date",
    title="📊 Last 104 Weeks Sales"
)
# ------------------------------------
# Footer
# ------------------------------------
footer()
