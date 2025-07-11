import streamlit as st
import pandas as pd

from metrics import (total_sales, 
                     load_rf_summary, 
                     span_stats, 
                     week_span,
                     build_ranking_table
)

from data_loader import load_sales

from ui_components import (
     human_formatter,
     header,
     footer,
     show_metrics_generic,
     sidebar_controls,
     plot_timeseries,
     show_ranking_grid,
 )
import altair as alt
from filters import filter_df, aggregate_timeseries
import json


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
# 1) Build the store list and render its selectbox
stores = ["All Stores"] + [str(s) for s in sorted(df["store"].unique())]
store_choice = st.sidebar.selectbox("Select Store", stores, key="sel_store")

# 2) Based on that choice, build the dept list
if store_choice == "All Stores":
    # show every department
    dept_values = sorted(df["dept"].unique())
else:
    # compare store_choice (a string) to the string-cast of df["store"]
    mask = df["store"].astype(str) == store_choice
    dept_values = sorted(df.loc[mask, "dept"].unique())

dept_options = ["All Departments"] + [str(d) for d in dept_values]

# 3) Render the dept selectbox with the filtered options
dept_choice = st.sidebar.selectbox("Select Department", dept_options, key="sel_dept")

# 4) The rest is unchanged
window_choice = st.sidebar.selectbox(
    "Limit to:",
    ["Full Range", "Last 52 Weeks", "Last 104 Weeks"],
    key="sel_window",
)
ranking_choice = st.sidebar.selectbox(
    "Ranking criterion",
    ["Total Sales", "% Growth", "$ Growth"],
    key="sel_ranking",
)

# 5) Pack into your controls dict (so downstream code stays the same)
controls = {
    "store":   store_choice,
    "dept":    dept_choice,
    "window":  window_choice,
    "ranking": ranking_choice,
}
store_choice  = controls["store"]
dept_choice   = controls["dept"]
window_choice = controls["window"]
metric_label  = controls["ranking"]
metric_map    = {
    "Total Sales": "total",
    "% Growth":    "yoy_pct",
    "$ Growth":    "yoy_diff",
}
metric_col    = metric_map[metric_label]
limit_window  = controls["window"]
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

st.set_page_config(page_title="Walmart Dashboard", page_icon="🛒")

# show the logo + title at the top
header(
    logo_path="images/walmart_logo.svg",
    title="🏪 Walmart Database",
    logo_width=300,  # was 150
    logo_style={"padding": "2rem 0 1rem 0", "background": None},
    title_style={"value_size": 40, "value_color": "#333", "padding": "0 0 2rem 0"},
)



# ------------------------------------
# Get start/end dates and counts in one call
# ------------------------------------
stats = span_stats(df_sel)
stores_start, stores_end = stats["counts"]["store"]
# remove depts_start/end here
span_start  = stats["start_date"].date()
span_end    = stats["end_date"].date()
total_weeks = week_span(df_sel, date_col="date", limit_window=limit_window)

# ------------------------------------
# Render UI
# ------------------------------------
basic_metrics = []

# Stores: total vs specific (exactly your original logic)
if store_choice == "All Stores":
    basic_metrics.append(("🏪 Stores", str(stores_end)))
else:
    basic_metrics.append(("🏪 Store", store_choice))

# Depts: now taking store_choice into account
if dept_choice == "All Departments":
    if store_choice == "All Stores":
        # case: All Stores + All Depts → sum per‐store unique counts
        total_depts = (
            df_sel
            .groupby("store")["dept"]
            .nunique()
            .sum()
        )
    else:
        # case: single store + All Depts → just count unique in this store
        total_depts = df_sel["dept"].nunique()

    basic_metrics.append(("🗂 Depts", str(total_depts)))
else:
    # any store_selection + specific Dept → show the dept name
    basic_metrics.append(("📂 Dept", dept_choice))
# Weeks & date span
basic_metrics.extend([
    ("📆 Weeks Selected", str(total_weeks)),
    {
        "styled": True,
        "props": {
            "width": "100%", "display": "block", "text_align": "center",
            "margin": "0", "padding": "0",
            "label": "📅 Date Span", "label_size": 18, "label_color": "",
            "label_weight": "normal", "label_margin": "0 0 .25rem 0",
            "value": f"{span_start} → {span_end}",
            "value_size": 20, "value_color": "#000",
            "value_weight": "600", "value_margin": "0",
        }
    },
    ("💰 Total Sales", human_formatter(total_sales(df_sel), is_money=True))
])

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

# 1) Decide grouping
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

# 2) Single‐slice (no grouping): show one metric as before
if group_col is None:
    # build & stitch
    df_fc = (
        df_sel
        .groupby("date")
        .agg(
            Actual = ("weekly_sales", "sum")
        )
        .reset_index()
    )

    # compute the three
    act_tot   = df_fc["Actual"].sum()
    first_act = df_fc["Actual"].iloc[0]
    last_act  = df_fc["Actual"].iloc[-1]
    diff_act  = last_act - first_act
    pct_act   = (diff_act / first_act * 100) if first_act else float("nan")

    if metric_col == "total":
        disp = act_tot; m_flag, p_flag = True, False
    elif metric_col == "yoy_diff":
        disp = diff_act; m_flag, p_flag = True, False
    else:  # "yoy_pct"
        disp = pct_act; m_flag, p_flag = False, True

    st.metric(
        f"{label_name} – {metric_label}",
        human_formatter(disp, is_money=m_flag, is_percent=p_flag)
    )

# 3) Grouping: build a small ranking DataFrame & use show_ranking_grid
else:
    records = []
    for key, sub in df_sel.groupby(group_col):
        # build & stitch each subgroup
        df_fc_sub = (
            sub
            .groupby("date")
            .agg(
                Actual        = ("weekly_sales",                 "sum"),
            )
            .reset_index()
        )

        # compute metrics
        act_tot   = df_fc_sub["Actual"].sum()
        first_act = df_fc_sub["Actual"].iloc[0]
        last_act  = df_fc_sub["Actual"].iloc[-1]
        diff_act  = last_act - first_act
        pct_act   = (diff_act / first_act * 100) if first_act else float("nan")

        # pick the one we’re ranking on
        if metric_col == "total":
            val = act_tot
        elif metric_col == "yoy_diff":
            val = diff_act
        else:  # "yoy_pct"
            val = pct_act

        records.append({ label_name: key, metric_col: val })

    df_rank = pd.DataFrame.from_records(records)
        # 1) build your rename map
    rename_map = { group_col: label_name }
    if metric_col == "yoy_pct":
        rename_map["yoy_pct"]  = "% Growth"
        display_col = "% Growth"
    elif metric_col == "yoy_diff":
        rename_map["yoy_diff"] = "$ Growth"
        display_col = "$ Growth"
    else:
        display_col = metric_col  # e.g. "total"

    # 2) make a display copy with just the headers swapped
    df_rank_display = df_rank.rename(columns=rename_map)

    # 3) pass that into your grid, pointing at the new display name
    show_ranking_grid(
        df_rank    = df_rank_display,
        metric_col = display_col,
        label_col  = label_name,
        top_n      = 5,
        title      = f"Ranked by {metric_label}"
    )



# ────────────────────────────────────────────────────────
# Trend lines + change metric for Best & Worst groups
# ────────────────────────────────────────────────────────
if group_col is not None:
    best_key  = df_rank.nlargest(1, metric_col)[label_name].iloc[0]
    worst_key = df_rank.nsmallest(1, metric_col)[label_name].iloc[0]

    df_best  = df_sel[df_sel[group_col] == best_key].copy()
    df_worst = df_sel[df_sel[group_col] == worst_key].copy()

    # compute the half-span yw just like the grid
    if limit_window == "Last 52 Weeks":
        yw = 26
    elif limit_window == "Last 104 Weeks":
        yw = 52
    elif limit_window == "Full Range":
        total_days  = (df_sel["date"].max() - df_sel["date"].min()).days
        total_weeks = total_days // 7
        yw          = total_weeks // 2
    else:
        yw = 52

    # define flags now, so they exist no matter what
    money_flag   = (metric_col != "yoy_pct")
    percent_flag = (metric_col == "yoy_pct")

    # aggregate
    ts_best_raw  = df_best .groupby("date")["weekly_sales"].sum().sort_index().rename("value")
    ts_worst_raw = df_worst.groupby("date")["weekly_sales"].sum().sort_index().rename("value")

    # if we don’t have enough points for a yw-week diff, fall back to totals
    n_best  = len(ts_best_raw)
    n_worst = len(ts_worst_raw)
    if metric_col == "total" or n_best <= yw or n_worst <= yw:
        ts_best, ts_worst = ts_best_raw, ts_worst_raw
        y_axis_label      = "Total Weekly Sales"
    else:
        if metric_col == "yoy_pct":
            ts_best  = ts_best_raw .pct_change(yw) * 100
            ts_worst = ts_worst_raw.pct_change(yw) * 100
            y_axis_label = f"YOY % Change ({yw}w)"
        else:
            ts_best  = ts_best_raw .diff(yw)
            ts_worst = ts_worst_raw.diff(yw)
            y_axis_label = f"YOY $ Change ({yw}w)"

    # build and plot
    df_trend = pd.concat([
        ts_best .reset_index().assign(Series="Best"),
        ts_worst.reset_index().assign(Series="Worst"),
    ], ignore_index=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader(f"📈 Trend for Best: {best_key}")
        end_val   = ts_best.iloc[-1]
        delta_val = end_val if metric_col in ("yoy_pct","yoy_diff") and n_best>yw else (end_val - ts_best.iloc[0])
        st.metric(label=f"{metric_label} Change",
                  value="",
                  delta=human_formatter(delta_val, is_money=money_flag, is_percent=percent_flag))
        base = alt.Chart(df_trend[df_trend["Series"]=="Best"])
        main = base.mark_line(color="green") \
                   .encode(x="date:T", y=alt.Y("value:Q", title=y_axis_label), tooltip=["date:T","value:Q"])
        loess = base.transform_loess("date","value",bandwidth=0.3) \
                    .mark_line(color="lightgrey", strokeDash=[4,4], size=1) \
                    .encode(x="date:T", y="value:Q")
        st.altair_chart((main+loess).interactive(), use_container_width=True)

    with col2:
        st.subheader(f"📉 Trend for Worst: {worst_key}")
        end_val   = ts_worst.iloc[-1]
        delta_val = end_val if metric_col in ("yoy_pct","yoy_diff") and n_worst>yw else (end_val - ts_worst.iloc[0])
        st.metric(label=f"{metric_label} Change",
                  value="",
                  delta=human_formatter(delta_val, is_money=money_flag, is_percent=percent_flag))
        base = alt.Chart(df_trend[df_trend["Series"]=="Worst"])
        main = base.mark_line(color="red") \
                   .encode(x="date:T", y=alt.Y("value:Q", title=y_axis_label), tooltip=["date:T","value:Q"])
        loess = base.transform_loess("date","value",bandwidth=0.3) \
                    .mark_line(color="lightgrey", strokeDash=[4,4], size=1) \
                    .encode(x="date:T", y="value:Q")
        st.altair_chart((main+loess).interactive(), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚙️ Random Forest Sales Forecast Comparison")

# --- RF summary metrics
mae, r2 = load_rf_summary("RandomForest02_summary.csv")
show_metrics_generic([
    ("MAE", f"{mae:.2f}"),
    ("R²",  f"{r2:.3f}")
])

# --- df_sel is already filtered by your sidebar (store/dept/window)

# ─────────────────────────────────────────────────────────────────────────────
# Build the stitched forecast DataFrame from df_sel
# ─────────────────────────────────────────────────────────────────────────────
df_fc = (
    df_sel
      .groupby("date")
      .agg(
          Actual        = ("weekly_sales",                 "sum"),
          Predicted_raw = ("rf_02_predicted_weekly_sales", "sum")
      )
      .reset_index()
)

pred_start = df_fc.loc[df_fc["Predicted_raw"] > 0, "date"].min()

df_fc["Forecast"] = df_fc["Actual"]
df_fc.loc[df_fc["date"] >= pred_start, "Forecast"] = df_fc.loc[df_fc["date"] >= pred_start, "Predicted_raw"]

# ─────────────────────────────────────────────────────────────────────────────
# **KEY**: sort by date so iloc picks the true endpoints of the selected span
# ─────────────────────────────────────────────────────────────────────────────
df_fc = df_fc.sort_values("date").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plot Actual vs Stitched Forecast
# ─────────────────────────────────────────────────────────────────────────────

# Mask Forecast always
df_fc["Forecast_vis"] = df_fc["Forecast"].where(df_fc["date"] >= pred_start, pd.NA)

# Mask Actual only when in the 52-week view
if window_choice == "Last 52 Weeks":
    df_fc["Actual_vis"] = df_fc["Actual"].where(df_fc["date"] >= pred_start, pd.NA)
else:
    df_fc["Actual_vis"] = df_fc["Actual"]

df_chart = df_fc.melt(
    id_vars   = "date",
    value_vars= ["Actual_vis","Forecast_vis"],
    var_name  = "Series",
    value_name= "Sales"
)
df_chart["Series"] = df_chart["Series"].replace({
    "Actual_vis":   "Actual",
    "Forecast_vis": "Forecast"
})

color_scale = alt.Scale(domain=["Actual","Forecast"], range=["steelblue","orange"])
chart = (
    alt.Chart(df_chart)
       .mark_line(strokeWidth=2)
       .encode(
           x=alt.X("date:T", title="Date"),
           y=alt.Y("Sales:Q", title="Weekly Sales"),
           color=alt.Color("Series:N", scale=color_scale, legend=None),
           tooltip=["date:T","Series:N","Sales:Q"]
       )
       .properties(height=400)
       .interactive()
)
st.subheader("📈 Actual vs Stitched Forecast")
st.altair_chart(chart, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Compute all metrics from df_fc
# ─────────────────────────────────────────────────────────────────────────────
# Totals
actual_total = df_fc["Actual"].sum()
pred_total   = df_fc["Forecast"].sum()

# Actual start‐to‐end growth
first_act = df_fc["Actual"].iloc[0]
last_act  = df_fc["Actual"].iloc[-1]
delta_act = last_act - first_act
pct_act   = (delta_act / first_act * 100) if first_act else float("nan")

# Forecast start‐to‐end growth
first_pred = df_fc["Forecast"].iloc[0]
last_pred  = df_fc["Forecast"].iloc[-1]
delta_pred = last_pred - first_pred
pct_pred   = (delta_pred / first_pred * 100) if first_pred else float("nan")

# Differences between Forecast and Actual metrics
diff_total = pred_total  - actual_total
diff_delta = delta_pred  - delta_act
diff_pct   = pct_pred    - pct_act

# ─────────────────────────────────────────────────────────────────────────────
# Display metrics in two rows
# ─────────────────────────────────────────────────────────────────────────────
col_a1, col_a2, col_a3 = st.columns(3, gap="large")
with col_a1:
    st.metric("Actual",           human_formatter(actual_total, is_money=True))
with col_a2:
    st.metric("$ Growth Actual",  human_formatter(delta_act,      is_money=True))
with col_a3:
    st.metric("% Growth Actual",  human_formatter(pct_act,        is_percent=True))

col_f1, col_f2, col_f3 = st.columns(3, gap="large")
with col_f1:
    st.metric(
        "Forecast",
        human_formatter(pred_total, is_money=True),
        delta=human_formatter(diff_total, is_money=True)
    )
with col_f2:
    st.metric(
        "$ Growth Forecast",
        human_formatter(delta_pred,   is_money=True),
        delta=human_formatter(diff_delta, is_money=True)
    )
with col_f3:
    st.metric(
        "% Growth Forecast",
        human_formatter(pct_pred,     is_percent=True),
        delta=human_formatter(diff_pct,   is_percent=True)
    )

# ──────────────────────────────────────────────────────────────
# Ranking grid: best & worst forecast accuracy by store or dept
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🏆 Sales Forecast Accuracy Ranking")

# 1) decide grouping...
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

# 2) only makes sense when grouping
if group_col is None:
    st.info("Select a store *or* a department slice in the sidebar to see forecast accuracy rankings.")
else:
    records = []
    for key, sub in df_sel.groupby(group_col):
        # build & stitch Actual vs Forecast
        df_fc_sub = (
            sub.groupby("date")
               .agg(
                   Actual        = ("weekly_sales",                 "sum"),
                   Predicted_raw = ("rf_02_predicted_weekly_sales", "sum")
               )
               .reset_index()
        )
        pr_start = df_fc_sub.loc[df_fc_sub["Predicted_raw"] > 0, "date"].min()
        df_fc_sub["Forecast"] = df_fc_sub["Actual"]
        df_fc_sub.loc[df_fc_sub["date"] >= pr_start, "Forecast"] = (
            df_fc_sub.loc[df_fc_sub["date"] >= pr_start, "Predicted_raw"]
        )

        # compute totals & percent‐error
        actual_total   = df_fc_sub["Actual"].sum()
        forecast_total = df_fc_sub["Forecast"].sum()
        diff_frac      = (forecast_total - actual_total) / actual_total if actual_total else float("nan")
        err_pct        = abs(diff_frac) * 100  # now true percent
        acc_pct        = max(0, min(100, 100 - err_pct))

        records.append({
            label_name:   key,
            "Accuracy %": acc_pct,
            "Error %":    err_pct,
        })

    df_rank_acc = pd.DataFrame.from_records(records)

    show_ranking_grid(
        df_rank    = df_rank_acc,
        metric_col = "Accuracy %",
        label_col  = label_name,
        top_n      = 5,
        title      = "Top & Bottom 5 by Forecast Accuracy",
        green_cols = ["Error %"],
        red_cols   = ["Error %"]
    )



# ─────────────────────────────────────────────────────────────────────────────
# Model Feature Importance Ranking
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🏅 Random Forest Feature Importances")


# load the summary CSV (you’ve already got mae/r2 from it, now grab the rest)
df_summary = pd.read_csv("RandomForest02_summary.csv")

# parse the JSON dict in the ‘feat_importances’ column
feat_imp_dict = json.loads(df_summary.loc[0, "feat_importances"])

# turn into a DataFrame, sort by importance descending
df_feat_imp = (
    pd.Series(feat_imp_dict, name="importance")
      .reset_index()
      .rename(columns={"index": "feature"})
      .sort_values("importance", ascending=False)
      .reset_index(drop=True)
)

# take only the top N features
top_n = 6
df_top = df_feat_imp.head(top_n)

# compute percent of total importance
total_imp = df_feat_imp["importance"].sum()
df_top["pct_of_total"] = df_top["importance"] / total_imp * 100

# format just the percent
df_top["pct_of_total_fmt"] = df_top["pct_of_total"].apply(lambda v: human_formatter(v, is_percent=True))

# display only feature + % of total
st.dataframe(
    df_top[["feature", "pct_of_total_fmt"]]
         .rename(columns={
             "feature":         "Feature",
             "pct_of_total_fmt":"% of Total Importance"
         }),
    use_container_width=True
)







footer()

