import streamlit as st
import base64 
from typing import Dict, List
import pandas as pd
import altair as alt
from typing import Literal
from matplotlib import cm, colors
from typing import Optional
import numpy as np


def plot_timeseries(
    df: pd.DataFrame,
    value_col: str,
    title: str = None,
    date_col: str = "date",
    use_container_width: bool = True
):
    """
    df: long-form or wide-form DataFrame  
    value_col: the column to sum and plot  
    date_col: the column to group on (defaults to 'date')  
    """
    ts = df.groupby(date_col)[value_col].sum().sort_index()
    if title:
        st.subheader(title)
    st.line_chart(ts, use_container_width=use_container_width)

def styled_block(
    *,
    width: str             = "100%",
    display: str           = "inline-block",
    text_align: str        = "center",
    margin: str            = "0 0 1rem 0",
    padding: str           = "",
    background: str        = "",
    border: str            = "",
    border_radius: str     = "",
    label: str,
    label_size: int        = 16,
    label_color: str       = "gray",
    label_weight: str      = "normal",
    label_margin: str      = "0 0 .25rem 0",
    value: str,
    value_size: int        = 28,
    value_color: str       = "black",
    value_weight: str      = "600",
    value_margin: str      = "0",
) -> None:
    html = f"""
    <div style="
        display:        {display};
        width:          {width};
        text-align:     {text_align};
        margin:         {margin};
        padding:        {padding};
        background:     {background};
        border:         {border};
        border-radius:  {border_radius};
    ">
      <div style="
          font-size:     {label_size}px;
          color:         {label_color};
          font-weight:   {label_weight};
          margin:        {label_margin};
      ">{label}</div>
      <div style="
          font-size:     {value_size}px;
          color:         {value_color};
          font-weight:   {value_weight};
          margin:        {value_margin};
      ">{value}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)



def human_formatter(
    x,
    *,
    is_money:   bool = False,
    is_percent: bool = False,
) -> str:
    """
    Compact formatting:
    - Money uses K/M/B with one decimal for K and two for M/B/T.
    - Percent ALWAYS shows two decimals.
    - Otherwise integer formatting.
    """
    try:
        val = float(x)
    except (TypeError, ValueError):
        return str(x)

    if is_percent:
        # TWO decimals so 99.96 stays 99.96, not 100.00
        sign = "-" if val < 0 else ""
        return f"{sign}{abs(val):.2f} %"

    # now do money/number formatting
    sign = "-" if val < 0 else ""
    abs_val = abs(val)
    if abs_val >= 1e12:
        out = f"{sign}{abs_val/1e12:.2f} T"
    elif abs_val >= 1e9:
        out = f"{sign}{abs_val/1e9:.2f} B"
    elif abs_val >= 1e6:
        out = f"{sign}{abs_val/1e6:.2f} M"
    elif abs_val >= 1e3:
        out = f"{sign}{abs_val/1e3:.1f} K"
    else:
        out = f"{val:,.0f}"

    if is_money:
        return f"{out} $"
    return out


def header(
    logo_path: str,
    title: str,
    logo_width: int = 200,
    logo_style: dict  = None,
    title_style: dict = None,
):
    """
    Renders a centered logo + title using styled_block.
    
    Args:
      logo_path:    Local path (SVG/PNG/etc) to your logo.
      title:        The page title text.
      logo_width:   Width in px for the logo <img>.
      logo_style:   Optional overrides for the logo block (see styled_block params).
      title_style:  Optional overrides for the title block.
    
    Any keys you pass in logo_style/title_style will override the defaults below.
    """
    # --- 1) Embed image as data URI ---
    raw = open(logo_path, "rb").read()
    b64 = base64.b64encode(raw).decode()
    mime = "image/svg+xml" if logo_path.lower().endswith(".svg") else "image/png"
    data_uri = f"data:{mime};base64,{b64}"

    # --- 2) Default style for the logo block ---
    default_logo = {
        "width": "100%",
        "display": "block",
        "text_align": "center",
        "margin": "0",
        "padding": "1rem 0 0 0",
        "background": "",
        "border": "",
        "border_radius": "",
        "label": f'<img src="{data_uri}" width="{logo_width}">',
        "label_size": 0,
        "label_color": "black",
        "label_weight": "normal",
        "label_margin": "0 0 .5rem 0",
        "value": "",
        "value_size": 0,
        "value_color": "black",
        "value_weight": "normal",
        "value_margin": "0",
    }
    # merge user overrides
    if logo_style:
        default_logo.update(logo_style)

    # --- 3) Default style for the title block ---
    default_title = {
        "width": "100%",
        "display": "block",
        "text_align": "center",
        "margin": "0",
        "padding": "0 0 1rem 0",
        "background": "",
        "border": "",
        "border_radius": "",
        "label": "",
        "label_size": 0,
        "label_color": "gray",
        "label_weight": "normal",
        "label_margin": "0",
        "value": title,
        "value_size": 32,
        "value_color": "black",
        "value_weight": "bold",
        "value_margin": "0",
    }
    if title_style:
        default_title.update(title_style)

    # --- 4) Render both blocks ---
    styled_block(**default_logo)
    styled_block(**default_title)

    st.markdown("---")


def show_metrics_generic(
    metrics: list[dict|tuple[str,str]],
    gap: str = "large"
) -> None:
    """
    Render a row of metrics.  Each item in `metrics` can be either:
      • A tuple (label, value) → rendered via st.metric  
      • A dict {"styled": True, "props": {...}} → props passed straight to styled_block
    """
    cols = st.columns(len(metrics), gap=gap)
    for spec, col in zip(metrics, cols):
        with col:
            if isinstance(spec, dict) and spec.get("styled", False):
                # use your styled_block for this one
                styled_block(**spec["props"])
            else:
                label, value = spec
                st.metric(label, value)

# New: sidebar controls grouping

def sidebar_controls(
    stores: list[str],
    depts:  list[str],
) -> dict[str, str]:
    store_choice = st.sidebar.selectbox("Select Store", stores, key="sel_store")
    dept_choice  = st.sidebar.selectbox("Select Department", depts, key="sel_dept")

    window_choice = st.sidebar.selectbox(
        "Limit to:",
        ["Full Range", "Last 52 Weeks", "Last 104 Weeks"],
        key="sel_window",
    )

    # ← ranking moved here
    ranking_choice = st.sidebar.selectbox(
        "Ranking criterion",
        ["Total Sales", "% Growth", "$ Growth"],
        key="sel_ranking",
    )

    return {
        "store":   store_choice,
        "dept":    dept_choice,
        "window":  window_choice,
        "ranking": ranking_choice,
    }

def _smooth_cmap(base: str, lo: float = 0.75, hi: float = 1.4) -> colors.ListedColormap:
    """
    Trim a Matplotlib colormap to the [lo, hi] fraction of its range,
    yielding richer, more saturated colors.
    """
    base_cmap = cm.get_cmap(base, 256)
    sliced    = base_cmap(np.linspace(lo, hi, 256))
    return colors.ListedColormap(sliced)


def show_ranking_grid(
    df_rank:    pd.DataFrame,
    metric_col: str,
    label_col:  str,
    top_n:      int              = 5,
    title:      Optional[str]    = None,
    green_cols: Optional[list[str]] = None,  # cols to style green
    red_cols:   Optional[list[str]] = None,  # cols to style red
) -> None:
    if title:
        st.subheader(title)

    # pick & sort
    best = (
        df_rank
          .nlargest(top_n, metric_col)
          .sort_values(metric_col, ascending=True)
          .reset_index(drop=True)
    )
    worst = (
        df_rank
          .nsmallest(top_n, metric_col)
          .sort_values(metric_col, ascending=False)
          .reset_index(drop=True)
    )

    # build colormaps
    green_cmap = _smooth_cmap("Greens")
    red_cmap   = _smooth_cmap("Reds")

    # global vmin/vmax for the main metric
    vals = df_rank[metric_col].astype(float)
    vmin, vmax = float(vals.min()), float(vals.max())

    # detect percent vs money
    key = metric_col.lower()
    percent_flag = (
        key.endswith("%") or
        "_pct" in key or
        key.startswith("pct") or
        key.endswith("percent")
    )
    money_flag = not percent_flag

    # formatter for the main metric
    fmt_main  = lambda v: human_formatter(v, is_money=money_flag, is_percent=percent_flag)
    # always format error_% as percent
    fmt_error = lambda v: human_formatter(v, is_percent=True)

    # default columns for coloring if none passed
    green_subset = green_cols or [metric_col]
    red_subset   = red_cols   or [metric_col]

    best_style = (
        best.style
            .format({
                metric_col: fmt_main,
                "error_%":  fmt_error
            })
            # apply green gradient to chosen columns
            .background_gradient(
                cmap=green_cmap, subset=green_subset, vmin=vmin, vmax=vmax
            )
    )
    worst_style = (
        worst.style
            .format({
                metric_col: fmt_main,
                "error_%":  fmt_error
            })
            # apply red gradient to chosen columns
            .background_gradient(
                cmap=red_cmap.reversed(), subset=red_subset, vmin=vmin, vmax=vmax
            )
    )

    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.markdown("#### 🟢 Best")
        st.dataframe(best_style, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("#### 🔴 Worst")
        st.dataframe(worst_style, use_container_width=True, hide_index=True)



def footer() -> None:
    st.markdown("---")
    st.caption("Jorge Rodrigues | Data Analyst @ Ironhack")

