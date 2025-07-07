import streamlit as st
import base64 
from typing import Dict
import pandas as pd
import altair as alt

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



def human_formatter(x, pos=None):
    """
    Format a number into K/M/B/T suffix form:
      1 000 →   1.0K
      1 000 000 → 1.0M
      1 000 000 000 → 1.0B
      1 000 000 000 000 → 1.0T
    """
    try:
        val = float(x)
    except (TypeError, ValueError):
        return str(x)

    abs_val = abs(val)
    sign = "-" if val < 0 else ""

    if abs_val >= 1e12:
        return f"{sign}{abs_val/1e12:.1f}T"
    elif abs_val >= 1e9:
        return f"{sign}{abs_val/1e9:.1f}B"
    elif abs_val >= 1e6:
        return f"{sign}{abs_val/1e6:.1f}M"
    elif abs_val >= 1e3:
        return f"{sign}{abs_val/1e3:.1f}K"
    else:
        return f"{int(val)}"


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
    stores: list[str], depts: list[str]
) -> Dict[str, str|bool]:
    return {
        "store": st.sidebar.selectbox("Select Store", stores),
        "dept": st.sidebar.selectbox("Select Department", depts),
        "window": st.sidebar.selectbox(
            "Limit to:", ["Full Range","Last 52 Weeks","Last 104 Weeks"]
        ),
        "rf_only": st.sidebar.checkbox(
            "RF prediction span only", value=False
        )
    }

def show_rf_metrics(mae: float, r2: float) -> None:
    """
    Display the MAE and R² side by side.
    """
    col1, col2, *_ = st.columns([1,1,6])
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("R²",  f"{r2:.3f}")

def plot_actual_vs_rf(
    df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    rf_col: str,
    title: str = "Actual vs RF Predictions",
    height: int = 400
) -> None:
    """
    Build and render an Altair line chart comparing two series.
    """
    df_chart = (
        df
        .groupby(date_col)
        .agg(
            Actual       = (actual_col, "sum"),
            RandomForest = (rf_col,    "sum")
        )
        .reset_index()
        .melt(
            id_vars=date_col,
            var_name="Series",
            value_name="Sales"
        )
    )
    chart = (
        alt.Chart(df_chart)
           .mark_line()
           .encode(
               x=alt.X(f"{date_col}:T", title="Date"),
               y=alt.Y("Sales:Q",    title="Total Weekly Sales"),
               color=alt.Color(
                   "Series:N",
                   scale=alt.Scale(
                       domain=["Actual","RandomForest"],
                       range=["steelblue","orange"]
                   ),
                   legend=alt.Legend(title=None)
               ),
               tooltip=[f"{date_col}:T","Series:N","Sales:Q"]
           )
           .properties(height=height)
           .interactive()
    )
    if title:
        st.subheader(title)
    st.altair_chart(chart, use_container_width=True)

def footer() -> None:
    st.markdown("---")
    st.caption("Jorge Rodrigues | Data Analyst @ Ironhack")
