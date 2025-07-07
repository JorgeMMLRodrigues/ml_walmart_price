import pandas as pd
from datetime import datetime
import streamlit as st

@st.cache_data
def load_sales(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df
