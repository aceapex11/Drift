"""
Streamlit frontend for the existing live drift-detection project.

Run:
    streamlit run app.py

IMPORTANT:
- data_generator.py is unchanged
- drift_engine.py provides SGD incremental/continual learning.
- There is no TransferGBR or transfer-learning dependency.
- The live dashboard uses StrategyRunner.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_generator import generate, TRAIN_END, WINDOW
from drift_engine import StrategyRunner


# ============================================================
# PAGE / THEME
# ============================================================

st.set_page_config(
    page_title="Live Drift Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg": "#F5F7FB",
    "panel": "#FFFFFF",
    "grid": "#D9E1EC",
    "text": "#172033",
    "muted": "#64748B",
    "static": "#64748B",
    "incremental": "#16A34A",
    "data": "#0284C7",
    "relational": "#7C3AED",
    "alarm": "#D97706",
    "truth": "#E11D48",
    "adapt": "#0F766E",
    "danger": "#DC2626",
}

st.markdown(
    f"""
    <style>
        .stApp {{
            background: {COLORS["bg"]};
            color: {COLORS["text"]};
        }}

        [data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid #D9E1EC;
        }}

        [data-testid="stSidebar"] * {{
            color: {COLORS["text"]};
        }}

        [data-testid="stHeader"] {{
            background: #FFFFFF;
        }}

        [data-testid="stToolbar"] {{
            background: #FFFFFF;
        }}

        [data-testid="stMetric"] {{
            background: {COLORS["panel"]};
            border: 1px solid #D9E1EC;
            border-radius: 14px;
            padding: 14px 16px;
        }}

        [data-testid="stMetricLabel"] {{
            color: {COLORS["muted"]};
        }}

        [data-testid="stMetricValue"] {{
            color: {COLORS["text"]};
        }}

        .section-title {{
            font-size: 1.05rem;
            font-weight: 700;
