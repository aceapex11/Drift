"""
Streamlit dashboard for the existing project.

The project filenames are:
- data_generator.py
- drift_engine.py
- run_experiment.py
- app.py

DO NOT import drift_engine_incremental.py.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_generator import generate, TRAIN_END, WINDOW
from drift_engine import DualRunner, StrategyRunner, STRATEGY_NAMES


st.set_page_config(
    page_title="Drift Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("◈ Drift Intelligence")
st.caption(
    "Predict → Detect → Diagnose → Adapt → Recover  |  "
    "KS / PSI data drift  •  Page-Hinkley relational drift  •  "
    "Incremental / Continual Learning"
)

COLORS = {
    "static": "#94A3B8",
    "incremental": "#34D399",
    "replay": "#60A5FA",
    "data": "#38BDF8",
    "relational": "#C084FC",
    "alarm": "#FB923C",
    "truth": "#FB7185",
    "adapt": "#2DD4BF",
}

st.markdown(
    """
    <style>
    .stApp {background-color:#0B1220;color:#E2E8F0}
    [data-testid="stSidebar"] {background-color:#111D30}
    [data-testid="stMetric"] {
        background:#17243A;
        border:1px solid #30445F;
        border-radius:12px;
        padding:12px
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Generating synthetic stream...")
def load_data():
    return generate()


def fig_base(title, ylabel):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#111D30",
        plot_bgcolor="#111D30",
        height=360,
        margin=dict(l=30, r=20, t=65, b=35),
        xaxis_title="Window",
        yaxis_title=ylabel,
        hovermode="x unified",
    )
    return fig


def add_truth_lines(fig, df, show_truth):
    if not show_truth:
        return fig

    if "true_drift_point" in df.columns:
        points = (
            pd.to_numeric(
                df["true_drift_point"],
                errors="coerce",
            )
            .dropna()
            .unique()
        )

        for point in points:
            fig.add_vline(
                x=int(point),
                line_dash="dash",
                line_color=COLORS["truth"],
                opacity=0.7,
            )

    return fig


with st.sidebar:
    st.header("Stream controls")

    delay = st.slider(
        "Seconds per window",
        0.0,
        2.0,
        0.0,
        0.1,
    )

    show_truth = st.checkbox(
        "Show simulated ground truth",
        False,
    )

    start = st.button(
        "▶ Start / restart",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "The live learner is SGDRegressor-based. "
        "No transfer-learning module is required."
    )


df = load_data()

if "live_history" not in st.session_state:
    st.session_state.live_history = []

if "live_events" not in st.session_state:
    st.session_state.live_events = []

if start:
    st.session_state.live_history = []
    st.session_state.live_events = []

tabs = st.tabs(
    [
        "Live monitor",
        "Drift analysis",
        "Model performance",
        "Adaptation & recovery",
        "Detector evaluation",
        "Event log",
    ]
)


# ------------------------------------------------------------------
# LIVE MONITOR
# ------------------------------------------------------------------
with tabs[0]:
    st.subheader("Live stream")

    cols = st.columns(5)
    prediction_slot = st.empty()
    live_chart_slot = st.empty()

    if start or not st.session_state.live_history:
        runner = DualRunner(
            df,
            TRAIN_END,
            WINDOW,
            seed=0,
        )
        # Run the stream progressively.
        for s, e in [
            (s, min(s + WINDOW, len(df)))
            for s in range(TRAIN_END, len(df), WINDOW)
        ]:
            row = runner.step(s, e)
            st.session_state.live_history.append(row)

            h = pd.DataFrame(st.session_state.live_history)

            cols[0].metric(
                "Window",
                int(row["window"]),
            )
            cols[1].metric(
                "MAE",
                f"{row['mae']:.4f}",
            )
            cols[2].metric(
                "RMSE",
                f"{row['rmse']:.4f}",
            )
            cols[3].metric(
                "Drift",
                row["diagnosis"],
            )
            cols[4].metric(
                "Adapted",
                "YES" if row["adapted"] else "NO",
            )

            prediction_slot.write(
                {
                    "Prediction MAE": row["mae"],
                    "Prediction RMSE": row["rmse"],
                    "R²": row["r2"],
                    "Diagnosis": row["diagnosis"],
                    "Alarm": row["alarm"],
                    "Data alarm": row["data_alarm"],
                    "Relational alarm": row["relational_alarm"],
                }
            )

            fig = fig_base(
                "Incremental SGD vs Static",
                "MAE",
            )

            fig.add_trace(
                go.Scatter(
                    x=h["window"],
                    y=h["mae_static"],
                    name="Static",
                    line=dict(
                        color=COLORS["static"],
                        width=2,
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=h["window"],
                    y=h["mae"],
                    name="Incremental SGD",
                    line=dict(
                        color=COLORS["incremental"],
                        width=3,
                    ),
                )
            )

            fig = add_truth_lines(
                fig,
                df,
                show_truth,
            )

            live_chart_slot.plotly_chart(
                fig,
                use_container_width=True,
            )

            if delay:
                time.sleep(delay)

        st.success("Stream completed.")

    elif st.session_state.live_history:
        h = pd.DataFrame(st.session_state.live_history)
        st.dataframe(h, use_container_width=True)


# ------------------------------------------------------------------
# DRIFT ANALYSIS
# ------------------------------------------------------------------
with tabs[1]:
    st.subheader("Drift signals")

    saved = Path("stream_log.csv")

    if saved.exists():
        log = pd.read_csv(saved)

        if not log.empty:
            if "worst_ks" in log.columns:
                fig = fig_base(
                    "Worst KS statistic",
                    "KS",
                )
                fig.add_trace(
                    go.Scatter(
                        x=log["window"],
                        y=log["worst_ks"],
                        name="Worst KS",
                    )
                )
                fig.add_hline(
                    y=0.15,
                    line_dash="dash",
                    line_color=COLORS["data"],
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            if "worst_psi" in log.columns:
                fig = fig_base(
                    "Worst PSI",
                    "PSI",
                )
                fig.add_trace(
                    go.Scatter(
                        x=log["window"],
                        y=log["worst_psi"],
                        name="Worst PSI",
                    )
                )
                fig.add_hline(
                    y=0.25,
                    line_dash="dash",
                    line_color=COLORS["data"],
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            if "ph_stat" in log.columns:
                fig = fig_base(
                    "Page-Hinkley statistic",
                    "PH statistic",
                )
                fig.add_trace(
                    go.Scatter(
                        x=log["window"],
                        y=log["ph_stat"],
                        name="Page-Hinkley",
                    )
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )
    else:
        st.info(
            "Run run_experiment.py first to generate stream_log.csv."
        )


# ------------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------------
with tabs[2]:
    st.subheader("Model comparison")

    comparison_path = Path("model_comparison.csv")

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        metric_options = [
            c for c in ["MAE", "RMSE", "R2"]
            if c in comparison.columns
        ]

        if metric_options and "Strategy" in comparison.columns:
            metric = st.selectbox(
                "Metric",
                metric_options,
            )

            fig = fig_base(
                f"{metric} by strategy",
                metric,
            )

            fig.add_trace(
                go.Bar(
                    x=comparison["Strategy"],
                    y=comparison[metric],
                    name=metric,
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )
    else:
        st.info(
            "Run run_experiment.py first to generate model_comparison.csv."
        )


# ------------------------------------------------------------------
# ADAPTATION
# ------------------------------------------------------------------
with tabs[3]:
    st.subheader("Adaptation and recovery")

    if Path("stream_log.csv").exists():
        log = pd.read_csv("stream_log.csv")

        if "adapted" in log.columns:
            st.metric(
                "Adapted windows",
                int(log["adapted"].sum()),
            )

        if "recovery" in log.columns:
            st.metric(
                "Recovery events",
                int(log["recovery"].sum()),
            )

        if "strategy" in log.columns:
            st.dataframe(
                log[
                    [
                        c for c in [
                            "window",
                            "strategy",
                            "diagnosis",
                            "alarm",
                            "adapted",
                            "recovery",
                            "mae",
                            "rmse",
                            "r2",
                        ]
                        if c in log.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Run run_experiment.py first.")


# ------------------------------------------------------------------
# DETECTOR EVALUATION
# ------------------------------------------------------------------
with tabs[4]:
    st.subheader("Detector evaluation")

    metrics_path = Path(
        "detector_metrics_summary.csv"
    )

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Detector metrics use simulated ground truth only for "
            "offline evaluation; ground truth is not used by the "
            "live detector."
        )
    else:
        st.info(
            "Run run_experiment.py first to generate detector metrics."
        )


# ------------------------------------------------------------------
# EVENT LOG
# ------------------------------------------------------------------
with tabs[5]:
    st.subheader("Detector events")

    event_path = Path("detector_events.csv")

    if event_path.exists():
        events = pd.read_csv(event_path)

        st.dataframe(
            events,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Run run_experiment.py first to generate detector_events.csv."
        )
