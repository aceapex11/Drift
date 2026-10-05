
"""
Streamlit frontend for the existing drift-detection project.

IMPORTANT:
- Keep this file named: app.py
- Keep the backend file named: drift_engine.py
- data_generator.py is unchanged
- No drift_engine_incremental.py is required.

Run:
    streamlit run app.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
            color: {COLORS["text"]};
            margin: 0.6rem 0 0.25rem 0;
        }}

        .small-note {{
            color: {COLORS["muted"]};
            font-size: 0.85rem;
        }}

        div[data-testid="stTabs"] button {{
            font-weight: 650;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("◈ Live Drift Intelligence")
st.caption(
    "Predict → Detect → Diagnose → Adapt → Recover  |  "
    "KS / PSI data drift  •  Page-Hinkley relational drift  •  "
    "Incremental / Continual Learning"
)


# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner="Generating synthetic stream…")
def load_data():
    return generate()


# Data is loaded lazily only when the user starts the live stream.
# This keeps the dashboard visible immediately after deployment.
df = None


# ============================================================
# HELPERS
# ============================================================

def make_fig(title, y_title=None, height=370):
    fig = go.Figure()

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        template="plotly_white",
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"]),
        height=height,
        margin=dict(l=45, r=35, t=62, b=42),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(
            title="Stream window",
            gridcolor=COLORS["grid"],
            zeroline=False,
        ),
        yaxis=dict(
            title=y_title or "",
            gridcolor=COLORS["grid"],
            zeroline=False,
        ),
    )

    return fig


def read_csv(name):
    path = Path(__file__).resolve().parent / name

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def add_event_lines(fig, events, show_truth):
    seen = set()

    for event in events:
        kind = event["type"]

        if kind == "truth" and not show_truth:
            continue

        color = {
            "truth": COLORS["truth"],
            "alarm": COLORS["alarm"],
            "adapt": COLORS["adapt"],
        }[kind]

        dash = {
            "truth": "dash",
            "alarm": "solid",
            "adapt": "dot",
        }[kind]

        label = {
            "truth": "True drift",
            "alarm": "Drift detected",
            "adapt": "Adaptation",
        }[kind]

        if kind not in seen:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=label,
                    line=dict(
                        color=color,
                        dash=dash,
                        width=2,
                    ),
                )
            )
            seen.add(kind)

        fig.add_vline(
            x=event["window"],
            line_color=color,
            line_dash=dash,
            line_width=1.5,
            opacity=0.75,
        )

    return fig


def build_truth_events(data):
    events = []

    if "true_drift_point" not in data.columns:
        return events

    points = (
        pd.to_numeric(
            data["true_drift_point"],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    for point in points:
        point = int(point)

        # Convert row position into stream-window position.
        if point >= TRAIN_END:
            window_number = (point - TRAIN_END) // WINDOW
            events.append(
                {
                    "window": int(window_number),
                    "type": "truth",
                    "row": point,
                }
            )

    return events


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Stream controls")

    delay = st.slider(
        "Seconds per window",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
    )

    show_truth = st.checkbox(
        "Show simulated ground truth",
        value=False,
        help=(
            "Ground truth is used only for offline evaluation and "
            "visualization. It is never used by the live detector."
        ),
    )

    start = st.button(
        "▶ Start / restart live stream",
        type="primary",
        width="stretch",
    )

    st.divider()

    st.markdown("### Pipeline")

    st.markdown(
        """
        **1. Predict**  
        SGDRegressor

        **2. Detect**  
        KS / PSI + Page-Hinkley

        **3. Diagnose**  
        DATA / RELATIONAL / BOTH

        **4. Adapt**  
        Incremental learning

        **5. Recover**  
        Monitor post-drift error
        """
    )

    st.divider()

    st.caption(
        "Static and incremental SGD are compared live. "
        "Full retraining and replay strategies are evaluated by "
        "`run_experiment.py`."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "events" not in st.session_state:
    st.session_state.events = []

if "live_runner" not in st.session_state:
    st.session_state.live_runner = None

if "stream_df" not in st.session_state:
    st.session_state.stream_df = None

if "stream_index" not in st.session_state:
    st.session_state.stream_index = 0

if start:
    st.session_state.history = []
    st.session_state.events = []


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "◉ Live Monitor",
        "◈ Drift Analysis",
        "▤ Model Performance",
        "↗ Adaptation & Recovery",
        "◎ Detector Evaluation",
        "☷ Event Log",
    ]
)


# ============================================================
# LIVE MONITOR
# ============================================================

with tabs[0]:

    metric_cols = st.columns(6)
    metric_slots = [c.empty() for c in metric_cols]

    st.markdown(
        '<div class="section-title">Live model behaviour</div>',
        unsafe_allow_html=True,
    )

    live_chart_slot = st.empty()

    if not st.session_state.history:
        st.info(
            "Click **Start / restart live stream** to run the "
            "incremental-learning stream. The dashboard loads immediately; "
            "the synthetic data is generated only when you start the stream."
        )

    elif st.session_state.history:

        history = pd.DataFrame(
            st.session_state.history
        )

        last = history.iloc[-1]

        metric_slots[0].metric(
            "CURRENT WINDOW",
            int(last["window"]) + 1,
        )

        metric_slots[1].metric(
            "DIAGNOSIS",
            str(last["diagnosis"]).upper(),
        )

        metric_slots[2].metric(
            "STATIC MAE",
            f"{last['mae_static']:.3f}",
        )

        metric_slots[3].metric(
            "INCREMENTAL MAE",
            f"{last['mae']:.3f}",
            f"{last['mae'] - last['mae_static']:+.3f} vs static",
            delta_color="inverse",
        )

        metric_slots[4].metric(
            "RMSE",
            f"{last['rmse']:.3f}",
        )

        metric_slots[5].metric(
            "R²",
            f"{last['r2']:.3f}",
        )

        fig = make_fig(
            "Model error · lower is better",
            "MAE",
            400,
        )

        fig.add_trace(
            go.Scatter(
                x=history["window"],
                y=history["mae_static"],
                mode="lines+markers",
                name="Static — no adaptation",
                line=dict(
                    color=COLORS["static"],
                    width=2.5,
                ),
                marker=dict(size=4),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=history["window"],
                y=history["mae"],
                mode="lines+markers",
                name="Incremental SGD",
                line=dict(
                    color=COLORS["incremental"],
                    width=3,
                ),
                marker=dict(size=4),
            )
        )

        fig = add_event_lines(
            fig,
            st.session_state.events,
            show_truth,
        )

        live_chart_slot.plotly_chart(
            fig,
            width="stretch",
        )


# ============================================================
# DRIFT ANALYSIS
# ============================================================

with tabs[1]:

    history = pd.DataFrame(
        st.session_state.history
    )

    if history.empty:
        saved = read_csv("stream_log.csv")
        if saved is not None:
            history = saved

    if history.empty:
        st.info(
            "Run the live stream or `run_experiment.py` first."
        )

    else:

        left, right = st.columns(2)

        with left:
            if "worst_ks" in history.columns:

                fig = make_fig(
                    "Worst KS statistic",
                    "KS statistic",
                )

                fig.add_trace(
                    go.Scatter(
                        x=history["window"],
                        y=history["worst_ks"],
                        name="Worst KS",
                        line=dict(
                            color=COLORS["data"],
                            width=2.5,
                        ),
                    )
                )

                fig.add_hline(
                    y=0.15,
                    line_dash="dash",
                    line_color=COLORS["alarm"],
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

        with right:
            if "worst_psi" in history.columns:

                fig = make_fig(
                    "Worst PSI",
                    "PSI",
                )

                fig.add_trace(
                    go.Scatter(
                        x=history["window"],
                        y=history["worst_psi"],
                        name="Worst PSI",
                        line=dict(
                            color=COLORS["relational"],
                            width=2.5,
                        ),
                    )
                )

                fig.add_hline(
                    y=0.25,
                    line_dash="dash",
                    line_color=COLORS["alarm"],
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

        if "ph_stat" in history.columns:

            fig = make_fig(
                "Page-Hinkley relational-drift signal",
                "PH statistic",
            )

            fig.add_trace(
                go.Scatter(
                    x=history["window"],
                    y=history["ph_stat"],
                    name="Page-Hinkley",
                    line=dict(
                        color=COLORS["relational"],
                        width=2.5,
                    ),
                )
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

        if "ks" in history.columns:
            latest = history.iloc[-1]

            ks_values = latest["ks"]

            if isinstance(ks_values, str):
                try:
                    import ast as _ast
                    ks_values = _ast.literal_eval(ks_values)
                except Exception:
                    ks_values = {}

            if isinstance(ks_values, dict) and ks_values:

                sensor_df = (
                    pd.Series(
                        ks_values,
                        dtype=float,
                    )
                    .sort_values()
                    .reset_index()
                )

                sensor_df.columns = [
                    "Sensor",
                    "KS",
                ]

                fig = go.Figure(
                    go.Bar(
                        x=sensor_df["KS"],
                        y=sensor_df["Sensor"],
                        orientation="h",
                    )
                )

                fig.update_layout(
                    title="Latest sensor-level KS statistics",
                    template="plotly_white",
                    paper_bgcolor=COLORS["panel"],
                    plot_bgcolor=COLORS["panel"],
                    font=dict(color=COLORS["text"]),
                    height=360,
                    xaxis=dict(
                        title="KS statistic",
                        gridcolor=COLORS["grid"],
                    ),
                    yaxis=dict(
                        title="Sensor",
                        gridcolor=COLORS["grid"],
                    ),
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

with tabs[2]:

    comparison = read_csv("model_comparison.csv")

    if comparison is None or comparison.empty:

        st.info(
            "Run `run_experiment.py` first to generate "
            "`model_comparison.csv`."
        )

    else:

        st.subheader("Learning strategy comparison")

        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
        )

        metric_candidates = [
            "MAE",
            "RMSE",
            "R2",
            "R²",
        ]

        available = [
            x for x in metric_candidates
            if x in comparison.columns
        ]

        strategy_col = next(
            (
                x for x in
                ["Strategy", "strategy", "Model", "Method"]
                if x in comparison.columns
            ),
            None,
        )

        if available and strategy_col:

            selected_metric = st.selectbox(
                "Metric",
                available,
            )

            fig = make_fig(
                f"{selected_metric} by learning strategy",
                selected_metric,
                400,
            )

            fig.add_trace(
                go.Bar(
                    x=comparison[strategy_col],
                    y=comparison[selected_metric],
                    name=selected_metric,
                )
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

        st.markdown("### Metrics")

        st.markdown(
            """
            **MAE ↓** — average absolute prediction error  
            **RMSE ↓** — penalizes larger errors more strongly  
            **R² ↑** — proportion of target variance explained
            """
        )


# ============================================================
# ADAPTATION & RECOVERY
# ============================================================

with tabs[3]:

    history = pd.DataFrame(
        st.session_state.history
    )

    if history.empty:
        history = read_csv("stream_log.csv")

    if history is None or history.empty:
        st.info(
            "Run the live stream or `run_experiment.py` first."
        )

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Drift alarms",
            int(
                history["alarm"].astype(bool).sum()
            )
            if "alarm" in history
            else 0,
        )

        c2.metric(
            "Adapted windows",
            int(
                history["adapted"].astype(bool).sum()
            )
            if "adapted" in history
            else 0,
        )

        c3.metric(
            "Recovery events",
            int(
                history["recovery"].astype(bool).sum()
            )
            if "recovery" in history
            else 0,
        )

        c4.metric(
            "Final MAE",
            f"{history['mae'].iloc[-1]:.3f}"
            if "mae" in history
            else "—",
        )

        fig = make_fig(
            "Prediction error and adaptation",
            "MAE",
            400,
        )

        if "mae_static" in history:
            fig.add_trace(
                go.Scatter(
                    x=history["window"],
                    y=history["mae_static"],
                    name="Static",
                    line=dict(
                        color=COLORS["static"],
                        width=2.5,
                    ),
                )
            )

        if "mae" in history:
            fig.add_trace(
                go.Scatter(
                    x=history["window"],
                    y=history["mae"],
                    name="Incremental SGD",
                    line=dict(
                        color=COLORS["incremental"],
                        width=3,
                    ),
                )
            )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.dataframe(
            history[
                [
                    c for c in [
                        "window",
                        "diagnosis",
                        "alarm",
                        "adapted",
                        "recovery",
                        "mae",
                        "rmse",
                        "r2",
                        "updates_seen",
                    ]
                    if c in history.columns
                ]
            ].tail(100),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# DETECTOR EVALUATION
# ============================================================

with tabs[4]:

    metrics = read_csv(
        "detector_metrics_summary.csv"
    )

    if metrics is None or metrics.empty:

        st.info(
            "Run `run_experiment.py` first to generate "
            "`detector_metrics_summary.csv`."
        )

    else:

        st.subheader(
            "Drift detector performance"
        )

        st.dataframe(
            metrics,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Ground-truth drift labels are used only here, "
            "offline, to evaluate the detector. They are not "
            "used during prediction, detection, diagnosis, "
            "or adaptation."
        )

        by_seed = read_csv(
            "detector_metrics_by_seed.csv"
        )

        if by_seed is not None and not by_seed.empty:
            st.markdown(
                "### Metrics across random seeds"
            )
            st.dataframe(
                by_seed,
                width="stretch",
                hide_index=True,
            )


# ============================================================
# EVENT LOG
# ============================================================

with tabs[5]:

    events = read_csv(
        "detector_events.csv"
    )

    if events is None or events.empty:

        st.info(
            "Run `run_experiment.py` first to generate "
            "`detector_events.csv`."
        )

    else:

        st.subheader("Detector event log")

        st.dataframe(
            events,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# LIVE STREAM EXECUTION
# ============================================================
# ============================================================
# LIVE STREAM EXECUTION
# ============================================================
#
# Streamlit reruns the script from top to bottom after every interaction.
# For a real live dashboard, process ONE window per rerun and keep the
# runner/history in session_state. This lets the browser receive the
# updated charts after every window.

if "stream_running" not in st.session_state:
    st.session_state.stream_running = False

if "stream_finished" not in st.session_state:
    st.session_state.stream_finished = False

if start:
    with st.spinner(
        "Generating the synthetic stream and initializing "
        "incremental/continual learners..."
    ):
        st.session_state.stream_df = load_data()

    st.session_state.live_runner = StrategyRunner(
        st.session_state.stream_df,
        TRAIN_END,
        WINDOW,
        seed=0,
        strategies=[
            "Static",
            "Incremental",
            "Continual + replay(100)",
            "Continual + replay(500)",
        ],
    )

    st.session_state.history = []
    st.session_state.events = []
    st.session_state.stream_index = 0
    st.session_state.stream_running = True
    st.session_state.stream_finished = False

    st.rerun()


# ------------------------------------------------------------
# PROCESS EXACTLY ONE WINDOW PER SCRIPT RUN
# ------------------------------------------------------------

if st.session_state.stream_running:

    runner = st.session_state.live_runner
    stream_df = st.session_state.stream_df
    windows = runner.windows
    i = st.session_state.stream_index

    if i < len(windows):

        s, e = windows[i]

        batch = runner.step(s, e)

        primary = batch["Incremental"]

        # Save every learning strategy.
        for strategy_name, row in batch.items():
            saved_row = dict(row)
            saved_row["strategy"] = strategy_name
            st.session_state.history.append(saved_row)

        # Detector/adaptation events are based on the Incremental stream.
        if primary.get("alarm"):
            st.session_state.events.append(
                {
                    "window": int(primary["window"]),
                    "type": "alarm",
                    "diagnosis": str(
                        primary.get("diagnosis", "")
                    ),
                }
            )

        if primary.get("adapted"):
            st.session_state.events.append(
                {
                    "window": int(primary["window"]),
                    "type": "adapt",
                    "diagnosis": str(
                        primary.get("diagnosis", "")
                    ),
                }
            )

        if show_truth:
            truth_events_now = build_truth_events(stream_df)

            for event in truth_events_now:
                if event["window"] == primary["window"]:
                    if not any(
                        x["type"] == "truth"
                        and x["window"] == event["window"]
                        for x in st.session_state.events
                    ):
                        st.session_state.events.append(event)

        st.session_state.stream_index += 1

        # If this was the final window, stop after displaying it.
        if st.session_state.stream_index >= len(windows):
            st.session_state.stream_running = False
            st.session_state.stream_finished = True


# ------------------------------------------------------------
# RENDER THE CURRENT LIVE STATE
# ------------------------------------------------------------

if st.session_state.history:

    history = pd.DataFrame(
        st.session_state.history
    )

    primary_history = history[
        history["strategy"] == "Incremental"
    ].copy()

    latest = primary_history.iloc[-1]

    # Current metrics.
    current_rows = {
        strategy: history[
            history["strategy"] == strategy
        ].iloc[-1]
        for strategy in [
            "Static",
            "Incremental",
            "Continual + replay(100)",
            "Continual + replay(500)",
        ]
        if not history[
            history["strategy"] == strategy
        ].empty
    }

    metric_cols = st.columns(6)

    metric_cols[0].metric(
        "CURRENT WINDOW",
        int(latest["window"]) + 1,
    )

    metric_cols[1].metric(
        "DIAGNOSIS",
        str(latest["diagnosis"]).upper(),
    )

    metric_cols[2].metric(
        "INCREMENTAL MAE",
        f"{current_rows['Incremental']['mae']:.3f}",
    )

    metric_cols[3].metric(
        "REPLAY 100 MAE",
        f"{current_rows['Continual + replay(100)']['mae']:.3f}",
    )

    metric_cols[4].metric(
        "REPLAY 500 MAE",
        f"{current_rows['Continual + replay(500)']['mae']:.3f}",
    )

    metric_cols[5].metric(
        "INCREMENTAL R²",
        f"{current_rows['Incremental']['r2']:.3f}",
    )

    # --------------------------------------------------------
    # CHART 1: LEARNING BEHAVIOUR
    # --------------------------------------------------------

    model_fig = make_fig(
        "Live learning behaviour",
        "MAE",
        430,
    )

    chart_specs = [
        (
            "Static",
            "Static — no adaptation",
            COLORS["static"],
            2.5,
        ),
        (
            "Incremental",
            "Incremental SGD",
            COLORS["incremental"],
            3,
        ),
        (
            "Continual + replay(100)",
            "Continual + Replay 100",
            "#2563EB",
            2.5,
        ),
        (
            "Continual + replay(500)",
            "Continual + Replay 500",
            "#9333EA",
            2.5,
        ),
    ]

    for strategy, label, color, width in chart_specs:

        h = history[
            history["strategy"] == strategy
        ]

        if not h.empty:
            model_fig.add_trace(
                go.Scatter(
                    x=h["window"],
                    y=h["mae"],
                    mode="lines+markers",
                    name=label,
                    line=dict(
                        color=color,
                        width=width,
                    ),
                    marker=dict(size=4),
                )
            )

    model_fig = add_event_lines(
        model_fig,
        st.session_state.events,
        show_truth,
    )

    st.plotly_chart(
        model_fig,
        width="stretch",
        key=f"live_model_chart_{len(history)}",
    )

    # --------------------------------------------------------
    # CHART 2: DRIFT DETECTION
    # --------------------------------------------------------

    drift_fig = make_fig(
        "Live drift detection",
        "Signal",
        400,
    )

    drift_fig.add_trace(
        go.Scatter(
            x=primary_history["window"],
            y=primary_history["worst_ks"],
            mode="lines+markers",
            name="Worst KS",
            line=dict(
                color=COLORS["data"],
                width=2.5,
            ),
            marker=dict(size=3),
        )
    )

    drift_fig.add_trace(
        go.Scatter(
            x=primary_history["window"],
            y=primary_history["worst_psi"],
            mode="lines+markers",
            name="Worst PSI",
            line=dict(
                color=COLORS["relational"],
                width=2.5,
            ),
            marker=dict(size=3),
        )
    )

    drift_fig.add_trace(
        go.Scatter(
            x=primary_history["window"],
            y=primary_history["ph_stat"],
            mode="lines+markers",
            name="Page-Hinkley",
            line=dict(
                color=COLORS["alarm"],
                width=2.5,
            ),
            marker=dict(size=3),
        )
    )

    st.plotly_chart(
        drift_fig,
        width="stretch",
        key=f"live_drift_chart_{len(history)}",
    )

    # --------------------------------------------------------
    # LIVE STATUS
    # --------------------------------------------------------

    total_windows = (
        len(st.session_state.live_runner.windows)
        if st.session_state.live_runner is not None
        else 0
    )

    if st.session_state.stream_running:

        progress = (
            st.session_state.stream_index
            / max(1, total_windows)
        )

        st.progress(
            progress,
            text=(
                f"Live window "
                f"{st.session_state.stream_index}/{total_windows} "
                f"• {str(latest['diagnosis']).upper()} "
                f"• Alarm: "
                f"{'YES' if latest['alarm'] else 'NO'} "
                f"• Adapted: "
                f"{'YES' if latest['adapted'] else 'NO'}"
            ),
        )

        # Wait a little, then trigger the next window. The browser has
        # already rendered the current charts before the next rerun.
        time.sleep(max(0.05, float(delay)))
        st.rerun()

    elif st.session_state.stream_finished:

        st.success(
            f"✓ Live stream completed — "
            f"{total_windows} windows processed."
        )

elif not st.session_state.stream_running:

    st.info(
        "Click **Start / restart live stream** to begin. "
        "The charts will update window-by-window like a live monitor."
    )
