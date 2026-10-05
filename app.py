"""
Streamlit dashboard for the live drift-detection project.

Run:
    streamlit run app.py

Backend: current data_generator.py + drift_engine_incremental.py
The predictor is SGDRegressor with incremental/continual adaptation.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_generator import generate, TRAIN_END, WINDOW
from drift_engine_incremental import DualRunner, SGDStreamEngine, make_windows


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
            color: {COLORS["text"]};
            margin: 0.6rem 0 0.25rem 0;
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
    "SGD incremental / continual adaptation"
)


# ============================================================
# EXISTING MODEL / DATA PIPELINE
# ============================================================

@st.cache_resource(show_spinner="Generating data and pre-training the source model…")
def load_assets():
    df = generate()
    return df, None


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


def add_event_lines(fig, events, show_truth):
    labels_seen = set()

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
            "truth": "True drift start",
            "alarm": "Drift detected",
            "adapt": "Adaptation / recovery",
        }[kind]

        if kind not in labels_seen:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=label,
                    line=dict(color=color, dash=dash, width=2),
                )
            )
            labels_seen.add(kind)

        fig.add_vline(
            x=event["window"],
            line_color=color,
            line_dash=dash,
            line_width=1.5,
            opacity=0.75,
        )

    return fig


def get_history_df():
    if not st.session_state.history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.history).sort_values("window")


def read_optional_csv(filename):
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def add_regime_bands(fig, df, show_truth):
    if not show_truth or "truth_regime" not in df.columns:
        return fig

    regimes = df[["window", "truth_regime"]].drop_duplicates().sort_values("window")

    for i in range(len(regimes)):
        start = regimes.iloc[i]["window"]
        end = (
            regimes.iloc[i + 1]["window"]
            if i + 1 < len(regimes)
            else df["window"].max()
        )

        regime = str(regimes.iloc[i]["truth_regime"]).lower()

        color = {
            "stable": "rgba(148,163,184,0.05)",
            "data_drift": "rgba(56,189,248,0.10)",
            "relational_drift": "rgba(167,139,250,0.10)",
            "both": "rgba(239,68,68,0.10)",
        }.get(regime, "rgba(148,163,184,0.04)")

        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            line_width=0,
            layer="below",
        )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Stream controls")

    delay = st.slider(
        "Seconds per window",
        min_value=0.0,
        max_value=2.0,
        value=0.2,
        step=0.1,
    )

    show_truth = st.checkbox(
        "Show simulated ground truth",
        value=False,
        help=(
            "Ground truth is displayed for demonstration/evaluation only. "
            "It is never used by the live detector."
        ),
    )

    start = st.button(
        "▶ Start / restart live stream",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.markdown("### Pipeline")
    st.markdown(
        """
        **1. Predict**  
        Incremental SGD model

        **2. Detect**  
        KS / PSI + Page-Hinkley

        **3. Diagnose**  
        DATA / RELATIONAL / BOTH

        **4. Adapt**  
        Incremental SGD update

        **5. Recover**  
        Monitor post-drift error
        """
    )

    st.divider()
    st.caption(
        "The existing data generator and drift engine are preserved. "
        "This file changes the Streamlit presentation only."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "events" not in st.session_state:
    st.session_state.events = []

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

with tabs[0]:
    metric_cols = st.columns(6)
    metric_slots = [c.empty() for c in metric_cols]

    st.markdown(
        '<div class="section-title">Live model behaviour</div>',
        unsafe_allow_html=True,
    )
    live_mae_slot = st.empty()

    st.markdown(
        '<div class="section-title">Prediction view</div>',
        unsafe_allow_html=True,
    )
    prediction_slot = st.empty()

with tabs[1]:
    left, right = st.columns(2)
    sensor_slot = left.empty()
    signal_slot = right.empty()
    st.markdown(
        '<div class="section-title">Drift / alarm timeline</div>',
        unsafe_allow_html=True,
    )
    timeline_slot = st.empty()

with tabs[2]:
    rolling_slot = st.empty()
    summary_slot = st.empty()
    saved_slot = st.empty()

with tabs[3]:
    recovery_slot = st.empty()
    adaptation_slot = st.empty()

with tabs[4]:
    detector_slot = st.empty()

with tabs[5]:
    log_slot = st.empty()


# ============================================================
# RENDER DASHBOARD
# ============================================================

def render_dashboard():
    d = get_history_df()

    if d.empty:
        return

    events = st.session_state.events
    last = st.session_state.history[-1]
    render_id = int(last.get("window", len(st.session_state.history) - 1))

    diagnosis = str(last.get("diagnosis", "none")).upper()

    metric_slots[0].metric(
        "CURRENT WINDOW",
        int(last["window"]) + 1,
    )

    metric_slots[1].metric(
        "DIAGNOSIS",
        diagnosis,
    )

    metric_slots[2].metric(
        "STATIC MAE",
        f'{float(last["mae_static"]):.2f}',
    )

    delta = float(last["mae"]) - float(last["mae_static"])

    metric_slots[3].metric(
        "INCREMENTAL MAE",
        f'{float(last["mae"]):.2f}',
        f"{delta:+.2f} vs static",
        delta_color="inverse",
    )

    alarms_now = sum(1 for e in events if e.get("type") == "alarm")
    adaptations_now = sum(1 for e in events if e.get("type") == "adapt")

    metric_slots[4].metric(
        "DETECTED ALARMS",
        alarms_now,
    )

    metric_slots[5].metric(
        "ADAPTATION EVENTS",
        adaptations_now,
    )

    # --------------------------------------------------------
    # LIVE MAE
    # --------------------------------------------------------

    fig = make_fig(
        "Live model error · lower is better",
        "MAE",
        390,
    )

    fig.add_trace(
        go.Scatter(
            x=d["window"],
            y=d["mae_static"],
            mode="lines+markers",
            name="Static — no adaptation",
            line=dict(color=COLORS["static"], width=2.5),
            marker=dict(size=4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d["window"],
            y=d["mae"],
            mode="lines+markers",
            name="Incremental SGD",
            line=dict(color=COLORS["incremental"], width=3),
            marker=dict(size=4),
        )
    )

    fig = add_regime_bands(fig, d, show_truth)
    fig = add_event_lines(fig, events, show_truth)

    live_mae_slot.plotly_chart(
        fig,
        use_container_width=True,
        key=f"live_mae_chart_{render_id}",
    )

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    actual_col = next(
        (x for x in ["actual", "y_true", "target"] if x in d.columns),
        None,
    )

    predicted_col = next(
        (x for x in ["predicted", "y_pred", "prediction"] if x in d.columns),
        None,
    )

    if actual_col and predicted_col:
        fig = make_fig(
            "Actual vs predicted",
            "Target",
            390,
        )

        fig.add_trace(
            go.Scatter(
                x=d["window"],
                y=d[actual_col],
                name="Actual",
                line=dict(color=COLORS["text"], width=2.5),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=d["window"],
                y=d[predicted_col],
                name="Predicted",
                line=dict(color=COLORS["incremental"], width=2.5),
            )
        )

        prediction_slot.plotly_chart(
            add_event_lines(fig, events, show_truth),
            use_container_width=True,
            key=f"prediction_chart_{render_id}",
        )
    else:
        prediction_slot.info(
            "The current DualRunner output exposes window-level MAE, "
            "but not aligned actual/predicted arrays. Therefore this "
            "dashboard does not fabricate an actual-vs-predicted chart."
        )

    # ========================================================
    # DRIFT ANALYSIS
    # ========================================================

    ks = last.get("ks") or {}

    if ks:
        series = (
            pd.Series(ks, dtype="float64")
            .dropna()
            .sort_values(ascending=True)
        )

        colors = [
            COLORS["alarm"] if value >= 0.3 else COLORS["data"]
            for value in series.values
        ]

        fig = go.Figure(
            go.Bar(
                x=series.values,
                y=series.index,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Sensor-level KS statistic",
            template="plotly_white",
            paper_bgcolor=COLORS["panel"],
            plot_bgcolor=COLORS["panel"],
            font=dict(color=COLORS["text"]),
            height=380,
            margin=dict(l=30, r=25, t=60, b=40),
            xaxis_title="KS statistic",
            yaxis_title="Sensor",
            xaxis=dict(gridcolor=COLORS["grid"]),
            yaxis=dict(gridcolor=COLORS["grid"]),
        )

        sensor_slot.plotly_chart(
            fig,
            use_container_width=True,
            key=f"sensor_ks_chart_{render_id}",
        )

    # --------------------------------------------------------
    # DATA VS RELATIONAL SIGNAL
    # --------------------------------------------------------

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=d["window"],
            y=d["worst_ks"],
            name="Worst KS · data drift",
            line=dict(color=COLORS["data"], width=2.5),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=d["window"],
            y=d["mean_resid"],
            name="Residual · relational drift",
            line=dict(color=COLORS["relational"], width=2.5),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Data-drift and relational-drift signals",
        template="plotly_white",
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"]),
        height=380,
        margin=dict(l=40, r=45, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
    )

    fig.update_xaxes(
        title="Stream window",
        gridcolor=COLORS["grid"],
    )

    fig.update_yaxes(
        title_text="Worst KS",
        secondary_y=False,
        gridcolor=COLORS["grid"],
    )

    fig.update_yaxes(
        title_text="Normalized old-rule residual",
        secondary_y=True,
        showgrid=False,
    )

    signal_slot.plotly_chart(
        add_event_lines(fig, events, show_truth),
        use_container_width=True,
        key=f"drift_signal_chart_{render_id}",
    )

    # --------------------------------------------------------
    # TIMELINE
    # --------------------------------------------------------

    fig = make_fig(
        "Drift → detection → adaptation",
        "Event",
        330,
    )

    event_y = {
        "truth": 3,
        "alarm": 2,
        "adapt": 1,
    }

    event_labels = {
        1: "Adaptation / recovery",
        2: "Drift detected",
        3: "True drift start",
    }

    for kind in ["truth", "alarm", "adapt"]:
        selected = [
            e for e in events
            if e["type"] == kind
            and (kind != "truth" or show_truth)
        ]

        if not selected:
            continue

        fig.add_trace(
            go.Scatter(
                x=[e["window"] for e in selected],
                y=[event_y[kind]] * len(selected),
                mode="markers",
                name=event_labels[event_y[kind]],
                marker=dict(
                    size=12,
                    color=COLORS[kind],
                    line=dict(color="#FFFFFF", width=1),
                ),
                hovertemplate=(
                    "Window %{x}<br>"
                    + event_labels[event_y[kind]]
                    + "<extra></extra>"
                ),
            )
        )

    fig.update_yaxes(
        tickvals=[1, 2, 3],
        ticktext=[
            "Adaptation",
            "Detected alarm",
            "True drift start",
        ],
        range=[0.5, 3.5],
    )

    timeline_slot.plotly_chart(
        fig,
        use_container_width=True,
        key=f"timeline_chart_{render_id}",
    )

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    fig = make_fig(
        "Rolling/window MAE comparison",
        "MAE",
        390,
    )

    fig.add_trace(
        go.Scatter(
            x=d["window"],
            y=d["mae_static"],
            name="Static",
            line=dict(color=COLORS["static"], width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d["window"],
            y=d["mae"],
            name="Incremental SGD",
            line=dict(color=COLORS["incremental"], width=3),
        )
    )

    rolling_slot.plotly_chart(
        add_event_lines(fig, events, show_truth),
        use_container_width=True,
        key=f"rolling_mae_chart_{render_id}",
    )

    # --------------------------------------------------------
    # SAVED MODEL COMPARISON
    # --------------------------------------------------------

    comparison = read_optional_csv("model_comparison.csv")

    if comparison is not None and not comparison.empty:
        summary_slot.subheader("Saved experiment · model comparison")

        preferred = [
            "Model",
            "Method",
            "Strategy",
            "MAE",
            "RMSE",
            "R2",
            "R²",
            "Training time (s)",
            "Recovery time",
            "train+predict seconds",
        ]

        shown = [c for c in preferred if c in comparison.columns]

        if shown:
            summary_slot.dataframe(
                comparison[shown],
                use_container_width=True,
                hide_index=True,
            )
        else:
            summary_slot.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )
    else:
        saved_slot.info(
            "model_comparison.csv was not found beside app.py. "
            "The live comparison above is still available."
        )

    # ========================================================
    # ADAPTATION / RECOVERY
    # ========================================================

    fig = make_fig(
        "Error around drift and adaptation",
        "MAE",
        390,
    )

    fig.add_trace(
        go.Scatter(
            x=d["window"],
            y=d["mae_static"],
            name="Static",
            line=dict(color=COLORS["static"], width=2.5),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=d["window"],
            y=d["mae"],
            name="Incremental SGD",
            line=dict(color=COLORS["incremental"], width=3),
        )
    )

    recovery_slot.plotly_chart(
        add_event_lines(fig, events, show_truth),
        use_container_width=True,
        key=f"recovery_chart_{render_id}",
    )

    alarms = [e for e in events if e["type"] == "alarm"]
    adaptations = [e for e in events if e["type"] == "adapt"]

    if alarms and adaptations:
        first_alarm = alarms[0]["window"]
        first_adapt = adaptations[0]["window"]
        delay_windows = max(0, first_adapt - first_alarm)
        delay_text = f"{delay_windows} windows"
    else:
        delay_text = "—"

    adaptation_slot.info(
        f"**Adaptation status:** {len(adaptations)} adaptation events · "
        f"{len(alarms)} detected alarms · "
        f"**first adaptation delay:** {delay_text}."
    )

    # ========================================================
    # DETECTOR EVALUATION
    # ========================================================

    alarm_count = int(d["alarm"].fillna(False).sum()) if "alarm" in d else 0
    recovery_count = int(d["recovery"].fillna(False).sum()) if "recovery" in d else 0

    detector_slot.subheader("Detector event evaluation")

    # Precision / recall / F1 / accuracy are for the BINARY DRIFT ALARM.
    # They are offline metrics only; ground truth is never used by the live detector.
    if show_truth and "true_drift_event" in d.columns:
        y_true = d["true_drift_event"].astype(int)
        y_pred = d["alarm"].astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        c1, c2, c3, c4 = detector_slot.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
        c2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
        c3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
        c4.metric("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")

        detector_slot.caption(
            f"TP={cm[1,1]} · FP={cm[0,1]} · FN={cm[1,0]} · TN={cm[0,0]}. "
            "These metrics evaluate drift-alarm detection, not regression prediction."
        )
    else:
        detector_slot.info(
            "Enable **Show simulated ground truth** to calculate offline "
            "accuracy, precision, recall and F1 for the drift alarm. "
            "The live detector never reads the ground-truth columns."
        )

    worst_ks = float(d["worst_ks"].iloc[-1]) if "worst_ks" in d.columns else float("nan")
    mean_resid = float(d["mean_resid"].iloc[-1]) if "mean_resid" in d.columns else float("nan")
    detector_slot.caption(
        f"Observed alarms: {alarm_count} · Recovery events: {recovery_count} · "
        f"Worst KS: {worst_ks:.3f} · Relational signal: {mean_resid:.3f}"
    )

    detector_columns = [
        c for c in [
            "window", "diagnosis", "alarm", "recovery",
            "worst_ks", "mean_resid", "true_drift_event", "truth_regime"
        ] if c in d.columns
    ]
    detector_slot.dataframe(d[detector_columns].tail(100), use_container_width=True, hide_index=True)

    # ========================================================
    # EVENT LOG
    # ========================================================

    event_rows = []

    for row in st.session_state.history:
        moved = row.get("features_moved") or []

        event_rows.append(
            {
                "Window": row["window"],
                "Diagnosis": row.get("diagnosis"),
                "Alarm": bool(row.get("alarm", False)),
                "Recovery": bool(row.get("recovery", False)),
                "Sensors moved": ", ".join(map(str, moved)),
                "Worst KS": row.get("worst_ks"),
                "Residual": row.get("mean_resid"),
                **(
                    {"Answer key": row.get("truth_regime")}
                    if show_truth
                    else {}
                ),
            }
        )

    log = pd.DataFrame(event_rows)

    if not log.empty:
        log_slot.dataframe(
            log.sort_values("Window", ascending=False).head(250),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# STREAM EXECUTION
# ============================================================

if start:
    df, _ = load_assets()

    runner = DualRunner(
        df,
        TRAIN_END,
        WINDOW,
        seed=0,
    )

    windows = list(
        make_windows(
            len(df),
            TRAIN_END,
            WINDOW,
        )
    )

    previous_truth = None

    progress = st.progress(
        0,
        text="Starting live stream…",
    )

    for i, (s, e) in enumerate(windows):
        msg = runner.step(s, e)

        # Ground truth is ONLY an offline/demo answer key.
        truth = (
            str(df.regime.iloc[s])
            if "regime" in df.columns
            else None
        )

        msg["truth_regime"] = truth
        if "true_drift_point" in df.columns:
            points = pd.to_numeric(df["true_drift_point"], errors="coerce").dropna().astype(int).to_numpy()
            msg["true_drift_event"] = bool(np.any((points >= s) & (points < e)))
        else:
            msg["true_drift_event"] = False
        st.session_state.history.append(msg)

        # True regime transition — visualization/evaluation only.
        if (
            truth is not None
            and previous_truth is not None
            and truth != previous_truth
        ):
            st.session_state.events.append(
                {
                    "window": msg["window"],
                    "type": "truth",
                    "description": f"{previous_truth} → {truth}",
                }
            )

        previous_truth = truth

        # Existing detector alarm.
        if msg.get("alarm"):
            st.session_state.events.append(
                {
                    "window": msg["window"],
                    "type": "alarm",
                    "description": str(msg.get("diagnosis", "")),
                }
            )

        # Existing recovery/re-adaptation event.
        if msg.get("recovery"):
            st.session_state.events.append(
                {
                    "window": msg["window"],
                    "type": "adapt",
                    "description": "Drift ended → re-adapt",
                }
            )

        render_dashboard()

        progress.progress(
            (i + 1) / len(windows),
            text=f"Streaming window {i + 1}/{len(windows)}",
        )

        if delay > 0:
            time.sleep(delay)

    progress.empty()
    st.success("Stream finished.")

elif st.session_state.history:
    render_dashboard()

else:
    st.info(
        "Press **Start / restart live stream** in the sidebar to begin."
    )
