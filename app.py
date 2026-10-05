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
            color: {COLORS["text"]};
            margin: 0.6rem 0 0.25rem 0;
        }}

        .small-note {{
            color: {COLORS["muted"]};
            font-size: 0.85rem;
        }}

        .status-pill {{
            display: inline-block;
            padding: 5px 11px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: .02em;
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
    "KS / PSI data drift  •  Page-Hinkley relational drift  •  Incremental / Continual Learning"
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Generating synthetic stream data…")
def load_data():
    return generate()


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


def diagnosis_color(diagnosis):
    value = str(diagnosis).lower()
    if value == "both":
        return COLORS["danger"]
    if value == "data":
        return COLORS["data"]
    if value == "relational":
        return COLORS["relational"]
    return COLORS["adapt"]


def read_optional_csv(filename):
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def add_regime_bands(fig, df, show_truth):
    if not show_truth or "regime" not in df.columns:
        return fig

    # Only add bands when the generated dataset actually exposes regimes.
    regimes = df[["window", "regime"]].drop_duplicates().sort_values("window")

    for i in range(len(regimes)):
        start = regimes.iloc[i]["window"]
        end = (
            regimes.iloc[i + 1]["window"]
            if i + 1 < len(regimes)
            else df["window"].max()
        )

        regime = str(regimes.iloc[i]["regime"]).lower()

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
        help="Ground truth is displayed for demonstration/evaluation only. "
             "It is never used by the live detector.",
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
        Incremental / continual learning

        **5. Recover**  
        Monitor post-drift error
        """
    )

    st.divider()
    st.caption(
        "Static, Incremental SGD, and Continual Learning with Replay(100/500). "
        "Ground truth is used only for evaluation/visualization."
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

if "stream_running" not in st.session_state:
    st.session_state.stream_running = False

if "stream_finished" not in st.session_state:
    st.session_state.stream_finished = False

if start:
    st.session_state.history = []
    st.session_state.events = []
    st.session_state.stream_index = 0
    st.session_state.stream_running = True
    st.session_state.stream_finished = False
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
# DATA HELPERS
# ============================================================

def history_for(strategy=None):
    d = pd.DataFrame(st.session_state.history)

    if d.empty:
        return d

    if strategy is not None and "strategy" in d.columns:
        d = d[d["strategy"] == strategy].copy()

    return d.sort_values("window").reset_index(drop=True)


def current_rows():
    d = pd.DataFrame(st.session_state.history)

    if d.empty or "strategy" not in d.columns:
        return {}

    result = {}
    for name in [
        "Static",
        "Incremental",
        "Continual + replay(100)",
        "Continual + replay(500)",
    ]:
        x = d[d["strategy"] == name]
        if not x.empty:
            result[name] = x.iloc[-1]

    return result


def build_truth_events(data):
    events = []

    if data is None or "true_drift_point" not in data.columns:
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

        if point >= TRAIN_END:
            events.append(
                {
                    "window": int(
                        (point - TRAIN_END) // WINDOW
                    ),
                    "type": "truth",
                }
            )

    return events


def add_event_lines_clean(fig, events, show_truth):
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
            "alarm": "New drift alarm",
            "adapt": "Adaptation",
        }[kind]

        if kind not in labels_seen:
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
            labels_seen.add(kind)

        fig.add_vline(
            x=event["window"],
            line_color=color,
            line_dash=dash,
            line_width=1.5,
            opacity=0.75,
        )

    return fig


def plot_learning_chart(d, height=390):
    fig = make_fig(
        "Live learning behaviour · lower MAE is better",
        "MAE",
        height,
    )

    specs = [
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

    for strategy, label, color, width in specs:
        h = d[d["strategy"] == strategy]

        if h.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=h["window"],
                y=h["mae"],
                mode="lines+markers",
                name=label,
                line=dict(
                    color=color,
                    width=width,
                ),
                marker=dict(size=3),
            )
        )

    return fig


def plot_drift_chart(d, height=390):
    fig = make_fig(
        "Drift detector signals",
        "Signal",
        height,
    )

    if d.empty:
        return fig

    inc = d[d["strategy"] == "Incremental"]

    if inc.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=inc["window"],
            y=inc["worst_ks"],
            mode="lines+markers",
            name="Worst KS",
            line=dict(
                color=COLORS["data"],
                width=2.5,
            ),
            marker=dict(size=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=inc["window"],
            y=inc["worst_psi"],
            mode="lines+markers",
            name="Worst PSI",
            line=dict(
                color=COLORS["relational"],
                width=2.5,
            ),
            marker=dict(size=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=inc["window"],
            y=inc["ph_stat"],
            mode="lines+markers",
            name="Page-Hinkley",
            line=dict(
                color=COLORS["alarm"],
                width=2.5,
            ),
            marker=dict(size=3),
        )
    )

    return fig


# ============================================================
# LIVE MONITOR
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">Live model behaviour</div>',
        unsafe_allow_html=True,
    )

    rows = current_rows()

    if not rows:
        st.info(
            "Press **Start / restart live stream** to begin. "
            "The charts will update one stream window at a time."
        )

    else:

        last = rows["Incremental"]

        cols = st.columns(6)

        cols[0].metric(
            "CURRENT WINDOW",
            int(last["window"]) + 1,
        )

        cols[1].metric(
            "CURRENT DIAGNOSIS",
            str(last["diagnosis"]).upper(),
        )

        cols[2].metric(
            "NEW ALARM",
            "YES" if bool(last["alarm"]) else "NO",
        )

        cols[3].metric(
            "INCREMENTAL MAE",
            f"{last['mae']:.3f}",
        )

        cols[4].metric(
            "REPLAY 100 MAE",
            f"{rows['Continual + replay(100)']['mae']:.3f}",
        )

        cols[5].metric(
            "INCREMENTAL R²",
            f"{last['r2']:.3f}",
        )

        all_history = pd.DataFrame(
            st.session_state.history
        )

        st.plotly_chart(
            add_event_lines_clean(
                plot_learning_chart(all_history),
                st.session_state.events,
                show_truth,
            ),
            width="stretch",
            key=f"live_learning_{len(all_history)}",
        )

        st.plotly_chart(
            plot_drift_chart(all_history),
            width="stretch",
            key=f"live_drift_{len(all_history)}",
        )

        st.caption(
            "Diagnosis shows the current drift state. "
            "**New Alarm** shows whether a new alarm fired in this window. "
            "A diagnosis can remain DATA/RELATIONAL/BOTH after the original "
            "alarm because the detector keeps the active drift state latched."
        )


# ============================================================
# DRIFT ANALYSIS
# ============================================================

with tabs[1]:

    inc = history_for("Incremental")

    if inc.empty:
        st.info(
            "Run the live stream first to populate drift analysis."
        )

    else:

        left, right = st.columns(2)

        with left:
            fig = make_fig(
                "Worst KS statistic",
                "KS statistic",
                360,
            )

            fig.add_trace(
                go.Scatter(
                    x=inc["window"],
                    y=inc["worst_ks"],
                    mode="lines+markers",
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
            fig = make_fig(
                "Worst PSI",
                "PSI",
                360,
            )

            fig.add_trace(
                go.Scatter(
                    x=inc["window"],
                    y=inc["worst_psi"],
                    mode="lines+markers",
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

        fig = make_fig(
            "Page-Hinkley relational-drift signal",
            "PH statistic",
            360,
        )

        fig.add_trace(
            go.Scatter(
                x=inc["window"],
                y=inc["ph_stat"],
                mode="lines+markers",
                name="Page-Hinkley",
                line=dict(
                    color=COLORS["alarm"],
                    width=2.5,
                ),
            )
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

with tabs[2]:

    comparison = read_optional_csv(
        "model_comparison.csv"
    )

    expected_strategies = [
        "Static",
        "Full retraining",
        "Incremental",
        "Continual + replay(100)",
        "Continual + replay(500)",
    ]

    if comparison is None or comparison.empty:

        st.info(
            "Run `run_experiment.py` to generate the current "
            "five-strategy `model_comparison.csv`."
        )

    else:

        # Never display legacy results such as Transfer learning.
        # Also refuse to display an incomplete comparison, because that
        # could make an old CSV look like a valid current experiment.
        available = set(
            comparison.get("Strategy", pd.Series(dtype=str))
            .astype(str)
            .tolist()
        )

        missing = [
            name
            for name in expected_strategies
            if name not in available
        ]

        if missing:
            st.warning(
                "The existing `model_comparison.csv` is outdated or "
                "incomplete. It is not being displayed. Run "
                "`python run_experiment.py` to regenerate the results "
                "with Incremental SGD and Continual Replay (100/500)."
            )
        else:
            comparison = comparison[
                comparison["Strategy"].isin(expected_strategies)
            ].copy()

            order = {
                name: i
                for i, name in enumerate(expected_strategies)
            }
            comparison["_order"] = comparison["Strategy"].map(order)
            comparison = (
                comparison
                .sort_values("_order")
                .drop(columns="_order")
            )

            st.subheader(
                "Offline learning-strategy comparison"
            )

            shown = [
                "Strategy",
                "MAE",
                "RMSE",
                "R2",
                "Windows",
            ]

            st.dataframe(
                comparison[
                    [c for c in shown if c in comparison.columns]
                ],
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "Lower MAE/RMSE is better; higher R² is better. "
                "Transfer learning / fine-tuning is not part of the "
                "current experiment."
            )

    live = pd.DataFrame(
        st.session_state.history
    )

    if not live.empty:

        st.markdown(
            "### Live MAE comparison"
        )

        st.plotly_chart(
            plot_learning_chart(
                live,
                height=430,
            ),
            width="stretch",
        )

        st.markdown(
            """
            **MAE ↓** — average absolute prediction error  
            **RMSE ↓** — penalizes large errors more strongly  
            **R² ↑** — variance explained by the model
            """
        )


# ============================================================
# ADAPTATION & RECOVERY
# ============================================================

with tabs[3]:

    inc = history_for("Incremental")

    if inc.empty:

        st.info(
            "Run the live stream first."
        )

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "NEW ALARMS",
            int(inc["alarm"].astype(bool).sum()),
        )

        c2.metric(
            "ADAPTATION WINDOWS",
            int(inc["adapted"].astype(bool).sum()),
        )

        c3.metric(
            "RECOVERY EVENTS",
            int(inc["recovery"].astype(bool).sum()),
        )

        c4.metric(
            "FINAL MAE",
            f"{inc['mae'].iloc[-1]:.3f}",
        )

        st.plotly_chart(
            add_event_lines_clean(
                plot_learning_chart(
                    pd.DataFrame(
                        st.session_state.history
                    ),
                    400,
                ),
                st.session_state.events,
                show_truth,
            ),
            width="stretch",
        )

        st.dataframe(
            inc[
                [
                    c for c in [
                        "window",
                        "diagnosis",
                        "alarm",
                        "recovery",
                        "adapted",
                        "mae",
                        "rmse",
                        "r2",
                        "updates_seen",
                    ]
                    if c in inc.columns
                ]
            ].tail(100),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# DETECTOR EVALUATION
# ============================================================

with tabs[4]:

    summary = read_optional_csv(
        "detector_metrics_summary.csv"
    )

    if summary is None or summary.empty:

        st.info(
            "Run `run_experiment.py` first to generate "
            "detector evaluation metrics."
        )

    else:

        st.subheader(
            "Offline drift-detector evaluation"
        )

        st.caption(
            "Primary metric is event-level detection: one alarm is matched "
            "to a true drift episode if it occurs during that episode or "
            "within the allowed delay. This avoids penalising a detector "
            "for not raising a new alarm on every window of a long drift."
        )

        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Ground-truth drift labels are used only for offline "
            "evaluation. They are never used by the live detector. "
            "TN/Accuracy are intentionally not headline metrics for "
            "event-based drift detection."
        )

        by_seed = read_optional_csv(
            "detector_metrics_by_seed.csv"
        )

        if by_seed is not None and not by_seed.empty:
            # The detector itself is deterministic for a fixed stream, so
            # the seed table is retained only as an audit view.
            with st.expander("Detailed detector results by seed"):
                st.dataframe(
                    by_seed,
                    width="stretch",
                    hide_index=True,
                )

        window_metrics = read_optional_csv(
            "detector_window_metrics_by_seed.csv"
        )

        if window_metrics is not None and not window_metrics.empty:
            with st.expander("Legacy window-level metrics (diagnostic only)"):
                st.dataframe(
                    window_metrics,
                    width="stretch",
                    hide_index=True,
                )


# ============================================================
# EVENT LOG
# ============================================================

with tabs[5]:

    inc = history_for("Incremental")

    if inc.empty:

        st.info(
            "Run the live stream first."
        )

    else:

        log_cols = [
            c for c in [
                "window",
                "diagnosis",
                "data_alarm",
                "relational_alarm",
                "alarm",
                "adapted",
                "recovery",
                "worst_ks",
                "worst_psi",
                "ph_stat",
                "mean_resid",
                "features_moved",
            ]
            if c in inc.columns
        ]

        st.dataframe(
            inc[log_cols].tail(250),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# ONE-WINDOW LIVE EXECUTION
# ============================================================
#
# IMPORTANT:
# We process exactly one window per rerun. This makes Plotly charts
# visibly update in the browser instead of waiting for the full stream.

if st.session_state.stream_running:

    runner = st.session_state.live_runner
    data = st.session_state.stream_df

    if runner is not None and st.session_state.stream_index < len(
        runner.windows
    ):

        i = st.session_state.stream_index
        s, e = runner.windows[i]

        batch = runner.step(s, e)

        for strategy, row in batch.items():

            saved = dict(row)
            saved["strategy"] = strategy
            st.session_state.history.append(saved)

        # Detector events come from the Incremental stream only.
        primary = batch["Incremental"]

        if primary["alarm"]:

            st.session_state.events.append(
                {
                    "window": int(primary["window"]),
                    "type": "alarm",
                }
            )

        if primary["adapted"]:

            st.session_state.events.append(
                {
                    "window": int(primary["window"]),
                    "type": "adapt",
                }
            )

        # Ground truth is only used to draw an optional answer-key line.
        if show_truth:

            truth_events = build_truth_events(data)

            for event in truth_events:

                if (
                    event["window"] == primary["window"]
                    and not any(
                        e["type"] == "truth"
                        and e["window"] == event["window"]
                        for e in st.session_state.events
                    )
                ):

                    st.session_state.events.append(event)

        st.session_state.stream_index += 1

        if st.session_state.stream_index >= len(
            runner.windows
        ):

            st.session_state.stream_running = False
            st.session_state.stream_finished = True

        else:

            # Re-run immediately after the current chart has been built.
            # The next rerun receives the next window and redraws the
            # charts with one more point.
            time.sleep(
                max(
                    0.05,
                    float(delay),
                )
            )

            st.rerun()

    else:

        st.session_state.stream_running = False
        st.session_state.stream_finished = True


if st.session_state.stream_finished:

    st.success(
        "✓ Live stream completed. "
        "Use the tabs above to inspect drift, model performance, "
        "adaptation/recovery, detector evaluation, and the event log."
    )
