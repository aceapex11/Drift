"""
Professional Streamlit frontend for the live drift-detection project.

Run:
    streamlit run app.py

The dashboard uses StrategyRunner from drift_engine.py.
Ground-truth labels are NOT displayed or used by the live detector.
They remain available only to run_experiment.py for offline evaluation.
"""

import time
from pathlib import Path

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
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg": "#F4F7FB",
    "panel": "#FFFFFF",
    "border": "#E2E8F0",
    "grid": "#E8EDF4",
    "text": "#172033",
    "muted": "#64748B",
    "static": "#64748B",
    "incremental": "#16A34A",
    "replay100": "#2563EB",
    "replay500": "#7C3AED",
    "data": "#0284C7",
    "relational": "#7C3AED",
    "alarm": "#D97706",
    "adapt": "#0F766E",
    "danger": "#DC2626",
}

STRATEGIES = [
    "Static",
    "Incremental",
    "Continual + replay(100)",
    "Continual + replay(500)",
]

STRATEGY_LABELS = {
    "Static": "Static",
    "Incremental": "Incremental SGD",
    "Continual + replay(100)": "Replay 100",
    "Continual + replay(500)": "Replay 500",
}

STRATEGY_COLORS = {
    "Static": COLORS["static"],
    "Incremental": COLORS["incremental"],
    "Continual + replay(100)": COLORS["replay100"],
    "Continual + replay(500)": COLORS["replay500"],
}

st.markdown(
    """
    <style>
        .stApp { background: #F4F7FB; color: #172033; }
        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }
        [data-testid="stSidebar"] * { color: #172033; }
        [data-testid="stHeader"] { background: #FFFFFF; }
        [data-testid="stToolbar"] { background: #FFFFFF; }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 12px 14px;
        }
        [data-testid="stMetricLabel"] {
            color: #64748B;
            font-size: 0.78rem;
            font-weight: 650;
        }
        [data-testid="stMetricValue"] {
            color: #172033;
            font-size: 1.45rem;
        }
        .hero {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 14px;
        }
        .hero-title {
            font-size: 1.65rem;
            font-weight: 760;
            color: #172033;
            margin-bottom: 3px;
        }
        .hero-subtitle {
            color: #64748B;
            font-size: 0.9rem;
        }
        .section-title {
            font-size: 1.02rem;
            font-weight: 720;
            color: #172033;
            margin: 1rem 0 0.35rem 0;
        }
        .section-note {
            color: #64748B;
            font-size: 0.82rem;
            margin-bottom: 0.45rem;
        }
        div[data-testid="stTabs"] button {
            font-weight: 680;
        }
        .stButton > button {
            border-radius: 9px;
            font-weight: 650;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">◈ Live Drift Intelligence</div>
        <div class="hero-subtitle">
            Predict → Detect → Diagnose → Adapt → Recover
            &nbsp;·&nbsp; KS / PSI data drift
            &nbsp;·&nbsp; Page-Hinkley relational drift
            &nbsp;·&nbsp; Incremental & continual learning
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA / SESSION STATE
# ============================================================

@st.cache_data(show_spinner="Generating synthetic stream data…")
def load_data():
    return generate()


for key, default in {
    "history": [],
    "events": [],
    "live_runner": None,
    "stream_df": None,
    "stream_index": 0,
    "stream_running": False,
    "stream_finished": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# HELPERS
# ============================================================

def read_optional_csv(filename):
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def history_df():
    if not st.session_state.history:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.history).sort_values(
        ["window", "strategy"]
    ).reset_index(drop=True)


def strategy_history(strategy):
    d = history_df()
    if d.empty:
        return d
    return d[d["strategy"] == strategy].sort_values("window")


def current_rows():
    d = history_df()
    if d.empty or "strategy" not in d.columns:
        return {}
    return {
        strategy: d[d["strategy"] == strategy].iloc[-1]
        for strategy in STRATEGIES
        if not d[d["strategy"] == strategy].empty
    }


def current_best_model():
    rows = current_rows()
    if not rows:
        return "—", None
    best = min(rows.items(), key=lambda item: float(item[1]["mae"]))
    return STRATEGY_LABELS[best[0]], float(best[1]["mae"])


def add_event_lines(fig, events):
    seen = set()
    for event in events:
        kind = event.get("type")
        if kind not in {"alarm", "adapt"}:
            continue

        color = COLORS["alarm"] if kind == "alarm" else COLORS["adapt"]
        dash = "solid" if kind == "alarm" else "dot"
        label = "New drift alarm" if kind == "alarm" else "Adaptation"

        if kind not in seen:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="lines", name=label,
                    line=dict(color=color, dash=dash, width=2),
                )
            )
            seen.add(kind)

        fig.add_vline(
            x=event["window"],
            line_color=color,
            line_dash=dash,
            line_width=1.3,
            opacity=0.7,
        )
    return fig


def base_fig(title, y_title, height=360):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=16)),
        template="plotly_white",
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"]),
        height=height,
        margin=dict(l=48, r=28, t=58, b=42),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=11),
        ),
        xaxis=dict(
            title="Stream window", gridcolor=COLORS["grid"],
            zeroline=False, fixedrange=False,
        ),
        yaxis=dict(
            title=y_title, gridcolor=COLORS["grid"],
            zeroline=False,
        ),
    )
    return fig


def plot_best_model(d, height=350):
    """Show only the model that has the lowest MAE at each window."""
    fig = base_fig("Best model at each window", "MAE", height)
    if d.empty:
        return fig

    pivot = d.pivot_table(
        index="window", columns="strategy", values="mae", aggfunc="first"
    ).sort_index()

    winner = pivot.idxmin(axis=1, skipna=True)
    best_value = pivot.min(axis=1, skipna=True)

    # One continuous line connects every window.
    # The marker colour identifies which strategy won that window.
    # This avoids gaps when the best strategy changes from one window to the next.
    winner_labels = winner.map(STRATEGY_LABELS).tolist()
    winner_colors = [STRATEGY_COLORS.get(w, COLORS["incremental"]) for w in winner]

    fig.add_trace(
        go.Scatter(
            x=pivot.index,
            y=best_value,
            mode="lines+markers",
            name="Best model",
            line=dict(color=COLORS["text"], width=3),
            marker=dict(
                size=8,
                color=winner_colors,
                line=dict(color=COLORS["panel"], width=1.5),
            ),
            customdata=winner_labels,
            hovertemplate=(
                "Window %{x}<br>"
                "Best model: %{customdata}<br>"
                "MAE: %{y:.3f}<extra></extra>"
            ),
        )
    )

    return fig


def plot_all_models(d, height=360):
    fig = base_fig("Model MAE comparison", "MAE", height)
    if d.empty:
        return fig

    for strategy in STRATEGIES:
        h = d[d["strategy"] == strategy]
        if h.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=h["window"], y=h["mae"],
                mode="lines+markers",
                name=STRATEGY_LABELS[strategy],
                line=dict(color=STRATEGY_COLORS[strategy], width=2.4),
                marker=dict(size=3),
            )
        )
    return fig


def plot_drift_signals(d, height=330):
    fig = base_fig("Drift detector signals", "Signal", height)
    if d.empty:
        return fig
    inc = d[d["strategy"] == "Incremental"]
    if inc.empty:
        return fig

    signals = [
        ("worst_ks", "Worst KS", COLORS["data"]),
        ("worst_psi", "Worst PSI", COLORS["relational"]),
        ("ph_stat", "Page-Hinkley", COLORS["alarm"]),
    ]
    for col, label, color in signals:
        if col in inc.columns:
            fig.add_trace(
                go.Scatter(
                    x=inc["window"], y=inc[col],
                    mode="lines", name=label,
                    line=dict(color=color, width=2.2),
                )
            )

    # Show detector events directly on the drift-signal chart so that
    # the signal and the resulting action can be read on the same timeline.
    fig = add_event_lines(fig, st.session_state.events)
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### Stream controls")
    delay = st.slider(
        "Seconds per window", 0.0, 1.0, 0.2, 0.1,
        help="Controls the live playback speed.",
    )

    start = st.button(
        "▶  Start / restart stream",
        type="primary",
        width="stretch",
    )

    st.divider()
    st.markdown("### Pipeline")
    st.markdown(
        "**01 Predict**  ·  SGDRegressor\n\n"
        "**02 Detect**  ·  KS / PSI + Page-Hinkley\n\n"
        "**03 Diagnose**  ·  Data / Relational / Both\n\n"
        "**04 Adapt**  ·  Incremental / Replay\n\n"
        "**05 Recover**  ·  Post-drift error"
    )

    st.divider()
    st.caption(
        "Ground truth is intentionally hidden from the live dashboard. "
        "It is used only for offline detector evaluation."
    )


# ============================================================
# START / RESTART
# ============================================================

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
        strategies=STRATEGIES,
    )


# ============================================================
# TWO-PAGE LAYOUT
# ============================================================

tab_dashboard, tab_analysis = st.tabs([
    "● Dashboard",
    "◎ Evaluation & Analysis",
])


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

with tab_dashboard:
    rows = current_rows()

    if not rows:
        st.info("Click **Start / restart stream** to begin the live dashboard.")
    else:
        inc = rows["Incremental"]
        best_name, best_mae = current_best_model()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CURRENT WINDOW", int(inc["window"]) + 1)
        c2.metric("BEST MODEL", best_name)
        c3.metric("BEST MAE", f"{best_mae:.3f}")
        c4.metric("DIAGNOSIS", str(inc["diagnosis"]).upper())
        c5.metric("NEW ALARM", "YES" if bool(inc["alarm"]) else "NO")

        all_history = history_df()

        st.markdown('<div class="section-title">Best model — winner only</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">At every stream window, only the model with the lowest MAE is plotted.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            add_event_lines(
                plot_best_model(all_history),
                st.session_state.events,
            ),
            width="stretch",
            key=f"best_model_{len(all_history)}",
        )

        st.markdown('<div class="section-title">Model comparison</div>', unsafe_allow_html=True)
        st.plotly_chart(
            add_event_lines(
                plot_all_models(all_history),
                st.session_state.events,
            ),
            width="stretch",
            key=f"all_models_{len(all_history)}",
        )

        st.markdown('<div class="section-title">Drift detection</div>', unsafe_allow_html=True)
        st.plotly_chart(
            plot_drift_signals(all_history),
            width="stretch",
            key=f"drift_{len(all_history)}",
        )

        inc_hist = strategy_history("Incremental")
        if not inc_hist.empty:
            a, b, c = st.columns(3)
            a.metric("DRIFT ALARMS", int(inc_hist["alarm"].astype(bool).sum()))
            b.metric("ADAPTATION WINDOWS", int(inc_hist["adapted"].astype(bool).sum()))
            c.metric("RECOVERY EVENTS", int(inc_hist["recovery"].astype(bool).sum()))


# ============================================================
# PAGE 2 — EVALUATION & ANALYSIS
# ============================================================

with tab_analysis:
    st.markdown('<div class="section-title">Learning strategy summary</div>', unsafe_allow_html=True)
    comparison = read_optional_csv("model_comparison.csv")
    if comparison is not None and not comparison.empty:
        st.dataframe(comparison, width="stretch", hide_index=True)
    else:
        st.info("Run `run_experiment.py` to generate model comparison results.")

    st.markdown('<div class="section-title">Offline drift-detector evaluation</div>', unsafe_allow_html=True)
    summary = read_optional_csv("detector_metrics_summary.csv")
    if summary is None or summary.empty:
        st.info("Run `run_experiment.py` to generate detector evaluation metrics.")
    else:
        st.caption(
            "Event-level evaluation: an alarm is matched to a true drift episode "
            "when it occurs during the episode or within the allowed detection delay. "
            "Ground truth is used only offline."
        )
        st.dataframe(summary, width="stretch", hide_index=True)

        by_seed = read_optional_csv("detector_metrics_by_seed.csv")
        if by_seed is not None and not by_seed.empty:
            with st.expander("Detailed detector results by seed"):
                st.dataframe(by_seed, width="stretch", hide_index=True)

        window_metrics = read_optional_csv("detector_window_metrics_by_seed.csv")
        if window_metrics is not None and not window_metrics.empty:
            with st.expander("Legacy window-level metrics — diagnostic only"):
                st.dataframe(window_metrics, width="stretch", hide_index=True)

    st.markdown('<div class="section-title">Recent detector events</div>', unsafe_allow_html=True)
    inc = strategy_history("Incremental")
    if inc.empty:
        st.info("Run the live stream to populate detector events.")
    else:
        log_cols = [
            c for c in [
                "window", "diagnosis", "data_alarm", "relational_alarm",
                "alarm", "adapted", "recovery", "worst_ks", "worst_psi",
                "ph_stat", "mean_resid", "features_moved",
            ] if c in inc.columns
        ]
        st.dataframe(inc[log_cols].tail(100), width="stretch", hide_index=True)


# ============================================================
# ONE-WINDOW LIVE EXECUTION
# ============================================================

if st.session_state.stream_running:
    runner = st.session_state.live_runner
    data = st.session_state.stream_df

    if runner is not None and st.session_state.stream_index < len(runner.windows):
        i = st.session_state.stream_index
        s, e = runner.windows[i]
        batch = runner.step(s, e)

        for strategy, row in batch.items():
            saved = dict(row)
            saved["strategy"] = strategy
            st.session_state.history.append(saved)

        primary = batch["Incremental"]

        if primary.get("alarm"):
            st.session_state.events.append({
                "window": int(primary["window"]),
                "type": "alarm",
            })

        if primary.get("adapted"):
            st.session_state.events.append({
                "window": int(primary["window"]),
                "type": "adapt",
            })

        st.session_state.stream_index += 1

        if st.session_state.stream_index >= len(runner.windows):
            st.session_state.stream_running = False
            st.session_state.stream_finished = True
        else:
            time.sleep(max(0.05, float(delay)))

        st.rerun()
