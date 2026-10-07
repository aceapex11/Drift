"""Professional 2-page Streamlit dashboard for the live drift project."""

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_generator import generate, TRAIN_END, WINDOW
from drift_engine import StrategyRunner, STRATEGY_NAMES


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
    "full": "#EA580C",
    "incremental": "#16A34A",
    "replay100": "#2563EB",
    "replay500": "#7C3AED",
    "data": "#0284C7",
    "relational": "#7C3AED",
    "alarm": "#D97706",
    "adapt": "#0F766E",
    "recovery": "#059669",
    "danger": "#DC2626",
}

STRATEGIES = [
    "Static",
    "Full retraining",
    "Incremental",
    "Continual + replay(100)",
    "Continual + replay(500)",
]

# Keep this aligned with the engine. If the engine is changed later, use the
# engine's canonical names but preserve the dashboard labels.
if hasattr(STRATEGY_NAMES, "__iter__"):
    STRATEGIES = [s for s in STRATEGIES if s in STRATEGY_NAMES]

STRATEGY_LABELS = {
    "Static": "Static",
    "Full retraining": "Full retrain",
    "Incremental": "Incremental SGD",
    "Continual + replay(100)": "Replay 100",
    "Continual + replay(500)": "Replay 500",
}

STRATEGY_COLORS = {
    "Static": COLORS["static"],
    "Full retraining": COLORS["full"],
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
            padding: 11px 13px;
        }
        [data-testid="stMetricLabel"] {
            color: #64748B;
            font-size: 0.75rem;
            font-weight: 650;
        }
        [data-testid="stMetricValue"] {
            color: #172033;
            font-size: 1.38rem;
        }
        .hero {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 19px 23px;
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
            font-size: 0.88rem;
        }
        .section-title {
            font-size: 1.01rem;
            font-weight: 720;
            color: #172033;
            margin: 1rem 0 0.3rem 0;
        }
        .section-note {
            color: #64748B;
            font-size: 0.80rem;
            margin-bottom: 0.42rem;
        }
        .status-pill {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            border: 1px solid #E2E8F0;
            background: #FFFFFF;
            margin-bottom: 7px;
        }
        div[data-testid="stTabs"] button { font-weight: 680; }
        .stButton > button { border-radius: 9px; font-weight: 650; }
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


DEFAULTS = {
    "history": [],
    "events": [],
    "live_runner": None,
    "stream_df": None,
    "stream_index": 0,
    "stream_running": False,
    "stream_finished": False,
}

for key, default in DEFAULTS.items():
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
    return STRATEGY_LABELS.get(best[0], best[0]), float(best[1]["mae"])


def add_event_lines(fig, events):
    seen = set()
    for event in events:
        kind = event.get("type")
        if kind not in {"alarm", "adapt", "recovery"}:
            continue

        colors = {
            "alarm": (COLORS["alarm"], "solid", "New alarm"),
            "adapt": (COLORS["adapt"], "dot", "Adaptation"),
            "recovery": (COLORS["recovery"], "dash", "Recovery"),
        }
        color, dash, label = colors[kind]

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
            opacity=0.72,
        )
    return fig


def base_fig(title, y_title, height=350):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15)),
        template="plotly_white",
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"]),
        height=height,
        margin=dict(l=48, r=24, t=55, b=42),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=10),
        ),
        xaxis=dict(title="Stream window", gridcolor=COLORS["grid"], zeroline=False),
        yaxis=dict(title=y_title, gridcolor=COLORS["grid"], zeroline=False),
    )
    return fig


def plot_best_model(d, height=350):
    """Plot only the lowest-MAE model at each stream window."""
    fig = base_fig("Best model — winner only", "MAE", height)
    if d.empty:
        return fig

    pivot = d.pivot_table(
        index="window", columns="strategy", values="mae", aggfunc="first"
    ).sort_index()

    winner = pivot.idxmin(axis=1, skipna=True)
    best_value = pivot.min(axis=1, skipna=True)

    for strategy in STRATEGIES:
        if strategy not in pivot.columns:
            continue
        y = best_value.where(winner == strategy)
        if y.notna().sum() == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=y,
                mode="lines+markers",
                connectgaps=False,
                name=STRATEGY_LABELS.get(strategy, strategy),
                line=dict(color=STRATEGY_COLORS.get(strategy, "#334155"), width=3),
                marker=dict(size=5),
                hovertemplate=(
                    "Window %{x}<br>Best model: "
                    + STRATEGY_LABELS.get(strategy, strategy)
                    + "<br>MAE: %{y:.3f}<extra></extra>"
                ),
            )
        )
    return fig


def plot_all_models(d, height=340):
    fig = base_fig("Model MAE comparison", "MAE", height)
    if d.empty:
        return fig
    for strategy in STRATEGIES:
        h = d[d["strategy"] == strategy]
        if h.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=h["window"], y=h["mae"], mode="lines+markers",
                name=STRATEGY_LABELS.get(strategy, strategy),
                line=dict(color=STRATEGY_COLORS.get(strategy, "#334155"), width=2.1),
                marker=dict(size=3),
            )
        )
    return fig


def plot_drift_signals(d, height=330):
    fig = base_fig("Drift detection signals", "Signal", height)
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
                    x=inc["window"], y=inc[col], mode="lines", name=label,
                    line=dict(color=color, width=2.1),
                )
            )
    return fig


def status_text(row):
    diagnosis = str(row.get("diagnosis", "none")).lower()
    if diagnosis == "both":
        return "BOTH"
    if diagnosis == "data":
        return "DATA"
    if diagnosis == "relational":
        return "RELATIONAL"
    return "NORMAL"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### Stream controls")
    delay = st.slider(
        "Seconds per window", 0.0, 1.0, 0.2, 0.1,
        help="Playback delay between live stream windows.",
    )

    start = st.button(
        "▶  Start / restart stream",
        type="primary",
        width="stretch",
    )

    st.divider()
    st.markdown("### Detection pipeline")
    st.markdown(
        "**Predict** → SGD regression\n\n"
        "**Detect** → KS / PSI + Page-Hinkley\n\n"
        "**Diagnose** → Data / Relational / Both\n\n"
        "**Adapt** → Incremental / Replay\n\n"
        "**Recover** → Post-drift state"
    )
    st.divider()
    st.caption(
        "Live detection does not read the synthetic drift-label columns. "
        "Ground truth is reserved for offline evaluation."
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
# TWO PAGES ONLY
# ============================================================

tab_dashboard, tab_analysis = st.tabs([
    "● Live Monitor",
    "◎ Evaluation & Analysis",
])


# ============================================================
# PAGE 1 — LIVE MONITOR
# ============================================================

with tab_dashboard:
    rows = current_rows()

    if not rows:
        st.info("Click **Start / restart stream** to begin the live monitor.")
    else:
        inc = rows.get("Incremental", next(iter(rows.values())))
        best_name, best_mae = current_best_model()

        drift_active = bool(
            inc.get("drift_active", status_text(inc) != "NORMAL")
        )
        diagnosis = status_text(inc)
        new_alarm = bool(inc.get("new_alarm", inc.get("alarm", False)))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CURRENT WINDOW", int(inc["window"]) + 1)
        c2.metric("BEST MODEL", best_name)
        c3.metric("BEST MAE", f"{best_mae:.3f}")
        c4.metric("DRIFT STATUS", "ACTIVE" if drift_active else "NORMAL")
        c5.metric("DIAGNOSIS", diagnosis)

        if new_alarm:
            st.markdown(
                '<span class="status-pill">⚠ NEW DRIFT ALARM</span>',
                unsafe_allow_html=True,
            )
        elif drift_active:
            st.markdown(
                '<span class="status-pill">● DRIFT EPISODE ACTIVE</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill">✓ SYSTEM NORMAL</span>',
                unsafe_allow_html=True,
            )

        all_history = history_df()

        st.markdown('<div class="section-title">Best model — winner only</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">Only the lowest-MAE model at each window is visible on the primary performance chart.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            add_event_lines(plot_best_model(all_history), st.session_state.events),
            width="stretch",
            key=f"best_model_{len(all_history)}",
        )

        left, right = st.columns([1.35, 1])
        with left:
            st.markdown('<div class="section-title">Drift detection</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-note">KS / PSI monitor feature distributions; Page-Hinkley monitors the X→Y relationship.</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                add_event_lines(plot_drift_signals(all_history), st.session_state.events),
                width="stretch",
                key=f"drift_{len(all_history)}",
            )

        with right:
            st.markdown('<div class="section-title">Current drift state</div>', unsafe_allow_html=True)
            data_active = bool(inc.get("data_active", False))
            relational_active = bool(inc.get("relational_active", False))
            a, b = st.columns(2)
            a.metric("DATA DRIFT", "ACTIVE" if data_active else "OFF")
            b.metric("RELATIONAL", "ACTIVE" if relational_active else "OFF")

            st.markdown("<br>", unsafe_allow_html=True)
            a, b, c = st.columns(3)
            inc_hist = strategy_history("Incremental")
            a.metric("ALARMS", int(inc_hist.get("new_alarm", inc_hist.get("alarm", pd.Series(dtype=bool))).astype(bool).sum()))
            b.metric("ADAPTATION", int(inc_hist.get("adapted", pd.Series(dtype=bool)).astype(bool).sum()))
            c.metric("RECOVERY", int(inc_hist.get("recovery", pd.Series(dtype=bool)).astype(bool).sum()))


# ============================================================
# PAGE 2 — EVALUATION & ANALYSIS
# ============================================================

with tab_analysis:
    st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Offline summary across the learning strategies.</div>',
        unsafe_allow_html=True,
    )
    comparison = read_optional_csv("model_comparison.csv")
    if comparison is not None and not comparison.empty:
        st.dataframe(comparison, width="stretch", hide_index=True)
    else:
        st.info("Run `run_experiment.py` to generate model comparison results.")

    st.markdown('<div class="section-title">Drift detector evaluation</div>', unsafe_allow_html=True)
    summary = read_optional_csv("detector_metrics_summary.csv")
    if summary is None or summary.empty:
        st.info("Run `run_experiment.py` to generate detector evaluation metrics.")
    else:
        st.caption(
            "Event-level metrics are calculated offline using the synthetic ground-truth drift episodes."
        )
        st.dataframe(summary, width="stretch", hide_index=True)

        by_seed = read_optional_csv("detector_metrics_by_seed.csv")
        if by_seed is not None and not by_seed.empty:
            with st.expander("Detector results by seed"):
                st.dataframe(by_seed, width="stretch", hide_index=True)

        drift_type = read_optional_csv("drift_type_summary.csv")
        if drift_type is None:
            drift_type = read_optional_csv("drift_type_evaluation.csv")
        if drift_type is not None and not drift_type.empty:
            with st.expander("Data / relational / both evaluation"):
                st.dataframe(drift_type, width="stretch", hide_index=True)

    st.markdown('<div class="section-title">Stream event log</div>', unsafe_allow_html=True)
    inc = strategy_history("Incremental")
    if inc.empty:
        st.info("Run the live stream to populate the event log.")
    else:
        log_cols = [
            c for c in [
                "window", "diagnosis", "drift_active", "data_active",
                "relational_active", "new_alarm", "adapted", "recovery",
                "worst_ks", "worst_psi", "ph_stat", "mean_resid",
                "features_moved",
            ] if c in inc.columns
        ]
        st.dataframe(inc[log_cols].tail(100), width="stretch", hide_index=True)


# ============================================================
# ONE-WINDOW LIVE EXECUTION
# ============================================================

if st.session_state.stream_running:
    runner = st.session_state.live_runner

    if runner is not None and st.session_state.stream_index < len(runner.windows):
        i = st.session_state.stream_index
        s, e = runner.windows[i]
        batch = runner.step(s, e)

        for strategy, row in batch.items():
            saved = dict(row)
            saved["strategy"] = strategy
            st.session_state.history.append(saved)

        primary = batch["Incremental"]

        if primary.get("new_alarm", primary.get("alarm", False)):
            st.session_state.events.append({
                "window": int(primary["window"]),
                "type": "alarm",
            })

        if primary.get("adapted", False):
            st.session_state.events.append({
                "window": int(primary["window"]),
                "type": "adapt",
            })

        if primary.get("recovery", False):
            st.session_state.events.append({
                "window": int(primary["window"]),
                "type": "recovery",
            })

        st.session_state.stream_index += 1

        if st.session_state.stream_index >= len(runner.windows):
            st.session_state.stream_running = False
            st.session_state.stream_finished = True
        else:
            time.sleep(max(0.05, float(delay)))

        st.rerun()
