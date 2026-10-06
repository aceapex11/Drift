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
        /* ---------- Global light enterprise theme ---------- */
        :root {
            --bg: #F5F7FB;
            --surface: #FFFFFF;
            --surface-2: #F8FAFD;
            --line: #E5EAF2;
            --text: #12213F;
            --muted: #667085;
            --blue: #2563EB;
            --blue-soft: #EAF2FF;
            --green: #0F9D72;
            --green-soft: #E8F8F2;
            --amber: #C77700;
            --amber-soft: #FFF5E6;
            --red: #D64545;
            --red-soft: #FDEEEE;
        }

        .stApp { background: var(--bg); color: var(--text); }
        .main .block-container { max-width: 1500px; padding: 1.2rem 2rem 3rem; }
        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * { color: var(--text); }
        [data-testid="stHeader"] { background: rgba(255,255,255,.94); border-bottom: 1px solid var(--line); }
        [data-testid="stToolbar"] { background: transparent; }

        /* Hide the default Streamlit decoration that makes the app feel like a demo. */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* ---------- Brand / hero ---------- */
        .brand-row { display:flex; align-items:center; gap:12px; margin:4px 0 22px; }
        .brand-mark {
            width:40px; height:40px; border-radius:11px; display:flex; align-items:center; justify-content:center;
            background:linear-gradient(135deg,#EAF2FF,#DCEAFF); border:1px solid #CFE0FF; color:#2563EB;
            font-size:18px; font-weight:800; box-shadow:0 4px 12px rgba(37,99,235,.08);
        }
        .brand-name { font-size:1.02rem; font-weight:780; letter-spacing:-.02em; color:#10254B; }
        .brand-caption { color:#7A879C; font-size:.70rem; margin-top:1px; }
        .hero {
            background:linear-gradient(105deg,#FFFFFF 0%,#F7FAFF 100%);
            border:1px solid var(--line); border-radius:18px; padding:22px 26px; margin-bottom:18px;
            box-shadow:0 6px 22px rgba(16,33,63,.035);
        }
        .hero-eyebrow { color:#2563EB; font-size:.70rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin-bottom:6px; }
        .hero-title { font-size:1.75rem; line-height:1.15; font-weight:800; letter-spacing:-.035em; color:#10254B; margin-bottom:6px; }
        .hero-subtitle { color:#667085; font-size:.88rem; line-height:1.5; }
        .status-pill {
            display:inline-flex; align-items:center; gap:7px; margin-top:12px; padding:5px 10px; border-radius:999px;
            background:#ECFDF3; border:1px solid #CDEEDC; color:#087A57; font-size:.70rem; font-weight:750;
        }
        .status-dot { width:7px; height:7px; border-radius:50%; background:#10B981; box-shadow:0 0 0 3px #D9F8EA; }

        /* ---------- Navigation ---------- */
        div[data-testid="stTabs"] { margin-top:0; }
        div[data-testid="stTabs"] button {
            font-weight:720; color:#667085; font-size:.82rem; padding:9px 14px;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] { color:#2563EB; }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#2563EB; height:2px; }

        /* ---------- KPI cards ---------- */
        [data-testid="stMetric"] {
            background:#FFFFFF; border:1px solid var(--line); border-radius:14px; padding:13px 15px;
            box-shadow:0 4px 14px rgba(16,33,63,.025); min-height:82px;
        }
        [data-testid="stMetricLabel"] { color:#7A879C; font-size:.68rem; font-weight:800; letter-spacing:.045em; text-transform:uppercase; }
        [data-testid="stMetricValue"] { color:#12213F; font-size:1.32rem; font-weight:790; letter-spacing:-.02em; }
        [data-testid="stMetricDelta"] { font-size:.70rem; }
        .section-title { color:#12213F; font-size:1.02rem; font-weight:780; letter-spacing:-.01em; margin:1.2rem 0 .18rem; }
        .section-note { color:#7A879C; font-size:.77rem; margin-bottom:.6rem; }
        .eyebrow { color:#2563EB; font-size:.67rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.3rem; }

        /* ---------- Analysis page ---------- */
        .analysis-head {
            display:flex; justify-content:space-between; align-items:flex-start; gap:20px;
            background:#FFFFFF; border:1px solid var(--line); border-radius:18px; padding:20px 22px;
            box-shadow:0 6px 22px rgba(16,33,63,.035); margin-bottom:16px;
        }
        .analysis-title { font-size:1.48rem; font-weight:800; letter-spacing:-.03em; color:#10254B; }
        .analysis-subtitle { color:#667085; font-size:.80rem; line-height:1.45; margin-top:4px; max-width:780px; }
        .head-meta { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
        .meta-chip { background:#F7F9FC; border:1px solid var(--line); color:#526174; border-radius:9px; padding:7px 10px; font-size:.68rem; font-weight:700; white-space:nowrap; }
        .meta-chip b { color:#12213F; }
        .kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:8px 0 14px; }
        .kpi-card { background:#FFFFFF; border:1px solid var(--line); border-radius:14px; padding:14px 15px; min-height:86px; box-shadow:0 4px 14px rgba(16,33,63,.025); }
        .kpi-card.good { background:linear-gradient(180deg,#FFFFFF,#F7FCFA); border-color:#D7EEE5; }
        .kpi-card.blue { background:linear-gradient(180deg,#FFFFFF,#F7FAFF); border-color:#DCE8FF; }
        .kpi-label { color:#7A879C; font-size:.66rem; font-weight:800; text-transform:uppercase; letter-spacing:.07em; }
        .kpi-value { color:#12213F; font-size:1.35rem; line-height:1.15; font-weight:800; margin-top:5px; letter-spacing:-.025em; }
        .kpi-help { color:#8A96A8; font-size:.68rem; margin-top:4px; }
        .panel { background:#FFFFFF; border:1px solid var(--line); border-radius:16px; padding:15px 16px; box-shadow:0 5px 18px rgba(16,33,63,.028); }
        .panel-title { color:#12213F; font-size:.92rem; font-weight:780; }
        .panel-sub { color:#7A879C; font-size:.70rem; margin-top:2px; margin-bottom:9px; }
        .method-card { background:#F8FAFD; border:1px solid var(--line); border-radius:12px; padding:12px 14px; color:#526174; font-size:.72rem; line-height:1.5; margin:10px 0 4px; }
        .method-card b { color:#12213F; }
        .quality-banner {
            display:flex; align-items:center; justify-content:space-between; gap:12px; padding:11px 14px;
            background:#F0FBF6; border:1px solid #D4EFE3; border-radius:12px; color:#0A7755; margin:10px 0 14px;
        }
        .quality-main { font-size:.78rem; font-weight:800; }
        .quality-sub { font-size:.68rem; color:#5C8778; margin-top:2px; }
        .quality-score { font-size:1.05rem; font-weight:850; }

        /* ---------- Tables ---------- */
        .pro-table { width:100%; border-collapse:separate; border-spacing:0; overflow:hidden; border:1px solid var(--line); border-radius:12px; background:#fff; font-size:.73rem; }
        .pro-table th { text-align:left; color:#718096; font-size:.65rem; font-weight:800; text-transform:uppercase; letter-spacing:.05em; background:#F8FAFD; padding:10px 12px; border-bottom:1px solid var(--line); }
        .pro-table td { padding:10px 12px; border-bottom:1px solid #EEF1F5; color:#344054; }
        .pro-table tr:last-child td { border-bottom:0; }
        .pro-table tr.best td { background:#F2F8FF; color:#10254B; font-weight:720; }
        .best-tag { display:inline-block; margin-left:6px; padding:2px 6px; border-radius:999px; background:#E8F1FF; color:#2563EB; font-size:.58rem; font-weight:800; }
        .metric-good { color:#087A57; font-weight:800; }
        .metric-neutral { color:#344054; }

        /* ---------- Buttons / expanders ---------- */
        .stButton > button { border-radius:10px; font-weight:750; border:1px solid #DCE3EE; }
        .stButton > button[kind="primary"] { background:#2563EB; border-color:#2563EB; }
        [data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:#FFFFFF; }
        [data-testid="stExpanderToggleIcon"] { color:#667085; }
        .stAlert { border-radius:12px; }

        @media (max-width: 900px) {
            .main .block-container { padding:1rem; }
            .analysis-head { flex-direction:column; }
            .head-meta { justify-content:flex-start; }
            .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="brand-row">
        <div class="brand-mark">↗</div>
        <div>
            <div class="brand-name">Live Drift Intelligence</div>
            <div class="brand-caption">Industrial ML monitoring & adaptive learning</div>
        </div>
    </div>
    <div class="hero">
        <div class="hero-eyebrow">Industrial predictive maintenance · research console</div>
        <div class="hero-title">Live Drift Intelligence</div>
        <div class="hero-subtitle">
            A streaming regression system for detecting distribution and relational drift, diagnosing the change,
            and adapting the learner without exposing ground-truth labels to the live detector.
        </div>
        <div class="status-pill"><span class="status-dot"></span> Live monitoring architecture ready</div>
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
    st.markdown(
        """
        <div style="padding:4px 2px 12px">
            <div style="font-size:.67rem;font-weight:800;letter-spacing:.11em;color:#2563EB;text-transform:uppercase">Research console</div>
            <div style="font-size:1.02rem;font-weight:800;color:#12213F;margin-top:4px">Stream controls</div>
            <div style="font-size:.70rem;color:#7A879C;margin-top:3px;line-height:1.4">Configure playback and observe the model lifecycle.</div>
        </div>
        """, unsafe_allow_html=True
    )
    delay = st.slider(
        "Playback speed", 0.0, 1.0, 0.2, 0.1,
        help="Controls the live playback speed.",
    )

    start = st.button(
        "Start / restart stream",
        type="primary",
        width="stretch",
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background:#F8FAFD;border:1px solid #E5EAF2;border-radius:12px;padding:12px 13px;margin-bottom:12px">
            <div style="font-size:.66rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#7A879C;margin-bottom:8px">Dataset</div>
            <div style="font-size:.76rem;color:#344054;line-height:1.75">
                <b>60,000</b> hourly observations<br>
                <b>8</b> monitored machine variables<br>
                <b>600 h</b> streaming window<br>
                <b>6</b> controlled drift episodes
            </div>
        </div>
        <div style="background:#F8FAFD;border:1px solid #E5EAF2;border-radius:12px;padding:12px 13px">
            <div style="font-size:.66rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#7A879C;margin-bottom:8px">Detection stack</div>
            <div style="font-size:.74rem;color:#344054;line-height:1.8">
                <b>01</b> Predict · SGDRegressor<br>
                <b>02</b> Detect · KS / PSI / Page-Hinkley<br>
                <b>03</b> Diagnose · Data / Relational / Both<br>
                <b>04</b> Adapt · Incremental / Replay<br>
                <b>05</b> Recover · post-drift monitoring
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """<div style="font-size:.66rem;color:#8A96A8;line-height:1.45;margin-top:14px">
        Ground-truth labels are deliberately excluded from live detection and adaptation. They are used only for offline evaluation.
        </div>""",
        unsafe_allow_html=True,
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

with tab_analysis:
    comparison = read_optional_csv("model_comparison.csv")
    summary = read_optional_csv("detector_metrics_summary.csv")

    st.markdown(
        """
        <div class="analysis-head">
            <div>
                <div class="eyebrow">Offline research results</div>
                <div class="analysis-title">Evaluation & Analysis</div>
                <div class="analysis-subtitle">
                    Model adaptation performance and event-level drift detection quality across the streaming experiment.
                    Ground-truth labels remain outside the live decision loop.
                </div>
            </div>
            <div class="head-meta">
                <div class="meta-chip"><b>Dataset</b> · 60,000 hourly rows</div>
                <div class="meta-chip"><b>Window</b> · 600 hours</div>
                <div class="meta-chip"><b>Scoring</b> · Event level</div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    # ---------------- Model evaluation ----------------
    st.markdown('<div class="section-title">Learning strategy performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Lower MAE / RMSE is better. Higher R² is better.</div>', unsafe_allow_html=True)

    if comparison is None or comparison.empty:
        st.info("Run `run_experiment.py` to generate model comparison results.")
    else:
        best_mae_row = comparison.loc[comparison["MAE"].idxmin()]
        best_rmse_row = comparison.loc[comparison["RMSE"].idxmin()]
        best_r2_row = comparison.loc[comparison["R2"].idxmax()]

        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card blue">
                    <div class="kpi-label">Best strategy</div>
                    <div class="kpi-value" style="font-size:1.02rem">{best_mae_row['Strategy']}</div>
                    <div class="kpi-help">Lowest mean MAE</div>
                </div>
                <div class="kpi-card good">
                    <div class="kpi-label">Best MAE</div>
                    <div class="kpi-value">{best_mae_row['MAE']:.3f}</div>
                    <div class="kpi-help">{best_mae_row['Strategy']}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Best RMSE</div>
                    <div class="kpi-value">{best_rmse_row['RMSE']:.3f}</div>
                    <div class="kpi-help">{best_rmse_row['Strategy']}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Best R²</div>
                    <div class="kpi-value">{best_r2_row['R2']:.3f}</div>
                    <div class="kpi-help">{best_r2_row['Strategy']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        rows_html = []
        for _, r in comparison.iterrows():
            is_best = r["Strategy"] == best_mae_row["Strategy"]
            cls = "best" if is_best else ""
            tag = '<span class="best-tag">BEST MAE</span>' if is_best else ''
            rows_html.append(
                f'<tr class="{cls}"><td>{r["Strategy"]}{tag}</td>'
                f'<td>{r["MAE"]:.3f}</td><td>{r["RMSE"]:.3f}</td>'
                f'<td>{r["R2"]:.3f}</td><td>{int(r["Windows"])}</td></tr>'
            )
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Strategy comparison</div>
                <div class="panel-sub">Aggregate performance across the evaluated stream windows.</div>
                <table class="pro-table">
                    <thead><tr><th>Strategy</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th><th>Windows</th></tr></thead>
                    <tbody>
            """ + "".join(rows_html) + """
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True
        )

    # ---------------- Detector evaluation ----------------
    st.markdown('<div class="section-title">Drift detector performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Event-level scoring treats each contiguous drift episode as one detection target.</div>', unsafe_allow_html=True)

    if summary is None or summary.empty:
        st.info("Run `run_experiment.py` to generate detector evaluation metrics.")
    else:
        sm = summary.set_index("Metric")["Mean"].to_dict()
        precision = float(sm.get("Precision", float("nan")))
        recall = float(sm.get("Recall", float("nan")))
        f1 = float(sm.get("F1", float("nan")))
        episodes = int(sm.get("True drift episodes", 0))
        matched = int(sm.get("Matched detections", 0))
        alarms = int(sm.get("Detected alarms", 0))
        fp = int(sm.get("FP", 0))
        delay = float(sm.get("Mean detection delay (windows)", float("nan")))

        quality = "Strong" if f1 >= .80 else ("Moderate" if f1 >= .60 else "Needs tuning")
        quality_cls = "good" if f1 >= .80 else "blue"

        st.markdown(
            f"""
            <div class="quality-banner">
                <div><div class="quality-main">Detector quality · {quality}</div>
                <div class="quality-sub">{matched}/{episodes} true drift episodes matched · {fp} unmatched alarms · mean delay {delay:.2f} windows</div></div>
                <div class="quality-score">F1 {f1:.3f}</div>
            </div>
            """, unsafe_allow_html=True
        )

        detector_cards = [
            ("Precision", f"{precision:.0%}", "Alarms correctly matched", "good"),
            ("Recall", f"{recall:.0%}", "True episodes detected", "good"),
            ("F1 score", f"{f1:.3f}", "Precision / recall balance", quality_cls),
            ("True episodes", str(episodes), "Offline ground truth", ""),
            ("Detected alarms", str(alarms), "Live detector outputs", ""),
            ("Mean delay", f"{delay:.2f}", "Stream windows", ""),
        ]
        cards_html = '<div class="kpi-grid" style="grid-template-columns:repeat(6,minmax(0,1fr));">'
        for label, value, help_text, cls in detector_cards:
            cards_html += f'<div class="kpi-card {cls}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-help">{help_text}</div></div>'
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown(
            f"""<div class="method-card"><b>Evaluation protocol.</b> An alarm is counted as a correct detection when it occurs inside a true drift episode or within the allowed <b>2-window detection delay</b>. The live detector never reads <code>regime</code>, <code>data_drift</code>, <code>concept_drift</code>, or <code>true_drift_point</code>. <b>{matched}/{episodes}</b> episodes were matched from <b>{alarms}</b> emitted alarms.</div>""",
            unsafe_allow_html=True,
        )

        by_seed = read_optional_csv("detector_metrics_by_seed.csv")
        window_metrics = read_optional_csv("detector_window_metrics_by_seed.csv")

        if by_seed is not None and not by_seed.empty:
            with st.expander("Technical audit · detector results by seed"):
                cols = [c for c in ["Seed", "Precision", "Recall", "F1", "TP", "FP", "FN", "True drift episodes", "Detected alarms", "Matched detections", "Mean detection delay (windows)"] if c in by_seed.columns]
                st.dataframe(by_seed[cols], width="stretch", hide_index=True)

        if window_metrics is not None and not window_metrics.empty:
            with st.expander("Technical audit · legacy window-level metrics"):
                st.caption("Diagnostic only. These metrics are retained for auditability and are not the headline event-level detector score.")
                st.dataframe(window_metrics, width="stretch", hide_index=True)

    # ---------------- Recent evidence ----------------
    st.markdown('<div class="section-title">Recent live detector evidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Latest detector outputs from the live stream. Ground truth is intentionally absent.</div>', unsafe_allow_html=True)
    inc = strategy_history("Incremental")
    if inc.empty:
        st.info("Run the live stream to populate detector evidence.")
    else:
        log_cols = [c for c in ["window", "diagnosis", "data_alarm", "relational_alarm", "alarm", "adapted", "recovery", "worst_ks", "worst_psi", "ph_stat", "features_moved"] if c in inc.columns]
        recent = inc[log_cols].tail(10).copy()
        recent["window"] = recent["window"].astype(int) + 1
        st.dataframe(recent, width="stretch", hide_index=True)
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
