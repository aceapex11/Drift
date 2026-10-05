
Loading older messages…
my ohtrer o,d app ran this didint run
Yes, abhi. If your other old app.py runs but the one I gave you does not, then we should not keep guessing.

Most likely the newer app.py I gave you depends on something that your current data_generator.py / drift_engine.py does not expose exactly as expected.

Send me the error shown in the terminal/Streamlit screen (screenshot is fine). I’ll fix that exact issue without changing your working backend.

If you want, I can also take your working old app.py + current data_generator.py + drift_engine.py and modify only the dashboard UI so it definitely remains compatible.

thje what to do what shoudl i e du s u could check ?
Yes, abhi — I can check it for you. We should not keep changing code blindly.

Please upload these 3 files from the version that actually works:

app.py — the old/working one
data_generator.py
drift_engine.py
And if you have it:

requirements.txt
Then I’ll compare them and give you one corrected app.py that:

keeps your existing working backend untouched
uses the same function/class interfaces
keeps the dashboard improvements
removes whatever is causing the new app to fail
is syntax-checked before I give it to you
is ready to copy-paste directly into GitHub
If you already have an error message, upload a screenshot of the error too. That will make the fix much faster.


Pasted text(20261005-101155).txt
Document

Pasted text (2)(4).txt
Document
numpy
pandas
scipy
scikit-learn
matplotlib
streamlit
fastapi
uvicorn[standard]
websockets
plotly"""
Synthetic industrial-machine stream for drift research.

60,000 hourly rows (~6.8 years), 8 sensors, 11 labelled regimes covering every drift family:
sudden / gradual, data / relational / both, a recurring concept, plus a harmless sensor glitch
that a good detector should IGNORE.  Same column names as the original notebooks, so it is a
drop-in replacement for drift_regression_dataset.csv.

Run:  python data_generator.py   ->  data/drift_regression_dataset_large.csv
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

FEATURES = ["temperature", "vibration", "pressure", "load",
            "operating_hours", "humidity", "rpm", "oil_viscosity"]
# operating_hours is a maintenance counter (saw-tooth), so it is excluded from distribution monitoring
MONITORED = [f for f in FEATURES if f != "operating_hours"]
TARGET = "health_deterioration"
N_ROWS = 60_000
TRAIN_END = 14_400            # rows before this = normal training history
WINDOW = 600                  # the stream arrives 600 hours at a time

NOM_MEAN = dict(temperature=70, vibration=2.0, pressure=30, load=60, operating_hours=2000,
                humidity=45, rpm=1500, oil_viscosity=40)
NOM_SD = dict(temperature=6, vibration=0.5, pressure=3, load=12, operating_hours=1155,
              humidity=10, rpm=120, oil_viscosity=4)

# target = bias + linear + 3 interaction/curvature terms (+ small overheating knee)
#          [bias, temp, vib, press, load, hours, hum, rpm, oil, temp*load, vib^2, vib*press]
W_BASE = np.array([38, 7, 10, 5, 8, 7, 1, 3, -4, 3.5, 3.0, 0.0])
W_R1 = np.array([38, 7, 20, 5, -3, 7, 1, 3, -4, 3.5, 3.0, 7.0])    # vibration x2, load effect flips, new vib*press term
W_R2 = np.array([46, 2, 10, 9, 8, 14, 1, -5, -4, -3.5, 3.0, 0.0])  # wear dominated by hours, temp weak, interaction flips

# sensor shifts in units of the normal std: (mean shift, scale multiplier)
SHIFT_A = dict(temperature=(2.0, 1.0), vibration=(1.2, 1.3), rpm=(1.0, 1.0))
SHIFT_B = dict(load=(1.8, 1.0), temperature=(0.0, 1.6), humidity=(1.5, 1.0), pressure=(-1.0, 1.0))
SHIFT_C = dict(temperature=(1.5, 1.0), load=(1.2, 1.0), oil_viscosity=(-1.5, 1.2), vibration=(0.8, 1.0))

# start, end, regime, mode, data shift, relation
EVENTS = [
    (18_000, 21_000, "data_drift",       "sudden",  SHIFT_A, W_BASE),
    (25_200, 29_400, "relational_drift", "sudden",  None,    W_R1),
    (32_400, 37_200, "both",             "gradual", SHIFT_C, W_R2),   # ramps in over the first 2,400 rows
    (39_600, 42_600, "relational_drift", "sudden",  None,    W_R1),   # RECURRING concept (same as event 2)
    (45_000, 51_000, "data_drift",       "sudden",  SHIFT_B, W_BASE),
    (52_800, 60_000, "both",             "sudden",  SHIFT_A, W_R2),
]
GRADUAL_RAMP = 2_400
GLITCH = (30_600, 30_612)     # 12 rows of sensor spikes in a stable period: NOT a drift


def _ar1(n, phi, rng):
    return lfilter([1.0], [1.0, -phi], rng.normal(size=n)) * np.sqrt(1 - phi ** 2)


def generate(n=N_ROWS, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    day = np.sin(2 * np.pi * t / 24)
    year = np.sin(2 * np.pi * t / 8760)

    a_data = np.zeros(n)                 # how far the sensor distribution has moved (0..1)
    W = np.tile(W_BASE, (n, 1))          # relationship weights, row by row
    regime = np.array(["stable"] * n, dtype=object)
    data_flag, concept_flag = np.zeros(n, int), np.zeros(n, int)
    shifts = [None] * n
    drift_point = np.full(n, np.nan)
    for s, e, name, mode, shift, rel in EVENTS:
        ramp = np.clip((t[s:e] - s) / GRADUAL_RAMP, 0, 1) if mode == "gradual" else np.ones(e - s)
        regime[s:e] = name
        drift_point[s] = 1.0
        if shift is not None:
            a_data[s:e] = ramp; data_flag[s:e] = 1
            for i in range(s, e): shifts[i] = shift
        if name in ("relational_drift", "both"):
            W[s:e] = (1 - ramp)[:, None] * W_BASE + ramp[:, None] * rel
            concept_flag[s:e] = 1

    raw = {}
    z = {  # standardised "normal" signals before any drift
        "temperature": _ar1(n, 0.92, rng) * 0.8 + 0.4 * day + 0.2 * year,
        "vibration": _ar1(n, 0.90, rng) + 0.15 * day,
        "pressure": _ar1(n, 0.93, rng) - 0.1 * year,
        "load": _ar1(n, 0.88, rng) * 0.8 + 0.5 * day,
        "humidity": _ar1(n, 0.94, rng) * 0.9 - 0.2 * year,
        "rpm": _ar1(n, 0.85, rng) * 0.9 + 0.2 * rng.normal(size=n),
        "oil_viscosity": _ar1(n, 0.95, rng) * 0.9,
    }
    # slow shared factor: hot + heavy load tends to go with high vibration (realistic correlation)
    z["vibration"] += 0.35 * (z["temperature"] + z["load"]) / 2
    for f, v in z.items():
        mu_shift, sc = np.zeros(n), np.ones(n)
        for i in np.flatnonzero(a_data > 0):
            m, s_ = shifts[i].get(f, (0.0, 1.0)); mu_shift[i] = a_data[i] * m; sc[i] = 1 + a_data[i] * (s_ - 1)
        raw[f] = NOM_MEAN[f] + NOM_SD[f] * (v * sc + mu_shift)
    raw["operating_hours"] = (t % 4000).astype(float)               # maintenance reset every 4,000 h

    # sensor glitch (spikes only in readings; the machine itself is fine)
    g0, g1 = GLITCH
    raw["temperature"][g0:g1] += 6 * NOM_SD["temperature"]
    raw["vibration"][g0:g1] += 5 * NOM_SD["vibration"]

    u = {f: (raw[f] - NOM_MEAN[f]) / NOM_SD[f] for f in FEATURES}
    u_clean = {f: u[f].copy() for f in FEATURES}                      # the machine's true state ignores the glitch
    u_clean["temperature"][g0:g1] = ((raw["temperature"][g0:g1] - 6 * NOM_SD["temperature"]) - NOM_MEAN["temperature"]) / NOM_SD["temperature"]
    u_clean["vibration"][g0:g1] = ((raw["vibration"][g0:g1] - 5 * NOM_SD["vibration"]) - NOM_MEAN["vibration"]) / NOM_SD["vibration"]
    terms = np.column_stack([np.ones(n), u_clean["temperature"], u_clean["vibration"], u_clean["pressure"], u_clean["load"],
                             u_clean["operating_hours"], u_clean["humidity"], u_clean["rpm"], u_clean["oil_viscosity"],
                             u_clean["temperature"] * u_clean["load"], u_clean["vibration"] ** 2,
                             u_clean["vibration"] * u_clean["pressure"]])
    knee = 2.5 / (1 + np.exp(-3 * (u_clean["temperature"] - 1.2)))   # overheating knee (not polynomial)
    y = (W * terms).sum(1) + knee + rng.normal(0, 2.5, n)
    y = np.clip(y, 0, 100)

    df = pd.DataFrame({"timestamp": pd.date_range("2018-01-01", periods=n, freq="h")})
    for f in FEATURES: df[f] = raw[f]
    df[TARGET] = y
    df["regime"], df["data_drift"], df["concept_drift"], df["true_drift_point"] = regime, data_flag, concept_flag, drift_point
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    d = generate()
    d.to_csv("data/drift_regression_dataset_large.csv", index=False)
    print(d.shape); print(d.groupby("regime", sort=False)[TARGET].agg(["size", "mean", "std"]).round(2))numpy
pandas
scipy
scikit-learn
matplotlib
streamlit
fastapi
uvicorn[standard]
websockets
plotly"""Offline experiment: detector quality + Static vs Retrain vs Transfer learning on the 60k-row stream."""
import time, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_generator import generate, TRAIN_END, WINDOW, TARGET
from drift_engine import StreamEngine, TransferGBR, RetrainGBR, StaticGBR, make_windows

df = generate()
N = len(df); wins = make_windows(N, TRAIN_END, WINDOW)
regime = [df.regime.iloc[s] for s, e in wins]
true_pts = df.index[df.true_drift_point.notna()].tolist()

def run(cls, seed=0):
    eng = StreamEngine(df, TRAIN_END, WINDOW, cls, seed); t0 = time.time()
    rows = [eng.step(s, e) for s, e in wins]
    return pd.DataFrame(rows), time.time() - t0, eng

res, secs = {}, {}
for name, cls in [("Static (no adaptation)", StaticGBR), ("Full retrain (recent buffer)", RetrainGBR), ("Transfer learning (fine-tune)", TransferGBR)]:
    res[name], secs[name], eng = run(cls); print(f"{name}: {secs[name]:.0f}s", flush=True)

det = res["Transfer learning (fine-tune)"]       # detector output is identical for all (it never looks at the model)
# ---- detector evaluation ----
ev = []
for i, d in enumerate(true_pts):
    w0 = next(k for k, (s, e) in enumerate(wins) if s <= d < e); w1 = next((next(k for k, (s, e) in enumerate(wins) if s <= true_pts[i+1] < e) for _ in [0] if i+1 < len(true_pts)), len(wins))
    hits = det[(det.window >= w0) & (det.window < w1) & det.alarm]
    t = {(1,0):"data", (0,1):"relational", (1,1):"both"}[(int(df.data_drift.iloc[d]), int(df.concept_drift.iloc[d]))]
    ev.append(dict(event=i+1, true_type=t, mode="gradual" if i == 2 else "sudden", true_window=w0,
                   alarm_window=int(hits.window.iloc[0]) if len(hits) else None,
                   delay_windows=int(hits.window.iloc[0]) - w0 if len(hits) else None,
                   diagnosed_at_alarm=hits.diagnosis.iloc[0] if len(hits) else "MISSED",
                   n_alarms_in_event=len(hits)))
ev = pd.DataFrame(ev); print(ev.to_string(index=False))
stable_alarm_windows = [int(k) for k in det.window[det.alarm] if regime[k] == "stable" and not any(0 <= k - next(j for j,(s,e) in enumerate(wins) if s <= d < e) <= 1 for d in true_pts)]
print("alarms in stable periods (false alarms):", stable_alarm_windows)

# ---- model comparison ----
tab = pd.DataFrame({n: {r: r_.loc[[k for k, w in enumerate(regime) if w == r], "mae"].mean() for r in ["stable", "data_drift", "relational_drift", "both"]} for n, r_ in res.items()}).T
tab["overall"] = [r_["mae"].mean() for r_ in res.values()]; tab["train+predict seconds"] = list(secs.values())
print(tab.round(2).to_string())
ev.to_csv("results/detector_events.csv", index=False); tab.round(3).to_csv("results/model_comparison.csv")
det.drop(columns=["ks", "psi"]).to_csv("results/stream_log.csv", index=False)

# ---- plot ----
shade = {"stable": "#ffffff", "data_drift": "#fff2cc", "relational_drift": "#dae8fc", "both": "#f8cecc"}
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
for a in ax:
    for k, w in enumerate(regime): a.axvspan(k - .5, k + .5, color=shade[w], lw=0)
for (n, r_), c in zip(res.items(), ["#444", "#1f77b4", "#d62728"]): ax[0].plot(r_.window, r_.mae, label=n, color=c, lw=1.6)
ax[0].set_ylabel("MAE per batch (lower = better)"); ax[0].legend(); ax[0].set_title("Prediction error as the machine changes (yellow = data, blue = relational, red = both)")
ax[1].plot(det.window, det.worst_ks, color="teal", label="worst KS (data)"); ax[1].axhline(0.2, color="teal", ls=":")
ax[1].plot(det.window, det.mean_resid, color="purple", label="mean old-rule residual (relational)")
for k in det.window[det.alarm]: ax[1].axvline(k, color="orange", lw=2)
ax[1].set_xlabel("stream window (600 h)"); ax[1].legend(loc="upper left"); ax[1].set_title("Detector signals (orange = alarm)")
plt.tight_layout(); plt.savefig("results/experiment.png", dpi=110)
…
Yes, abhi. I checked the code you provided against your data_generator.py and drift_engine.py. I found the problem — you don't need to change your data generator or drift engine.

Your backend interfaces actually match the dashboard: DualRunner, StreamEngine, TransferGBR, and make_windows are all present. 

The main problem
In the dashboard around the Detector Evaluation section, you have:

f"Worst KS: {float(d["worst_ks"].iloc[-1]):.3f} · "
f"Relational signal: {float(d["mean_resid"].iloc[-1]):.3f}"

Those double quotes inside the f-string can cause a Python syntax error, especially on the Python version commonly used with Streamlit.

Change it to:

detector_slot.caption(
    f"Observed alarms: {alarm_count} · Recovery events: {recovery_count} · "
    f"Worst KS: {float(d['worst_ks'].iloc[-1]):.3f} · "
    f"Relational signal: {float(d['mean_resid'].iloc[-1]):.3f}"
)

There is also a second likely compatibility issue
Your dashboard uses:

width="stretch"

in many Streamlit calls.

If your other app uses an older Streamlit version, this can fail. Your requirements simply say streamlit without pinning a version, so the installed version can differ between environments.

For maximum compatibility, replace:

width="stretch"

with:

use_container_width=True

for calls such as:

st.plotly_chart(...)
st.dataframe(...)

Your backend is fine
Your drift_engine.py is designed exactly around:

Data drift       → KS / PSI
Relational drift → Page-Hinkley
Prediction       → Gradient Boosting
Adaptation       → Transfer learning
and DualRunner returns the live transfer model result plus static-model MAE. 
 

So do NOT modify data_generator.py or drift_engine.py just to make the dashboard run.

What I recommend
Don't try random fixes one by one.

I should make you a corrected app.py based exactly on these files, keeping your backend unchanged.

I'll fix:

the f-string syntax problem
Streamlit version compatibility
the model_comparison.csv path issue
any other dashboard/backend mismatches
keep all 6 tabs
keep the live streaming
keep your KS/PSI + Page-Hinkley logic
keep ground truth strictly for visualization/evaluation
keep Transfer Learning vs Static comparison
Then you can simply replace your current app.py.

You do not need to send anything else right now — I have enough to fix it.

ok do plz

from pathlib import Path

# Build a corrected, self-contained app.py compatible with the user's supplied
# data_generator.py and drift_engine.py interfaces.
app = r'''"""
Streamlit dashboard for the live drift-detection project.

Run:
    streamlit run app.py

Backend files are intentionally unchanged:
    data_generator.py
    drift_engine.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_generator import generate, TRAIN_END, WINDOW
from drift_engine import DualRunner, StreamEngine, TransferGBR, make_windows


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
    "transfer": "#16A34A",
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
    "Transfer-learning adaptation"
)


# ============================================================
# EXISTING MODEL / DATA PIPELINE
# ============================================================

@st.cache_resource(show_spinner="Generating data and pre-training the source model…")
def load_assets():
    df = generate()
    pretrained = StreamEngine(
        df, TRAIN_END, WINDOW, TransferGBR, 0
    ).model
    return df, pretrained


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
        Transfer-learning model

        **2. Detect**  
        KS / PSI + Page-Hinkley

        **3. Diagnose**  
        DATA / RELATIONAL / BOTH

        **4. Adapt**  
        Transfer-learning update

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
        "TRANSFER MAE",
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
            name="Transfer learning",
            line=dict(color=COLORS["transfer"], width=3),
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
                line=dict(color=COLORS["transfer"], width=2.5),
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
            name="Transfer learning",
            line=dict(color=COLORS["transfer"], width=3),
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
                use
ChatGPT is responding



