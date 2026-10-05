"""
Streamlit dashboard.   Run:  streamlit run app.py
Two sources:  (1) Embedded engine  - everything in this one process (use this on Streamlit Community Cloud)
              (2) FastAPI server   - connects to  uvicorn stream_server:app  over a WebSocket
"""
import json, time
import pandas as pd
import streamlit as st
from data_generator import generate, TRAIN_END, WINDOW, TARGET
from drift_engine import DualRunner, TransferGBR, make_windows

st.set_page_config(page_title="Live drift detection", layout="wide")
st.title("Live drift detection + transfer-learning adaptation")
st.caption("KS / PSI -> data drift   |   Page-Hinkley -> relational drift   |   gradient-boosting transfer learning adapts the model")

@st.cache_resource(show_spinner="Generating data and pre-training the source model (one-off, ~10 s)...")
def load_assets():
    df = generate()
    from drift_engine import StreamEngine
    pretrained = StreamEngine(df, TRAIN_END, WINDOW, TransferGBR, 0).model
    return df, pretrained

def embedded_stream(delay):
    df, pretrained = load_assets()
    runner = DualRunner(df, TRAIN_END, WINDOW, pretrained)
    yield {"type": "ready", "ks_thr": runner.ks_thr, "reference_mae": runner.reference_mae, "n_windows": len(make_windows(len(df), TRAIN_END, WINDOW))}
    for s, e in make_windows(len(df), TRAIN_END, WINDOW):
        out = runner.step(s, e); out["type"] = "window"; out["truth_regime"] = df.regime.iloc[s]
        yield out; time.sleep(delay)
    yield {"type": "done"}

def server_stream(url, delay):
    from websockets.sync.client import connect          # pip install websockets
    with connect(f"{url}?delay={delay}", max_size=None) as ws:
        for msg in ws:
            m = json.loads(msg); yield m
            if m["type"] == "done": return

with st.sidebar:
    source = st.radio("Data source", ["Embedded engine", "FastAPI server"])
    url = st.text_input("WebSocket URL", "ws://localhost:8000/ws") if source == "FastAPI server" else None
    delay = st.slider("Seconds per window", 0.0, 2.0, 0.3, 0.1)
    start = st.button("Start live stream", type="primary")
    show_truth = st.checkbox("Shade true regimes (answer key, demo only)", True)

c = st.columns(5); box = [x.empty() for x in c]
chart_mae, chart_sig, table = st.empty(), st.empty(), st.empty()
left, right = st.columns(2); ks_box, log_box = left.empty(), right.empty()

if start:
    rows, alarms = [], []
    stream = embedded_stream(delay) if source == "Embedded engine" else server_stream(url, delay)
    for msg in stream:
        if msg["type"] == "ready":
            ks_thr = msg["ks_thr"]; continue
        if msg["type"] == "done":
            st.success("Stream finished."); break
        rows.append(msg)
        d = pd.DataFrame(rows).set_index("window")
        box[0].metric("Window", f'{msg["window"] + 1}')
        box[1].metric("Diagnosis", msg["diagnosis"].upper())
        box[2].metric("Static model MAE", f'{msg["mae_static"]:.2f}')
        box[3].metric("Transfer model MAE", f'{msg["mae"]:.2f}', f'{msg["mae"] - msg["mae_static"]:+.2f} vs static', delta_color="inverse")
        box[4].metric("Model trees", msg.get("n_trees", "-"))
        plot = d[["mae_static", "mae"]].rename(columns={"mae_static": "Static (no adaptation)", "mae": "Transfer learning"})
        chart_mae.line_chart(plot, height=260)
        chart_sig.line_chart(d[["worst_ks", "mean_resid"]].rename(columns={"worst_ks": "worst KS (data)", "mean_resid": "old-rule residual x normal (relational)"}), height=220)
        ks_box.bar_chart(pd.Series(msg["ks"], name="KS per sensor"), height=220)
        if msg["alarm"] or msg["recovery"]:
            alarms.append({"window": msg["window"], "event": "DRIFT ALARM" if msg["alarm"] else "drift ended -> re-adapt",
                           "diagnosis": msg["diagnosis"], "sensors moved": ", ".join(msg["features_moved"]),
                           **({"answer key": msg["truth_regime"]} if show_truth else {})})
        if alarms: log_box.dataframe(pd.DataFrame(alarms), hide_index=True, use_container_width=True)
else:
    st.info("Press **Start live stream** in the sidebar.")
