"""
Streamlit dashboard.   Run:  streamlit run app.py
Everything (data, detectors, transfer-learning model) runs inside this one process.
"""
import time
import pandas as pd
import streamlit as st
from data_generator import generate, TRAIN_END, WINDOW
from drift_engine import DualRunner, StreamEngine, TransferGBR, make_windows

st.set_page_config(page_title="Live drift detection", layout="wide")
st.title("Live drift detection + transfer-learning adaptation")
st.caption("KS / PSI -> data drift   |   Page-Hinkley -> relational drift   |   gradient-boosting transfer learning adapts the model")


@st.cache_resource(show_spinner="Generating data and pre-training the source model (one-off, ~10 s)...")
def load_assets():
    df = generate()
    pretrained = StreamEngine(df, TRAIN_END, WINDOW, TransferGBR, 0).model
    return df, pretrained


with st.sidebar:
    delay = st.slider("Seconds per window", 0.0, 2.0, 0.3, 0.1)
    start = st.button("Start live stream", type="primary")
    show_truth = st.checkbox("Show true regime in alarm log (answer key, demo only)", True)

cols = st.columns(5)
box = [x.empty() for x in cols]
chart_mae, chart_sig = st.empty(), st.empty()
left, right = st.columns(2)
ks_box, log_box = left.empty(), right.empty()

if start:
    df, pretrained = load_assets()
    runner = DualRunner(df, TRAIN_END, WINDOW, pretrained)
    rows, alarms = [], []

    for s, e in make_windows(len(df), TRAIN_END, WINDOW):
        msg = runner.step(s, e)
        msg["truth_regime"] = df.regime.iloc[s]      # answer key, only used for the demo log
        rows.append(msg)
        d = pd.DataFrame(rows).set_index("window")

        box[0].metric("Window", f'{msg["window"] + 1}')
        box[1].metric("Diagnosis", msg["diagnosis"].upper())
        box[2].metric("Static model MAE", f'{msg["mae_static"]:.2f}')
        box[3].metric("Transfer model MAE", f'{msg["mae"]:.2f}',
                      f'{msg["mae"] - msg["mae_static"]:+.2f} vs static', delta_color="inverse")
        box[4].metric("Model trees", msg.get("n_trees", "-"))

        chart_mae.line_chart(
            d[["mae_static", "mae"]].rename(columns={"mae_static": "Static (no adaptation)", "mae": "Transfer learning"}),
            height=260)
        chart_sig.line_chart(
            d[["worst_ks", "mean_resid"]].rename(columns={
                "worst_ks": "worst KS (data)",
                "mean_resid": "old-rule residual x normal (relational)"}),
            height=220)
        ks_box.bar_chart(pd.Series(msg["ks"], name="KS per sensor"), height=220)

        if msg["alarm"] or msg["recovery"]:
            alarms.append({
                "window": msg["window"],
                "event": "DRIFT ALARM" if msg["alarm"] else "drift ended -> re-adapt",
                "diagnosis": msg["diagnosis"],
                "sensors moved": ", ".join(msg["features_moved"]),
                **({"answer key": msg["truth_regime"]} if show_truth else {}),
            })
        if alarms:
            log_box.dataframe(pd.DataFrame(alarms), hide_index=True)
        time.sleep(delay)

    st.success("Stream finished.")
else:
    st.info("Press **Start live stream** in the sidebar.")
