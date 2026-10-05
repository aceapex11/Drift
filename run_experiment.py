"""Offline experiment: detector quality + Static vs Retrain vs Transfer learning on the 60k-row stream."""
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
