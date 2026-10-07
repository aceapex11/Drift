"""Offline ML + drift experiment runner.

Outputs:
- model_comparison.csv
- stream_log.csv
- detector_events.csv
- detector_metrics_by_seed.csv
- detector_metrics_summary.csv
- detector_window_metrics_by_seed.csv
- drift_type_evaluation.csv
- drift_type_summary.csv

The live detector never uses ground-truth drift columns. They are used only
here, after the stream run, for offline evaluation.
"""

from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

from data_generator import TRAIN_END, WINDOW, generate
from drift_engine import STRATEGY_NAMES, StrategyRunner

SEEDS = [0, 1, 2, 3, 4]
MAX_DELAY_WINDOWS = 2


def window_truth(df, start, end):
    data_truth = bool(df.loc[start:end - 1, "data_drift"].astype(bool).any())
    concept_truth = bool(df.loc[start:end - 1, "concept_drift"].astype(bool).any())
    return data_truth, concept_truth


def true_drift_episodes(rows, df):
    flags = [
        bool(window_truth(df, int(r["start"]), int(r["end"]))[0]
              or window_truth(df, int(r["start"]), int(r["end"]))[1])
        for r in rows
    ]
    episodes = []
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flags) and flags[j + 1]:
            j += 1
        episodes.append((i, j))
        i = j + 1
    return episodes


def true_episode_type(rows, df, start_idx, end_idx):
    has_data = False
    has_concept = False
    for i in range(start_idx, end_idx + 1):
        d, c = window_truth(df, int(rows[i]["start"]), int(rows[i]["end"]))
        has_data |= d
        has_concept |= c
    if has_data and has_concept:
        return "both"
    if has_data:
        return "data"
    if has_concept:
        return "relational"
    return "none"


def evaluate_detector(rows, df, max_delay_windows=2):
    episodes = true_drift_episodes(rows, df)
    alarms = [i for i, r in enumerate(rows) if bool(r["alarm"])]
    matched = set()
    delays = []
    for start, end in episodes:
        candidates = [
            a for a in alarms
            if a not in matched and start <= a <= end + max_delay_windows
        ]
        if candidates:
            a = min(candidates)
            matched.add(a)
            delays.append(max(0, a - start))

    tp = len(matched)
    fn = len(episodes) - tp
    fp = len([a for a in alarms if a not in matched])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "Accuracy": np.nan,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": np.nan,
        "True drift episodes": len(episodes),
        "Detected alarms": len(alarms),
        "Matched detections": tp,
        "Mean detection delay (windows)": float(np.mean(delays)) if delays else np.nan,
        "Max acceptable delay (windows)": max_delay_windows,
    }


def evaluate_detector_windowwise(rows, df):
    y_true, y_pred = [], []
    for r in rows:
        d, c = window_truth(df, int(r["start"]), int(r["end"]))
        y_true.append(int(d or c))
        y_pred.append(int(bool(r["alarm"])))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def evaluate_drift_types(rows, df, max_delay_windows=2):
    episodes = true_drift_episodes(rows, df)
    out = []
    for ep_no, (start, end) in enumerate(episodes, start=1):
        true_type = true_episode_type(rows, df, start, end)
        end_plus = min(len(rows) - 1, end + max_delay_windows)
        observed = set()
        alarm_window = None
        for i in range(start, end_plus + 1):
            r = rows[i]
            if bool(r.get("alarm", False)) and alarm_window is None:
                alarm_window = i
            if bool(r.get("data_active", False)):
                observed.add("data")
            if bool(r.get("relational_active", False)):
                observed.add("relational")
        if observed == {"data", "relational"}:
            detected_type = "both"
        elif observed == {"data"}:
            detected_type = "data"
        elif observed == {"relational"}:
            detected_type = "relational"
        else:
            detected_type = "none"
        out.append({
            "episode": ep_no,
            "true_type": true_type,
            "detected_type": detected_type,
            "type_correct": detected_type == true_type,
            "alarm_detected": alarm_window is not None,
            "alarm_window": alarm_window,
            "episode_start_window": start,
            "episode_end_window": end,
        })
    return out


def run_seed(seed):
    # Each process creates its own deterministic copy of the same stream.
    df = generate()
    runner = StrategyRunner(df, TRAIN_END, WINDOW, seed=seed)
    records = runner.run()
    detector_rows = [r for r in records if r["strategy"] == "Incremental"]
    return {
        "seed": seed,
        "records": records,
        "detector": detector_rows,
        "detector_metric": evaluate_detector(detector_rows, df, MAX_DELAY_WINDOWS),
        "window_metric": evaluate_detector_windowwise(detector_rows, df),
        "type_results": evaluate_drift_types(detector_rows, df, MAX_DELAY_WINDOWS),
    }


def main():
    # Limit BLAS/OpenMP oversubscription when several seeds run in parallel.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    print("Running 5 deterministic seeds in parallel...")
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        results = list(pool.map(run_seed, SEEDS))

    results.sort(key=lambda x: x["seed"])

    all_model_rows = []
    detector_rows_all = []
    window_rows_all = []
    type_rows_all = []

    for result in results:
        seed = result["seed"]
        for row in result["records"]:
            row = dict(row)
            row["seed"] = seed
            all_model_rows.append(row)

        dm = dict(result["detector_metric"])
        dm["Seed"] = seed
        detector_rows_all.append(dm)

        wm = dict(result["window_metric"])
        wm["Seed"] = seed
        window_rows_all.append(wm)

        for item in result["type_results"]:
            item = dict(item)
            item["Seed"] = seed
            type_rows_all.append(item)

    model_df = pd.DataFrame(all_model_rows)
    comparison = (
        model_df.groupby("strategy", as_index=False)
        .agg(MAE=("mae", "mean"), RMSE=("rmse", "mean"), R2=("r2", "mean"), Windows=("window", "count"))
        .rename(columns={"strategy": "Strategy"})
    )
    order = {name: i for i, name in enumerate(STRATEGY_NAMES)}
    comparison["_order"] = comparison["Strategy"].map(order)
    comparison = comparison.sort_values("_order").drop(columns="_order")
    comparison.to_csv("model_comparison.csv", index=False)

    # Seed 0 drives the dashboard's detailed stream log.
    seed0 = next(r for r in results if r["seed"] == 0)
    stream_df = pd.DataFrame(seed0["detector"])
    stream_df.to_csv("stream_log.csv", index=False)

    event_rows = []
    df0 = generate()
    for row in seed0["detector"]:
        d, c = window_truth(df0, int(row["start"]), int(row["end"]))
        if row["alarm"] or row.get("data_alarm") or row.get("relational_alarm") or row.get("recovery"):
            if row["alarm"]:
                event = "detected_drift"
            elif row.get("data_alarm"):
                event = "data_drift_detected"
            elif row.get("relational_alarm"):
                event = "relational_drift_detected"
            else:
                event = "recovery"
            event_rows.append({
                "window": row["window"], "start": row["start"], "end": row["end"],
                "event": event, "diagnosis": row["diagnosis"],
                "drift_active": row.get("drift_active", False),
                "data_active": row.get("data_active", False),
                "relational_active": row.get("relational_active", False),
                "data_alarm": row["data_alarm"], "relational_alarm": row["relational_alarm"],
                "alarm": row["alarm"], "adapted": row["adapted"], "recovery": row["recovery"],
                "true_data_drift": d, "true_concept_drift": c,
            })
    pd.DataFrame(event_rows).to_csv("detector_events.csv", index=False)

    detector_by_seed = pd.DataFrame(detector_rows_all)
    detector_by_seed.to_csv("detector_metrics_by_seed.csv", index=False)
    pd.DataFrame(window_rows_all).to_csv("detector_window_metrics_by_seed.csv", index=False)

    summary_metrics = [
        "Precision", "Recall", "F1", "TP", "FP", "FN",
        "True drift episodes", "Detected alarms", "Matched detections",
        "Mean detection delay (windows)",
    ]
    pd.DataFrame([
        {"Metric": m, "Mean": detector_by_seed[m].mean(), "Std": detector_by_seed[m].std()}
        for m in summary_metrics
    ]).to_csv("detector_metrics_summary.csv", index=False)

    type_df = pd.DataFrame(type_rows_all)
    type_df.to_csv("drift_type_evaluation.csv", index=False)
    type_summary = (
        type_df.groupby("true_type", as_index=False)
        .agg(Episodes=("episode", "count"), Correct=("type_correct", "sum"),
             Type_Accuracy=("type_correct", "mean"), Alarm_Detected=("alarm_detected", "sum"))
    )
    type_summary.to_csv("drift_type_summary.csv", index=False)

    print("\nExperiment complete.")
    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print("\nDetector metrics:")
    print(detector_by_seed.to_string(index=False))
    print("\nDrift-type summary:")
    print(type_summary.to_string(index=False))
    print("\nFiles created: model_comparison.csv, stream_log.csv, detector_events.csv, detector_metrics_by_seed.csv, detector_metrics_summary.csv, detector_window_metrics_by_seed.csv, drift_type_evaluation.csv, drift_type_summary.csv")


if __name__ == "__main__":
    main()
