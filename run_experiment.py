"""
Offline experiment runner.

IMPORTANT:
The project filename is `run_experiment.py`.
It imports `drift_engine.py`.

Outputs:
- model_comparison.csv
- stream_log.csv
- detector_events.csv
- detector_metrics_by_seed.csv
- detector_metrics_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_generator import (
    TRAIN_END,
    WINDOW,
    generate,
)
from drift_engine import (
    STRATEGY_NAMES,
    StrategyRunner,
)


SEEDS = [0, 1, 2, 3, 4]


def window_truth(df, start, end):
    """
    Offline evaluation label for a stream window.

    A window is considered a true drift window if at least one row
    inside it has data drift OR concept/relational drift.
    """
    data_truth = bool(
        df.loc[start:end - 1, "data_drift"]
        .astype(bool)
        .any()
    )

    concept_truth = bool(
        df.loc[start:end - 1, "concept_drift"]
        .astype(bool)
        .any()
    )

    return data_truth, concept_truth


def true_drift_episodes(rows, df):
    """Return contiguous true-drift episodes at the stream-window level."""
    flags = []
    for row in rows:
        data_truth, concept_truth = window_truth(
            df, int(row["start"]), int(row["end"])
        )
        flags.append(bool(data_truth or concept_truth))

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


def evaluate_detector(rows, df, max_delay_windows=2):
    """
    Event-level drift evaluation.

    A drift detector normally emits an alarm at the start/transition of a
    drift episode, not once for every window inside a long drift regime.
    Therefore the old window-by-window confusion matrix made a detector that
    correctly raised one alarm for a long episode look like it missed all of
    the remaining drift windows. Here an alarm is a TP when it falls inside
    the true episode or within `max_delay_windows` after its end.
    """
    episodes = true_drift_episodes(rows, df)
    alarms = [i for i, row in enumerate(rows) if bool(row["alarm"])]

    matched_alarm_indices = set()
    matched_episodes = []
    delays = []

    for ep_idx, (start, end) in enumerate(episodes):
        candidates = [
            a for a in alarms
            if a not in matched_alarm_indices
            and start <= a <= end + max_delay_windows
        ]
        if candidates:
            a = min(candidates)
            matched_alarm_indices.add(a)
            matched_episodes.append(ep_idx)
            delays.append(max(0, a - start))

    tp = len(matched_episodes)
    fn = len(episodes) - tp
    fp = len([a for a in alarms if a not in matched_alarm_indices])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "Accuracy": np.nan,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": np.nan,
        "True drift episodes": int(len(episodes)),
        "Detected alarms": int(len(alarms)),
        "Matched detections": int(tp),
        "Mean detection delay (windows)": float(np.mean(delays)) if delays else np.nan,
        "Max acceptable delay (windows)": int(max_delay_windows),
    }


def evaluate_detector_windowwise(rows, df):
    """Keep the old window-level metrics as a diagnostic, not the headline metric."""
    y_true, y_pred = [], []
    for row in rows:
        data_truth, concept_truth = window_truth(df, int(row["start"]), int(row["end"]))
        y_true.append(int(data_truth or concept_truth))
        y_pred.append(int(bool(row["alarm"])))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def main():
    print("Generating/loading synthetic stream...")
    df = generate()

    all_model_rows = []
    detector_rows = []
    window_detector_rows = []

    # Run all five learning strategies for every seed.
    for seed in SEEDS:
        print(f"\nRunning seed {seed}...")

        runner = StrategyRunner(
            df,
            TRAIN_END,
            WINDOW,
            seed=seed,
        )

        records = runner.run()

        all_model_rows.extend(records)

        # Detector evaluation can use any one strategy because the
        # detector does not depend on the learner's prediction model.
        # We use Incremental for the primary detector result.
        detector_rows_seed = [
            r for r in records
            if r["strategy"] == "Incremental"
        ]

        detector_metric = evaluate_detector(
            detector_rows_seed,
            df,
            max_delay_windows=2,
        )
        detector_metric["Seed"] = seed
        detector_rows.append(detector_metric)

        window_metric = evaluate_detector_windowwise(
            detector_rows_seed, df
        )
        window_metric["Seed"] = seed
        window_detector_rows.append(window_metric)

        # Save first-seed stream details for the dashboard.
        if seed == SEEDS[0]:
            stream_df = pd.DataFrame(
                detector_rows_seed
            )
            stream_df.to_csv(
                "stream_log.csv",
                index=False,
            )

            event_rows = []

            for row in detector_rows_seed:
                data_truth, concept_truth = window_truth(
                    df,
                    int(row["start"]),
                    int(row["end"]),
                )

                if row["alarm"]:
                    event_rows.append(
                        {
                            "window": row["window"],
                            "start": row["start"],
                            "end": row["end"],
                            "event": "detected_drift",
                            "diagnosis": row["diagnosis"],
                            "data_alarm": row["data_alarm"],
                            "relational_alarm": row[
                                "relational_alarm"
                            ],
                            "adapted": row["adapted"],
                            "true_data_drift": data_truth,
                            "true_concept_drift": concept_truth,
                        }
                    )

            pd.DataFrame(event_rows).to_csv(
                "detector_events.csv",
                index=False,
            )

    model_df = pd.DataFrame(all_model_rows)

    comparison = (
        model_df
        .groupby("strategy", as_index=False)
        .agg(
            MAE=("mae", "mean"),
            RMSE=("rmse", "mean"),
            R2=("r2", "mean"),
            Windows=("window", "count"),
        )
        .rename(columns={"strategy": "Strategy"})
    )

    strategy_order = {
        name: i
        for i, name in enumerate(STRATEGY_NAMES)
    }
    comparison["_order"] = comparison["Strategy"].map(strategy_order)
    comparison = (
        comparison
        .sort_values("_order")
        .drop(columns="_order")
    )

    comparison.to_csv(
        "model_comparison.csv",
        index=False,
    )

    detector_by_seed = pd.DataFrame(detector_rows)
    detector_by_seed.to_csv(
        "detector_metrics_by_seed.csv",
        index=False,
    )

    pd.DataFrame(window_detector_rows).to_csv(
        "detector_window_metrics_by_seed.csv",
        index=False,
    )

    detector_summary = pd.DataFrame([
        {
            "Metric": metric,
            "Mean": detector_by_seed[metric].mean(),
            "Std": detector_by_seed[metric].std(),
        }
        for metric in [
            "Precision", "Recall", "F1", "TP", "FP", "FN",
            "True drift episodes", "Detected alarms", "Matched detections",
            "Mean detection delay (windows)",
        ]
    ])

    detector_summary.to_csv(
        "detector_metrics_summary.csv",
        index=False,
    )


    print("\nExperiment complete.")
    print("\nModel comparison:")
    print(comparison.to_string(index=False))

    print("\nDetector metrics by seed:")
    print(detector_by_seed.to_string(index=False))

    print(
        "\nFiles created:"
        "\n  model_comparison.csv"
        "\n  stream_log.csv"
        "\n  detector_events.csv"
        "\n  detector_metrics_by_seed.csv"
        "\n  detector_window_metrics_by_seed.csv"
        "\n  detector_metrics_summary.csv"
    )


if __name__ == "__main__":
    main()
