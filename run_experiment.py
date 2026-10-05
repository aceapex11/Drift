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


def evaluate_detector(rows, df):
    y_true = []
    y_pred = []

    for row in rows:
        data_truth, concept_truth = window_truth(
            df,
            int(row["start"]),
            int(row["end"]),
        )

        true_drift = data_truth or concept_truth
        detected = bool(row["alarm"])

        y_true.append(int(true_drift))
        y_pred.append(int(detected))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "True drift windows": int(sum(y_true)),
        "Detected drift windows": int(sum(y_pred)),
    }


def main():
    print("Generating/loading synthetic stream...")
    df = generate()

    all_model_rows = []
    detector_rows = []

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
        )
        detector_metric["Seed"] = seed
        detector_rows.append(detector_metric)

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

    detector_summary = pd.DataFrame(
        [
            {
                "Metric": "Accuracy",
                "Mean": detector_by_seed["Accuracy"].mean(),
                "Std": detector_by_seed["Accuracy"].std(),
            },
            {
                "Metric": "Precision",
                "Mean": detector_by_seed["Precision"].mean(),
                "Std": detector_by_seed["Precision"].std(),
            },
            {
                "Metric": "Recall",
                "Mean": detector_by_seed["Recall"].mean(),
                "Std": detector_by_seed["Recall"].std(),
            },
            {
                "Metric": "F1",
                "Mean": detector_by_seed["F1"].mean(),
                "Std": detector_by_seed["F1"].std(),
            },
            {
                "Metric": "TP",
                "Mean": detector_by_seed["TP"].mean(),
                "Std": detector_by_seed["TP"].std(),
            },
            {
                "Metric": "FP",
                "Mean": detector_by_seed["FP"].mean(),
                "Std": detector_by_seed["FP"].std(),
            },
            {
                "Metric": "FN",
                "Mean": detector_by_seed["FN"].mean(),
                "Std": detector_by_seed["FN"].std(),
            },
            {
                "Metric": "TN",
                "Mean": detector_by_seed["TN"].mean(),
                "Std": detector_by_seed["TN"].std(),
            },
        ]
    )

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
        "\n  detector_metrics_summary.csv"
    )


if __name__ == "__main__":
    main()
