"""Run the five SGD incremental/continual strategies and save model_comparison.csv."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_generator import generate, TRAIN_END, WINDOW
from drift_engine_incremental import StrategyRunner, STRATEGY_NAMES

SEEDS = [0, 1, 2, 3, 4]


def main():
    df = generate()
    rows = []
    for seed in SEEDS:
        runner = StrategyRunner(df, TRAIN_END, WINDOW, seed=seed)
        seed_rows = runner.run()
        for row in seed_rows:
            row["seed"] = seed
        rows.extend(seed_rows)

    detail = pd.DataFrame(rows)
    # Re-run labels from the generated data for offline evaluation only.
    true_points = df["true_drift_point"].dropna().astype(int).to_numpy() if "true_drift_point" in df.columns else np.array([], dtype=int)
    if len(true_points):
        detail["true_drift_event"] = [bool(np.any((true_points >= s) & (true_points < e))) for s, e in zip(detail["start"], detail["end"])]
    else:
        detail["true_drift_event"] = False
    detail.to_csv("stream_results_incremental.csv", index=False)

    summary = (
        detail.groupby("strategy")[["mae", "rmse", "r2", "mae_x_normal"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv("model_comparison.csv", index=False)

    # ------------------------------------------------------------
    # OFFLINE DRIFT-DETECTOR CLASSIFICATION METRICS
    # ------------------------------------------------------------
    # Precision / recall / F1 / accuracy are appropriate here because
    # the detector produces a binary event alarm. They are NOT used
    # as regression metrics for health_deterioration.
    # Ground truth is used only after the live run for evaluation.
    detector_rows = []
    for seed in SEEDS:
        seed_detail = detail[detail["seed"] == seed].copy() if "seed" in detail.columns else detail.copy()
        # true_drift_event is created below from the generator's true_drift_point.
        if "true_drift_event" not in seed_detail.columns:
            continue
        y_true = seed_detail["true_drift_event"].astype(int)
        y_pred = seed_detail["alarm"].astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        detector_rows.append({
            "seed": seed,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        })

    if detector_rows:
        detector_detail = pd.DataFrame(detector_rows)
        detector_detail.to_csv("detector_metrics_by_seed.csv", index=False)
        detector_summary = detector_detail[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"])
        detector_summary.to_csv("detector_metrics_summary.csv")
        print("\nDetector metrics (offline evaluation):")
        print(detector_summary.to_string())

    print("Strategies:", STRATEGY_NAMES)
    print("Saved: stream_results_incremental.csv")
    print("Saved: model_comparison.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
