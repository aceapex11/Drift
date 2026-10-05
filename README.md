# Live drift detection + transfer-learning adaptation (classic ML, no deep learning)

    python data_generator.py        # writes data/drift_regression_dataset_large.csv (60,000 rows, 8 sensors, 6 drift events + 1 glitch)
    python run_experiment.py        # detector evaluation + Static vs Retrain vs Transfer  -> results/
    streamlit run app.py            # dashboard, "Embedded engine" works with no other process
    uvicorn stream_server:app --port 8000   # optional FastAPI WebSocket; choose "FastAPI server" in the dashboard

## Pipeline
predict (test-then-train) -> detect -> diagnose -> adapt
* Data drift: per-sensor KS test on every 600-hour batch. Alarm level per sensor is calibrated from the normal training
  period (max(0.15, 1.2 x largest KS of a normal block)). PSI is reported as severity. `operating_hours` is a maintenance
  counter and is not monitored.
* Relational drift: Page-Hinkley (own implementation, same maths as river) on the clipped residual of a frozen poly-2 Ridge "old rule".
* Diagnosis: none / data / relational / both. A "drift ended" trigger fires when the diagnosis returns to none.
* Transfer learning: GradientBoostingRegressor pre-trained on normal history; on a trigger the existing trees are frozen and 80
  new trees are boosted on recent data (warm start). Data-only drift and recovery also replay 2,500 source rows; relational/both
  use recent windows only because the old relation is invalid.

## Dataset events (window = 600 h, stream starts at row 14,400)
| # | rows | type | mode |
|---|---|---|---|
| 1 | 18,000-21,000 | data | sudden |
| 2 | 25,200-29,400 | relational | sudden |
| 3 | 32,400-37,200 | both | gradual (2,400-row ramp) |
| 4 | 39,600-42,600 | relational | sudden, recurring concept of #2 |
| 5 | 45,000-51,000 | data | sudden (variance + mean) |
| 6 | 52,800-60,000 | both | sudden |
| - | 30,600-30,612 | sensor glitch, NOT a drift | should raise no alarm |

Ground-truth columns (regime, data_drift, concept_drift, true_drift_point) are used only for scoring and for shading the demo chart.
