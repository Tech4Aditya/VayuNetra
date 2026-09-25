# VayuNetra ML V2

This pack fixes the failed INSAT multitask formulation and adds a multi-horizon learned track experiment.

## INSAT V2
- regression targets are normalized using training-set mean/std
- class-weighted cross entropy handles small/imbalanced classes
- validation score combines classification + normalized regression losses
- ReduceLROnPlateau + gradient clipping + early stopping
- test evaluator reports accuracy, macro-F1, per-class metrics, confusion matrix, Wind MAE/RMSE/R2 and Pressure MAE/RMSE/R2

Checkpoint: `backend/checkpoints/insat_v2_best.pt`

## Track V2
Uses the real best-track CSVs, with storm-level splitting. Five observations are used to predict displacement at 3h, 6h and 9h horizons.

Checkpoint: `backend/checkpoints/track_gru_multi_best.pt`

Evaluation: `backend/data/processed/track_multi_evaluation.json`

## Run
From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_ml_v2.ps1
```

Do not overwrite your existing `backend/data` or working checkpoints before verifying the new evaluation outputs. V2 checkpoints are saved under new names.

## Interpretation
The dataset is small, so the results are experimental held-out-storm results, not operational forecast accuracy. Do not convert R2 into a generic accuracy percentage.
