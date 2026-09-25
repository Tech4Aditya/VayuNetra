# VayuNetra ML Completion Pack

This pack finishes the two unfinished scientific components identified in `Incomplete things.pdf`:

1. Proper storm-level INSAT train/validation/test preparation and evaluation.
2. A learned GRU cyclone track predictor that predicts Δlatitude/Δlongitude from recent track history.

## Important
The 5.5 GB TCIR HDF5 and your local INSAT processed arrays are not included in this ZIP. Therefore this package is **ready-to-train**, but the new checkpoints cannot be truthfully claimed as trained inside this chat. Run the supplied commands on the machine containing your dataset.

Existing VayuNetra checkpoints are preserved in `backend/checkpoints/`.

## Install

From project root:

```powershell
pip install torch numpy pandas scikit-learn h5py
```

## 1. Build storm-level INSAT splits

```powershell
python backend/ml/scripts/prepare_insat_splits.py
```

Output:
- `backend/data/processed/insat_splits/insat_train.csv`
- `backend/data/processed/insat_splits/insat_val.csv`
- `backend/data/processed/insat_splits/insat_test.csv`

No storm is shared across train/validation/test.

## 2. Train the INSAT model

```powershell
python backend/ml/scripts/train_insat_complete.py --epochs 50
```

Output:
- `backend/checkpoints/insat_complete_best.pt`
- `backend/data/processed/insat_complete_stats.json`

The model predicts cyclone category, wind and pressure. Size is included automatically if a usable size label exists in the manifests.

## 3. Evaluate INSAT

```powershell
python backend/ml/scripts/evaluate_insat_complete.py
```

Output:
- `backend/data/processed/insat_complete_evaluation.json`
- `backend/data/processed/insat_complete_evaluation.csv`

## 4. Build track dataset

The builder searches for `tcir_track_sequences/*.csv` first and can also consume `tcir_manifest.csv` if it contains latitude/longitude and the usual storm/time fields.

```powershell
python backend/ml/scripts/prepare_track_dataset.py
```

Output:
- `backend/data/processed/track_train.csv`
- `backend/data/processed/track_val.csv`
- `backend/data/processed/track_test.csv`

## 5. Train learned track predictor

```powershell
python backend/ml/scripts/train_track_gru.py --epochs 80
```

Output:
- `backend/checkpoints/track_gru_best.pt`
- `backend/data/processed/track_scaler.json`

The network predicts future Δlat and Δlon for the next interval.

## 6. Evaluate track prediction

```powershell
python backend/ml/scripts/evaluate_track.py
```

Output:
- latitude MAE
- longitude MAE
- position RMSE
- great-circle distance error
- 3h/6h/9h evaluation when matching horizons exist

## Integration

The existing VayuNetra UI/backend is intentionally preserved. After training, the new checkpoint can be wired into `/track` without replacing the existing TCIR intensity model.

The current motion projection remains available as a baseline so the learned model can be compared against it rather than silently replacing it.


## Track dataset fix
The track builder now consumes the real `tcir_temporal_train/val/test.csv` manifests used by VayuNetra, sorts observations chronologically, and predicts the displacement from the latest five observations to the next observation. This removes the previous target leakage where the target transition was already inside the input window.
