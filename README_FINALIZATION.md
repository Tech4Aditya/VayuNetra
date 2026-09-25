# VayuNetra — Final ML / End-to-End Finalization Pack

This pack finishes the remaining Phase 4 / Phase 6 engineering work without replacing the validated Track GRU or TCIR checkpoints.

## 1. INSAT V2 backend integration
`backend/app.py` now prefers:
`backend/checkpoints/insat_v2_best.pt`

It falls back to the legacy `classifier_insat.pt` only if V2 is absent.

The V2 endpoint uses the exact training-set normalization stored inside the V2 checkpoint and denormalizes wind/pressure predictions correctly.

## 2. Phase 4 baseline
Run:
```powershell
python backend\ml\scripts\evaluate_insat_baseline.py
```
This computes a transparent baseline:
- majority-class classification
- training-set mean wind
- training-set mean pressure

Do not call the V2 classification successful unless it is compared against this baseline.

## 3. Failure analysis
Run:
```powershell
python backend\ml\scripts\analyze_insat_failures.py
```
This writes the confusion matrix, predicted class distribution and hardest regression cases.

## 4. Final system evaluation
Run:
```powershell
python backend\ml\scripts\final_system_evaluation.py
```
This verifies that the required checkpoints, splits and evaluation artifacts exist.

## 5. API verification
Start the backend:
```powershell
python -m uvicorn backend.app:app --reload --port 8000
```
Then verify:
- `/health`
- `/data_sources`
- `/evaluation_summary`
- `/insat_demo`
- `/track?storm=AMPHAN`

## Current validated metrics
TCIR temporal:
- Wind MAE 12.43 kt; R² 0.6769
- Pressure MAE 8.84 hPa; R² 0.6944
- Size MAE 31.35 nmi; R² 0.4606

Track GRU, held-out storms:
- 3h 33.59 km
- 6h 51.87 km
- 9h 76.11 km

INSAT V2, held-out storms:
- Category accuracy 16.33%
- Macro F1 0.10
- Wind MAE 20.15 kt; R² 0.031
- Pressure MAE 11.42 hPa; R² -0.023

These numbers are evaluation results, not claims of operational forecast accuracy.
