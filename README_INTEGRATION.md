# VayuNetra — Integrated Cyclone Intelligence Console

## Drop-in files
Copy these into the existing frontend/backend locations:

- `index.html` -> frontend `index.html`
- `style.css` -> frontend `style.css`
- `app.js` -> frontend `app.js`
- `app.py` -> backend `app.py`

## Backend
From the project root:

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

The integrated backend exposes:
- `/health`
- `/data_sources`
- `/metrics`
- `/tcir_demo`
- `/predict_tcir_sequence`
- `/predict_tcir_single`
- `/insat_demo`
- `/predict_insat`
- legacy synthetic `/demo_sequence`, `/predict`, `/predict_image`

## What is integrated

### TCIR
- Real `Cyclone_Images.h5`
- 4-frame chronological temporal model
- IR + PMW input
- wind / pressure / size inference
- deterministic North Indian Ocean category from predicted wind
- single-frame TCIR model for comparison

### INSAT
- Real-data `classifier_insat.pt`
- TIR1 + WV
- 7 intensity categories
- wind regression
- real processed sample discovery under `backend/data/processed`

### Synthetic validation
Kept isolated and explicitly labelled as validation-only. It is not presented as real-world cyclone evidence.

## Frontend windows
- Overview
- Live TCIR Analysis
- Satellite Intelligence
- Forecast / Model Comparison
- Data Source Matrix
- Evaluation / Reports

The UI also keeps the main four-card dashboard visible and opens detailed analysis as command-console windows.

## Important
The backend reports data/model readiness from actual local paths. It does not pretend that a live external satellite feed exists unless the corresponding local data/model path is present.
