# VayuNetra — Submission Demo Build

## What this build fixes

- **No illustrative/demo cyclone coordinates.** The Track window uses a verified historical best-track replay for `200301L` (Ana, 2003).
- **Latitude/longitude are real geographic coordinates.** The bundled track catalog contains the observed positions used by the demo.
- **Track + Nowcast:** the observed path is shown separately from a short-horizon constant-velocity motion baseline (+3h/+6h/+9h).
- **Real INSAT source:** the Satellite window embeds the MOSDAC INSAT-3DR/3DS live observation interface instead of an empty local placeholder.
- **TCIR AI models remain unchanged:** temporal CNN/GRU intensity model and single-frame intensity model are still used from the existing project.
- **Synthetic path remains explicitly validation-only.**

## Important scientific distinction for the presentation

VayuNetra's current TCIR neural network predicts **wind, pressure and storm size**. It does not output latitude/longitude. Therefore the geographic track layer is intentionally implemented as:

1. observed best-track coordinates from the historical replay dataset;
2. transparent constant-velocity motion projection from the latest observed displacement.

Do **not** describe the +3h/+6h/+9h line as neural-network track prediction in the presentation unless a separately trained track model is added and evaluated.

## Run

From the existing VayuNetra project root, replace:

- `frontend/index.html`
- `frontend/app.js`
- `frontend/style.css`
- `backend/app.py`
- add `backend/track_catalog.json`

Keep your existing `backend/data/`, `backend/models/`, and other project files.

Then:

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

Open the frontend as you normally do.

## Demo flow

1. `RUN TCIR LIVE ANALYSIS`
2. Show 4 TCIR frames and intensity output.
3. Open `FORECAST` → show historical track + +3h/+6h/+9h motion projection.
4. Open `SATELLITE` → show live INSAT-3DR/3DS / MOSDAC panel.
5. Open `DATA SOURCES` → show TCIR / INSAT / labels / validation paths.
6. Open `REPORTS` → show available evaluation metrics.

## Data provenance

The track replay is based on the 2003 Atlantic storm `01L / Ana`. NOAA/NHC identifies Ana as storm number 1 in the 2003 Atlantic season; the best-track table used for the replay provides six-hourly positions and intensity values.

The INSAT live source is MOSDAC's INSAT-3DR/3DS visualization. MOSDAC documents INSAT-3DR's TIR, water-vapour and other imager products and its use for cyclone monitoring.
