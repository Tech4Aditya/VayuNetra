# VayuNetra — Learned Track GRU Integration

This is a drop-in update for the existing Track & Motion UI.

## Files
- `backend/app.py` — loads `backend/checkpoints/track_gru_multi_best.pt` and serves learned 3h/6h/9h trajectory predictions from `/track`.
- `backend/models/track_gru_multi.py` — architecture matching the trained checkpoint.
- `frontend_app.js` — renders the learned forecast instead of treating the constant-motion baseline as the main projection.
- `frontend_index.html` — labels the projection as AI GRU forecast.
- `frontend_style.css` — unchanged current Track Motion stylesheet; rename/copy to `frontend/style.css`.

## Required existing checkpoint
The trained checkpoint must already exist at:

`backend/checkpoints/track_gru_multi_best.pt`

The integration does **not** overwrite the checkpoint.

## Install / copy
From the VayuNetra project root, replace:

- `backend/app.py` with `backend/app.py` from this pack
- `frontend/app.js` with `frontend_app.js`
- `frontend/index.html` with `frontend_index.html`
- keep your existing `frontend/style.css` (the supplied `frontend_style.css` is the matching copy)
- add `backend/models/track_gru_multi.py`

## Run
```powershell
python -m uvicorn backend.app:app --reload --port 8000
```

Then open the existing VayuNetra frontend.

## What `/track` now returns
- `history`: observed coordinates
- `projection`: learned GRU positions at +3h, +6h, +9h
- `track_model`: model availability and feature information
- `motion_baseline`: old constant-motion projection retained for comparison
- `motion`: old motion metadata

The UI uses `projection` as the primary forecast and keeps the observed track separate.

## Model scope
The trained Track GRU was evaluated on held-out storms with great-circle errors of 33.59 km (3h), 51.87 km (6h), and 76.11 km (9h). It is a learned trajectory model using historical best-track features; it is not an end-to-end satellite-image-to-track model.
