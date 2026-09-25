VAYUNETRA — FRONTEND/BACKEND FIX PACK
=====================================

This pack fixes the current browser/API integration errors found in the VayuNetra console.

FIXES INCLUDED
--------------
1. Restores POST /predict for the synthetic validation panel.
   - Uses the existing trained synthetic classifier + predictor.
   - Returns mode=trained_model and explicitly marks the result as synthetic validation.
   - Does NOT present synthetic output as real-world cyclone performance.

2. Adds GET /insat_preview/tir1 and GET /insat_preview/wv.
   - Renders the first processed local INSAT TIR1/WV pair as PNG.
   - Fixes the dashboard preview requests that previously had no backend route.

3. Adds GET /debug_paths.
   - Gives the frontend non-secret diagnostics for TCIR, INSAT sample paths and model availability.

4. Keeps the existing TCIR, INSAT, Track, Metrics and Evaluation endpoints intact.

FILES
-----
backend/app.py       -> replace your current backend/app.py
frontend/app.js      -> replace your current frontend/app.js
frontend/index.html  -> replace your current frontend/index.html
TEST_ENDPOINTS.ps1  -> optional smoke test script

INSTALL
-------
1. Extract this ZIP into a temporary folder.
2. From your existing VayuNetra project, replace ONLY the files above with the files from this pack.
3. Keep your existing data/, checkpoints/, models/, assets/ and other project files.
4. Start the backend:

   python -m uvicorn backend.app:app --reload --port 8000

5. Hard-refresh the browser (Ctrl+Shift+R).
6. Run the smoke test if desired:

   powershell -ExecutionPolicy Bypass -File .\TEST_ENDPOINTS.ps1

EXPECTED
--------
The smoke test should report:
  /health             PASS
  /data_sources       PASS
  /demo_sequence      PASS
  /predict             PASS
  /insat_demo          PASS (if processed INSAT data exists)
  /insat_preview/tir1 PASS (if processed INSAT data exists)
  /insat_preview/wv   PASS (if processed INSAT data exists)
  /metrics             PASS
  /evaluation_summary PASS
  /track               PASS

IMPORTANT
---------
This is a drop-in integration fix pack, not a standalone copy of the entire ML project.
It intentionally does not include large HDF5 datasets, model checkpoints, credentials, or raw satellite data.
