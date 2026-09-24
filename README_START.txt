VAYUNETRA FINAL DELIVERY

1. Keep your existing backend/data and backend/checkpoints folders. Do NOT replace/delete them.
2. Replace only backend/app.py, backend/track_catalog.json, frontend/index.html, frontend/app.js, frontend/style.css with the files in this patch.
3. From the project root run:
   python -m uvicorn backend.app:app --reload --port 8000
4. Open:
   http://127.0.0.1:8000/debug_paths
   TCIR H5 and manifest must both show exists=true.
5. Hard refresh the frontend with Ctrl+Shift+R.
6. Run TCIR, then INSAT.

The backend now searches from the backend directory, project root, current working directory and parent roots for the actual dataset files. INSAT sample discovery searches the processed directory and all INSAT manifests for explicit TIR/WV paths.
