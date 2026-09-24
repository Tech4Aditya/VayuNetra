VayuNetra FINAL PATCH 2

IMPORTANT:
1. Keep the existing backend/data folder. It contains the 5.52 GB TCIR HDF5 and all manifests.
2. Replace only backend/app.py, frontend/index.html, frontend/app.js, frontend/style.css.
3. Restart uvicorn completely after replacing files.

Run from project root:
  python -m uvicorn backend.app:app --reload --port 8000

Then verify in browser:
  http://127.0.0.1:8000/debug_paths

Expected:
  tcir_h5.exists = true
  tcir_manifest.exists = true
  size_gb ~= 5.15

This patch:
- resolves data/checkpoint paths from the actual project root rather than cwd
- exposes /debug_paths so path problems are visible
- returns detailed TCIR path diagnostics
- removes the MOSDAC iframe (it is blocked by MOSDAC CSP frame-ancestors)
- renders the local processed INSAT TIR-1 and WV arrays directly
- keeps MOSDAC as the documented source instead of embedding its website
