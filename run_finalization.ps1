$ErrorActionPreference = "Stop"
Write-Host "=== VayuNetra Finalization ==="
python backend\ml\scripts\evaluate_insat_baseline.py
python backend\ml\scripts\analyze_insat_failures.py
python backend\ml\scripts\evaluate_insat_v2.py
python backend\ml\scripts\final_system_evaluation.py
Write-Host "=== Finalization artifacts generated ==="
