$ErrorActionPreference = 'Stop'
Write-Host '=== VayuNetra Phase 3: NIO Dataset Expansion ==='
python -m pip install imdtrack
python backend\ml\scripts\build_nio_besttrack.py
python backend\ml\scripts\audit_nio_data.py
python backend\ml\scripts\generate_mosdac_jobs.py
python backend\ml\scripts\build_final_nio_manifest.py
Write-Host 'Phase 3 local preparation complete.'
Write-Host 'MOSDAC download templates are in backend\data\mosdac_jobs.'
Write-Host 'For actual downloads, use the official MOSDAC mdapi client with your credentials.'
