# VayuNetra Phase 3 — NIO Dataset Expansion

This pack completes the reproducible/local part of Phase 3 without storing MOSDAC credentials.

## Target 10 storms
FANI, TAUKTAE, AMPHAN, YAAS, ASANI, NISARGA, BULBUL, GULAAB, BIPARJOY, MICHAUNG.

## Run
From project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_nio_phase3.ps1
```

The best-track builder uses `imdtrack`, which mirrors the IMD RSMC New Delhi best-track record and can update from the current record. It exports one CSV per target storm into `backend/data/labels/` and `backend/data/labels/nio_besttracks/`.

The audit checks best-track completeness and existing INSAT calibrated TIR1/WV pairs.

The MOSDAC job generator creates credential-free templates. Actual download still requires a MOSDAC account and the official `mdapi.py` client. MOSDAC's official API documentation specifies `datasetId`, date range, bounding box, count, and account authentication for downloads.

Default dataset: `3DIMG_L1C_ASIA_MER`, the INSAT-3D six-channel L1C Asian-sector product. It is half-hourly and geo-located.

After MOSDAC downloads are present, rerun the audit and manifest builder.

## Sources
- IMD RSMC New Delhi Best Track: https://rsmcnewdelhi.imd.gov.in/report.php?internal_menu=MzM
- MOSDAC API manual: https://mosdac.gov.in/downloadapi-manual
- MOSDAC INSAT-3D L1C Asian sector: https://mosdac.gov.in/doi/123/

## Automated MOSDAC runner
The pack also includes `run_mosdac_jobs.ps1`. It downloads the official MOSDAC `mdapi.zip`, injects credentials only in memory/runtime into the required `config.json`, and runs each generated storm job.

Set credentials only for the current PowerShell session:

```powershell
$env:MOSDAC_USERNAME="YOUR_USERNAME"
$env:MOSDAC_PASSWORD="YOUR_PASSWORD"
powershell -ExecutionPolicy Bypass -File .\run_mosdac_jobs.ps1
```

Do not commit or paste credentials into project files.
