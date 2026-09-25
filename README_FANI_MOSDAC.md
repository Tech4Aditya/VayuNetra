# FANI MOSDAC preprocessing pack

This script processes the newly downloaded FANI INSAT-3D MOSDAC HDF5 files using the project's existing INSAT preprocessing logic.

## Expected raw input

`backend/data/raw/mosdac/fani/3DIMG_L1C_ASIA_MER/**/*.h5`

## Command

From the VayuNetra project root:

```powershell
python backend\data\scripts\process_fani_mosdac.py
```

## Output

```text
backend/data/processed/insat_fani_calibrated/<HDF5-stem>/tir1_radiance.npy
backend/data/processed/insat_fani_calibrated/<HDF5-stem>/wv_radiance.npy
backend/data/labels/fani_insat_manifest.csv
```

The script keeps the existing project calibration coefficients, INSAT Mercator projection, 128x128 cyclone-centered crop, and +/- 1 hour best-track matching logic.
