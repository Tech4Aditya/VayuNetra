## Current Development Status

The project is currently transitioning from a synthetic-data prototype
to a real satellite-data pipeline.

### Completed

- Initial cyclone classification prototype.
- Multi-task CNN architecture.
- Cyclone presence detection.
- Cyclone category classification.
- Intensity regression head.
- Temporal prediction prototype.
- INSAT-3D dataset discovery through MOSDAC.
- IBTrACS North Indian Ocean track data integration.
- Real cyclone track preprocessing.
- Cyclone intensity/category label generation.
- Satellite observation matching manifest.
- HDF5 inspection tooling.

### Real Cyclone Track Dataset

North Indian Ocean cyclone observations are currently prepared from
NOAA IBTrACS data.

Initial storms:

- FANI (2019)
- AMPHAN (2020)
- NIVAR (2020)
- TAUKTAE (2021)
- YAAS (2021)
- BIPARJOY (2023)
- HAMOON (2023)
- MICHAUNG (2023)

Current track-label dataset:

- 456 observations
- 365 observations with available intensity labels
- 91 observations with missing intensity values
- Valid latitude/longitude coordinates
- 0 duplicate storm/timestamp records
- 3-hour track observation interval

### Satellite Data

The intended satellite source is INSAT-3D Level-1C IMAGER data
(`3DIMG_L1C_SGP`) obtained through MOSDAC.

The satellite ingestion pipeline is currently pending MOSDAC
account approval.

### Current Pipeline

```text
IBTrACS
   ↓
Cyclone Track Labels
   ↓
Satellite Observation Manifest
   ↓
INSAT-3D HDF5
   ↓
Channel Extraction
   ↓
Geolocation
   ↓
Cyclone Crop
   ↓
Real Training Dataset
   ↓
Detection + Classification + Intensity
   ↓
Temporal Track / Trend Prediction
