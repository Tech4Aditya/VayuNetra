# VayuNetra 🌪️

### Cyclone Identification, Intensity Analysis & Future Track Prediction using Satellite Imagery

**Smart India Hackathon 2026 — SIH26070**
**Ministry of Earth Sciences (MoES)**

---

## 📌 Overview

**VayuNetra** is a research-oriented cyclone analysis system designed to process satellite imagery and meteorological best-track information for tropical cyclone analysis.

The project combines:

- 🛰️ INSAT-3D satellite observations
- 🌊 Tropical Cyclone Image Recognition (TCIR) data
- 📍 IMD best-track observations
- 🧠 Deep learning
- ⏱️ Temporal sequence modeling
- 🗺️ Cyclone position tracking
- ⚡ Real-time inference through a backend API

The development strategy follows a staged approach:

```text
TCIR Real Satellite Data
        ↓
Intensity Model Pretraining
        ↓
Temporal Intensity Modeling
        ↓
INSAT-3D / NIO Domain Adaptation
        ↓
Multi-Storm NIO Dataset
        ↓
Intensity Fine-Tuning
        ↓
Track Prediction
        ↓
End-to-End Cyclone Analysis
```

> **Important:** VayuNetra is currently a research prototype. Claims of operational cyclone detection, forecasting, or prediction are not made until the corresponding real-world evaluation is completed.

---

## 🎯 Problem Statement

Cyclone monitoring requires the analysis of rapidly changing atmospheric systems using satellite observations and meteorological information.

Traditional workflows can involve:

- Manual satellite-image interpretation
- Separate analysis of historical observations
- Delayed intensity estimation
- Limited temporal modeling
- Difficulty integrating heterogeneous satellite products

VayuNetra aims to develop a unified machine-learning pipeline capable of learning cyclone characteristics from historical satellite observations and eventually providing:

- Cyclone intensity estimation
- Temporal intensity analysis
- Cyclone position/track prediction
- Satellite-image based cyclone analysis
- NIO-specific domain adaptation

---

## 🧠 System Architecture

```
                    ┌─────────────────────┐
                    │   Satellite Data    │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┴─────────────────┐
             │                                   │
             ▼                                   ▼
     ┌───────────────┐                   ┌───────────────┐
     │     TCIR      │                   │   INSAT-3D    │
     │  Historical   │                   │    MOSDAC     │
     │   Dataset     │                   │   Observations│
     └───────┬───────┘                   └───────┬───────┘
             │                                   │
             ▼                                   ▼
     ┌───────────────┐                   ┌───────────────┐
     │ CNN Intensity │                   │ INSAT Adapter │
     │    Model      │                   │ TIR1 + WV     │
     └───────┬───────┘                   └───────┬───────┘
             │                                   │
             ▼                                   ▼
     ┌───────────────┐                   ┌───────────────┐
     │ Temporal CNN  │                   │ NIO Fine-tune │
     │     + GRU     │                   │     Model     │
     └───────┬───────┘                   └───────┬───────┘
             │                                   │
             └─────────────────┬─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Cyclone Analysis    │
                    │                     │
                    │ • Wind              │
                    │ • Pressure          │
                    │ • Size              │
                    │ • Position          │
                    │ • Future Track      │
                    └─────────────────────┘
```

---

## 🛰️ Data Sources

### TCIR

The Tropical Cyclone Image Recognition dataset is used for initial real-satellite model development and temporal learning.

Official TCIR resource: [https://www.csie.ntu.edu.tw/~htlin/program/TCIR/](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/)

The downloaded subset contains:

- 21,076 original image records
- 10,538 unique image-time samples after duplicate removal
- 485 cyclones
  - Western Pacific
  - Atlantic
  - Eastern Pacific

The original dataset contains four satellite-image channels.

**Duplicate Cleaning**

The downloaded data contained exact duplicate (cyclone_id, timestamp) records.

After removing duplicate pairs:

| Metric | Count |
|---|---|
| Original samples | 21,076 |
| Clean samples | 10,538 |
| Cyclones | 485 |

The cleaned dataset is used for model development.

### 🛰️ INSAT-3D / MOSDAC

To adapt VayuNetra toward the North Indian Ocean (NIO), real INSAT-3D observations are being acquired through MOSDAC.

The current experimental pipeline uses:

```
INSAT-3D HDF5
      ↓
IMG_TIR1
IMG_WV
      ↓
Geolocation
      ↓
Cyclone-centered crop
      ↓
Radiance calibration
      ↓
Normalization
      ↓
NIO model
```

Current INSAT channels:

| Channel | Dataset | Current use |
|---|---|---|
| TIR1 | IMG_TIR1 | Primary thermal channel |
| WV | IMG_WV | Water-vapour information |

The current tensor contract is:

- Shape: `(2, 128, 128)`
- Dtype: `float32`
- Channel 0: TIR1 radiance
- Channel 1: WV radiance

### 🌊 North Indian Ocean Dataset

The NIO dataset is being constructed using IMD best-track information combined with temporally matched INSAT-3D observations.

Current target storms:

| Storm | Year | Basin |
|---|---|---|
| FANI | 2019 | Bay of Bengal |
| BULBUL | 2019 | Bay of Bengal |
| AMPHAN | 2020 | Bay of Bengal |
| NISARGA | 2020 | Arabian Sea |
| TAUKTAE | 2021 | Arabian Sea |
| YAAS | 2021 | Bay of Bengal |
| GULAAB | 2021 | Bay of Bengal |
| ASANI | 2022 | Bay of Bengal |
| BIPARJOY | 2023 | Arabian Sea |
| MICHAUNG | 2023 | Bay of Bengal |

This gives coverage across:

- Bay of Bengal
- Arabian Sea
- Multiple cyclone intensities
- Multiple years
- Different storm structures and tracks

### 📍 IMD Best-Track Data

IMD best-track information provides the meteorological reference labels used to supervise the satellite models.

The dataset contains:

- Timestamp
- Latitude
- Longitude
- Maximum sustained wind
- Minimum sea-level pressure
- Cyclone category

The current cleaned NIO storm collection contains:

| Metric | Value |
|---|---|
| Storms | 10 |
| Best-track records | 416 |
| Years | 2019–2023 |
| Basins | Bay of Bengal + Arabian Sea |

The satellite observations are associated with the nearest available IMD observation using a temporal matching tolerance of approximately one hour.

---

## 🔗 Satellite / Best-Track Alignment

Each matched observation stores:

- Storm
- Label timestamp
- INSAT timestamp
- Time difference
- Latitude
- Longitude
- Wind speed
- Pressure
- Pixel row
- Pixel column

The alignment pipeline is:

```
IMD Best Track
      │
      ├── Latitude
      ├── Longitude
      └── Timestamp
              │
              ▼
       INSAT timestamp match
              │
              ▼
      Lat/Lon → Pixel coordinates
              │
              ▼
       128 × 128 crop
```

The cyclone location from the IMD best track is currently used to center the satellite crop.

---

## 🧪 INSAT Preprocessing Pipeline

For each INSAT observation:

**1. HDF5 validation**

The pipeline checks for required datasets:

- IMG_TIR1
- IMG_WV
- X
- Y

Invalid or incomplete HDF5 files are skipped.

**2. Projection handling**

INSAT-3D image coordinates are interpreted using the product projection metadata. The current processing uses the INSAT Mercator projection parameters.

**3. Geographic alignment**

The IMD cyclone latitude/longitude is transformed into image coordinates.

```
Latitude / Longitude
        ↓
INSAT projection
        ↓
Pixel row / column
```

**4. Cyclone-centered crop**

A **128 × 128** region is extracted around the estimated cyclone position.

**5. Radiance calibration**

Raw digital counts are converted into radiance using the calibration coefficients stored in the INSAT product metadata.

The calibration follows:

```
R = aC² + bC + c
```

where:
- `C` = raw digital count
- `R` = calibrated radiance

**6. Normalization**

The calibrated channels are normalized before being passed to the neural network.

---

## 🧠 Machine Learning Pipeline

### Stage 1 — TCIR Intensity Model

The first model learns cyclone intensity characteristics from TCIR satellite imagery.

Architecture:

```
Satellite Image
      ↓
Conv Block
      ↓
Conv Block
      ↓
Conv Block
      ↓
Conv Block
      ↓
Adaptive Global Pooling
      ↓
Feature Layer
      ↓
 ┌────┬────────┬──────┐
 ▼    ▼        ▼
Wind Pressure Size
```

The model predicts:

- Maximum sustained wind
- Minimum sea-level pressure
- Storm size

### ⏱️ Stage 2 — Temporal Intensity Model

Cyclones evolve continuously, so a single image does not contain the complete temporal context.

VayuNetra therefore introduces a CNN + GRU temporal architecture.

```
Frame t-9h ──┐
Frame t-6h ──┤
Frame t-3h ──┼──► CNN Encoder ──► GRU ──► Intensity Heads
Frame t    ──┘
```

Current temporal configuration:

| Parameter | Value |
|---|---|
| Number of frames | 4 |
| Frame interval | ~3 hours |
| History | ~9 hours |
| CNN features | 128 |
| GRU hidden size | 128 |

Outputs: Wind, Pressure, Storm Size

---

## 📊 TCIR Temporal Model Results

Current held-out TCIR test performance:

| Target | MAE | RMSE | R² |
|---|---|---|---|
| Wind | 12.43 kt | 16.91 kt | 0.6769 |
| Pressure | 8.84 hPa | 12.86 hPa | 0.6944 |
| Size | 31.35 nmi | 44.59 nmi | 0.4606 |

Best validation checkpoint:

- Best epoch: 27
- Best val loss: 0.3087

Final test loss: **0.3354**

> These results are from the TCIR test split. They are **not** INSAT-3D/NIO validation results.

---

## 🛰️ INSAT Domain Adaptation

The TCIR model and INSAT-3D observations differ in:

- Satellite sensor
- Channel definitions
- Radiometric characteristics
- Spatial characteristics
- Geographic domain
- Storm population

Therefore, TCIR performance cannot simply be transferred to INSAT.

The current approach is:

```
TCIR
 │
 ▼
Real Satellite Pretraining
 │
 ▼
Learned Intensity Features
 │
 ▼
INSAT-3D Adapter
 │
 ▼
NIO Fine-Tuning
 │
 ▼
Held-out NIO Storm Evaluation
```

The INSAT model is being developed separately rather than treating the two satellite products as identical.

---

## 🌪️ Current Real INSAT Dataset

Initial INSAT processing has been completed for:

| Storm | Matched observations |
|---|---|
| FANI | 10 |
| TAUKTAE | 12 |
| BULBUL | 12 |

**Current processed total: 34 real INSAT observations**

These observations are being used for initial pipeline validation and domain-adaptation development. The dataset is currently being expanded with additional NIO storms.

---

## 🔬 Current Dataset Expansion

Additional acquisition is in progress for:

- AMPHAN
- NISARGA
- YAAS
- GULAAB
- ASANI
- BIPARJOY
- MICHAUNG

The objective is to increase the number of independent storm cases rather than relying on a small number of observations from only a few storms.

This is important because random image-level splits can produce overly optimistic results when frames from the same cyclone appear in both training and testing.

Therefore, the final evaluation will prioritize storm-level separation.

---

## 🧪 Planned NIO Evaluation Strategy

The final NIO evaluation will use storm-level separation.

Example:

```
TRAIN
├── FANI
├── TAUKTAE
├── AMPHAN
├── YAAS
├── ASANI
└── ...

VALIDATION
└── Held-out storm(s)

TEST
└── Completely unseen storm(s)
```

This is intended to measure whether the model generalizes to cyclone systems that were not present during training.

---

## 🗺️ Track Prediction

Track prediction is a separate development stage.

The planned target is cyclone displacement: ΔLatitude, ΔLongitude

For consecutive observations:

```
Current Position
       ↓
Temporal Model
       ↓
Predicted ΔLat / ΔLon
       ↓
Future Position
```

Track errors will ultimately be converted into geographic distance rather than relying only on degree-based errors.

Planned evaluation:

- Latitude MAE
- Longitude MAE
- Position RMSE
- Great-circle distance
- Track error in km
- Track error in km/h

The track model has not yet been treated as a validated final model.

---

## 🚫 Legacy Synthetic Models

Earlier development included synthetic-data models:

- `classifier.pt`
- `predictor.pt`

These models are retained for development history and reference.

However, synthetic-only performance should not be interpreted as real-world cyclone detection capability. In particular, out-of-distribution testing showed that the synthetic classifier can become overconfident on unrelated imagery.

Therefore: **the legacy synthetic classifier is not used as the project's validated real-world cyclone detector.** The current development focuses on real satellite datasets.

---

## 🧩 Backend Architecture

The backend is responsible for:

```
Frontend Request
      ↓
FastAPI / Backend
      ↓
Model Inference
      ↓
Prediction Processing
      ↓
JSON Response
```

The project contains separate components for:

- TCIR inference
- Temporal inference
- INSAT preprocessing
- Best-track processing
- Satellite/best-track matching
- Model training
- Dataset preparation

---

## 📁 Project Structure

```
VayuNetra/
│
├── backend/
│   │
│   ├── checkpoints/
│   │   ├── classifier_insat.pt
│   │   ├── classifier.pt
│   │   ├── predictor.pt
│   │   ├── tcir_intensity_best.pt
│   │   └── tcir_temporal_best.pt
│   │
│   ├── data/
│   │   │
│   │   ├── raw/
│   │   │   ├── tcir/
│   │   │   ├── insat/
│   │   │   └── imd_besttrack/
│   │   │
│   │   ├── labels/
│   │   │   ├── *_besttrack.csv
│   │   │   ├── *_insat_manifest.csv
│   │   │   └── tcir_track_sequences/
│   │   │
│   │   ├── processed/
│   │   │   ├── insat_fani_calibrated/
│   │   │   ├── insat_tauktae_calibrated/
│   │   │   ├── insat_bulbul_calibrated/
│   │   │   └── insat_combined/
│   │   │
│   │   └── scripts/
│   │       ├── process_insat_storm.py
│   │       ├── extract_fani_insat_crops.py
│   │       ├── extract_tauktae_insat_crops.py
│   │       ├── extract_bulbul_insat_crops.py
│   │       ├── calibrate_fani_insat.py
│   │       ├── calibrate_tauktae_insat.py
│   │       ├── calibrate_bulbul_insat.py
│   │       └── ...
│   │
│   ├── models/
│   │   ├── tcir_intensity.py
│   │   └── tcir_temporal.py
│   │
│   ├── train_tcir.py
│   ├── train_tcir_temporal.py
│   └── ...
│
├── mosdac/
│   └── api/
│       ├── mdapi.py
│       ├── bulk_nio_download.py
│       └── bulk_nio_download_v2.py
│
├── frontend/
│   └── ...
│
├── README.md
└── .gitignore
```

---

## 💾 Model Checkpoints

The repository contains the following trained checkpoints:

| File | Purpose | Status |
|---|---|---|
| `classifier_insat.pt` | INSAT-3D/NIO real-satellite cyclone intensity classifier + wind-speed regression model | Available |
| `classifier.pt` | Legacy synthetic classifier | Legacy |
| `predictor.pt` | Legacy synthetic predictor | Legacy |
| `tcir_intensity_best.pt` | TCIR real-satellite intensity model | Available |
| `tcir_temporal_best.pt` | TCIR CNN + GRU temporal model | Available |
| `insat_intensity_preliminary.pt` | INSAT/NIO intensity adaptation | In development |

Large raw datasets are intentionally not stored in the Git repository.

---

## 🔐 Data & Credentials

MOSDAC authentication information is stored locally and must never be committed.

For example, `mosdac/api/config.json` is excluded from Git.

**Never commit:**

- Passwords
- API credentials
- Tokens
- Private keys
- Large raw datasets

---

## ⚙️ Technologies

**Machine Learning**
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn

**Computer Vision**
- CNNs
- Image preprocessing
- Satellite-image cropping
- Geospatial coordinate transformation

**Temporal Modeling**
- GRU
- Sequence modeling
- Multi-frame cyclone analysis

**Satellite Data**
- INSAT-3D
- MOSDAC
- TCIR
- HDF5

**Meteorological Data**
- IMD best-track observations

**Backend**
- Python
- FastAPI
- REST APIs

**Frontend**
- React
- JavaScript
- HTML
- CSS

---

## 🚀 Development Roadmap

### Phase 1 — Dataset & Baseline
- [x] Obtain TCIR data
- [x] Inspect TCIR structure
- [x] Remove exact duplicates
- [x] Train initial intensity model
- [x] Evaluate wind/pressure/size
- [x] Build temporal model

### Phase 2 — INSAT Integration
- [x] Obtain MOSDAC access
- [x] Download INSAT-3D products
- [x] Inspect HDF5 structure
- [x] Understand projection metadata
- [x] Implement geographic alignment
- [x] Implement cyclone-centered cropping
- [x] Implement radiance calibration
- [x] Build storm-level manifests
- [x] Process initial FANI observations
- [x] Process initial TAUKTAE observations
- [x] Process initial BULBUL observations

### Phase 3 — NIO Dataset Expansion
- [x] Build 10-storm IMD best-track collection
- [x] Establish NIO storm list
- [ ] Build automated MOSDAC acquisition pipeline
- [x] Expand INSAT observations for remaining storms
- [ ] Audit all downloaded products
- [ ] Remove corrupt/incomplete files
- [ ] Generate final NIO dataset

### Phase 4 — NIO Model
- [x] Build training/validation/test splits
- [ ] Establish baseline
- [ ] Fine-tune using INSAT-3D observations
- [ ] Evaluate on held-out storms
- [ ] Compare against baseline
- [ ] Analyze failure cases

### Phase 5 — Track Prediction
- [ ] Generate temporal track sequences
- [ ] Train displacement model
- [ ] Predict Δlatitude / Δlongitude
- [ ] Convert predictions to geographic positions
- [ ] Calculate great-circle errors
- [ ] Evaluate on held-out storms

### Phase 6 — End-to-End System
- [ ] Integrate INSAT inference into backend
- [ ] Add temporal prediction endpoint
- [ ] Add track prediction endpoint
- [ ] Build visualization
- [ ] Add storm trajectory map
- [ ] Add intensity timeline
- [ ] Add model confidence / uncertainty where validated
- [ ] Final system evaluation

---

## 📊 Evaluation Philosophy

VayuNetra separates:

**Training performance** — Performance measured on data used to optimize the model.

**Validation performance** — Performance used for model selection and hyperparameter tuning.

**Test performance** — Performance on data withheld from training.

**Storm-level generalization** — Performance on an entire cyclone that was not represented in training.

The final system will prioritize the last category for NIO evaluation.

---

## ⚠️ Current Limitations

**1. Limited NIO dataset**
The INSAT dataset is still being expanded. The initial real INSAT set contains only 34 matched observations across three storms.

**2. Domain gap**
TCIR and INSAT-3D are different satellite products. Their channels cannot simply be treated as identical.

**3. Temporal availability**
INSAT observations and IMD best-track observations may not occur at exactly the same timestamp. The current pipeline therefore uses temporal matching.

**4. Best-track-centered crops**
The current crop location is derived from the IMD best-track position. This is useful for supervised development but does not constitute an independent cyclone localization system.

**5. Track prediction**
The track model is still under development.

**6. Real-world detection**
A validated cyclone-presence detector using real INSAT-3D data has not yet been established.

**7. Operational deployment**
The current system should not be considered an operational meteorological forecasting system.

---

## 📚 References

**TCIR**
- Tropical Cyclone Image Recognition: [https://www.csie.ntu.edu.tw/~htlin/program/TCIR/](https://www.csie.ntu.edu.tw/~htlin/program/TCIR/)

**TCRISI**
- Temporal cyclone intensity estimation using satellite imagery: [https://www.csie.ntu.edu.tw/~htlin/paper/doc/ecml20tcrisi.pdf](https://www.csie.ntu.edu.tw/~htlin/paper/doc/ecml20tcrisi.pdf)

**MOSDAC**
- INSAT-3D: [https://mosdac.gov.in/insat-3d](https://mosdac.gov.in/insat-3d)
- INSAT-3DS: [https://mosdac.gov.in/insat-3ds](https://mosdac.gov.in/insat-3ds)
- INSAT-3D L1C Asia MER: [https://mosdac.gov.in/doi/123/](https://mosdac.gov.in/doi/123/)
- INSAT-3D product documentation: [https://mosdac.gov.in/docs/INSAT3D_Products.pdf](https://mosdac.gov.in/docs/INSAT3D_Products.pdf)
- MOSDAC API: [https://mosdac.gov.in/downloadapi-manual](https://mosdac.gov.in/downloadapi-manual)
- MOSDAC FAQ: [https://mosdac.gov.in/faq-page](https://mosdac.gov.in/faq-page)
- Cyclone information: [https://www.mosdac.gov.in/cyclone](https://www.mosdac.gov.in/cyclone)

---

## 👥 Project

**VayuNetra**

Developed for: **Smart India Hackathon 2026 — SIH26070**

Problem statement: Cyclone Identification / Classification / Prediction

Organization: Ministry of Earth Sciences (MoES)

---

## 📌 Current Status

```
┌─────────────────────────────────────────────┐
│             VAYUNETRA STATUS                │
├─────────────────────────────────────────────┤
│ TCIR Dataset             COMPLETE           │
│ TCIR Intensity Model     COMPLETE           │
│ TCIR Temporal Model      COMPLETE           │
│ IMD NIO Best Tracks      COMPLETE           │
│ MOSDAC Integration       COMPLETE           │
│ INSAT Calibration        COMPLETE           │
│ Initial NIO Samples      34                 │
│ NIO Expansion            IN PROGRESS        │
│ INSAT Fine-Tuning        NEXT               │
│ Track Prediction         NEXT               │
│ Final NIO Evaluation     PENDING            │
│ Operational Validation   PENDING            │
└─────────────────────────────────────────────┘
```

---

## 🔥 Core Principle

**Real data first. Honest evaluation always.**

VayuNetra distinguishes between synthetic experiments, TCIR-based pretraining, and real INSAT-3D/NIO validation rather than presenting results from one dataset as performance on another.

The ultimate goal is to develop a reproducible satellite-based cyclone analysis pipeline capable of learning from historical NIO cyclone systems and generalizing to previously unseen storms.
