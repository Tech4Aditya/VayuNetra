# VayuNetra 🌪️

### Cyclone Identification, Intensity Analysis & Future Track Prediction using Satellite Imagery

**Smart India Hackathon 2026 — SIH26070** **Ministry of Earth Sciences (MoES)**

---

## 📌 Overview

**VayuNetra** is a research-oriented cyclone analysis system designed to
process satellite imagery and meteorological best-track information for
tropical cyclone analysis.

The project combines:

-   🛰️ INSAT-3D satellite observations
-   🌊 Tropical Cyclone Image Recognition (TCIR) data
-   📍 IMD best-track observations
-   🧠 Deep learning
-   ⏱️ Temporal sequence modeling
-   🗺️ Cyclone position tracking
-   ⚡ Real-time inference through a backend API

The development strategy follows a staged approach:

``` text
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

> **Important:** VayuNetra is currently a research prototype. Claims of
> operational cyclone detection, forecasting, or prediction are not made
> until the corresponding real-world evaluation is completed.

---

## 🎯 Problem Statement

Cyclone monitoring requires the analysis of rapidly changing atmospheric
systems using satellite observations and meteorological information.

Traditional workflows can involve:

-   Manual satellite-image interpretation
-   Separate analysis of historical observations
-   Delayed intensity estimation
-   Limited temporal modeling
-   Difficulty integrating heterogeneous satellite products

VayuNetra aims to develop a unified machine-learning pipeline capable of
learning cyclone characteristics from historical satellite observations
and eventually providing:

-   Cyclone intensity estimation
-   Temporal intensity analysis
-   Cyclone position/track prediction
-   Satellite-image based cyclone analysis
-   NIO-specific domain adaptation

---

## 🧠 System Architecture

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
                        ┌─────────────────── ────────────┐
                        │ Cyclone Analysis               │
                        │                                │
                        │ • Wind                         │
                        │ • Pressure                     │
                        │ • Size                         │
                        │ • Position                     │
                        │ • Short-Horizon Track Forecast │
                        └────────────────────────────────┘

---

## 🛰️ Data Sources

### TCIR

The Tropical Cyclone Image Recognition dataset is used for initial
real-satellite model development and temporal learning.

Official TCIR resource:
<https://www.csie.ntu.edu.tw/~htlin/program/TCIR/>

The downloaded subset contains:

-   21,076 original image records
-   10,538 unique image-time samples after duplicate removal
-   485 cyclones
    -   Western Pacific
    -   Atlantic
    -   Eastern Pacific

The original dataset contains four satellite-image channels.

**Duplicate Cleaning**

The downloaded data contained exact duplicate (cyclone_id, timestamp)
records.

After removing duplicate pairs:

| Metric | Count |
|---|---|
| Original samples | 21,076 |
| Clean samples | 10,538 |
| Cyclones | 485 |

The cleaned dataset is used for model development.

### 🛰️ INSAT-3D / MOSDAC

To adapt VayuNetra toward the North Indian Ocean (NIO), real INSAT-3D
observations are being acquired through MOSDAC.

The current experimental pipeline uses:

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

Current INSAT channels:

| Channel | Dataset | Current use |
|---|---|---|
| TIR1 | IMG_TIR1 | Primary thermal channel |
| WV | IMG_WV | Water-vapour information |

The current tensor contract is:

-   Shape: `(2, 128, 128)`
-   Dtype: `float32`
-   Channel 0: TIR1 radiance
-   Channel 1: WV radiance

### 🌊 North Indian Ocean Dataset

The NIO dataset is being constructed using IMD best-track information
combined with temporally matched INSAT-3D observations.

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

-   Bay of Bengal
-   Arabian Sea
-   Multiple cyclone intensities
-   Multiple years
-   Different storm structures and tracks

### 📍 IMD Best-Track Data

IMD best-track information provides the meteorological reference labels
used to supervise the satellite models.

The dataset contains:

-   Timestamp
-   Latitude
-   Longitude
-   Maximum sustained wind
-   Minimum sea-level pressure
-   Cyclone category

The current cleaned NIO storm collection contains:

| Metric | Value |
|---|---|
| Storms | 10 |
| Best-track records | 416 |
| Years | 2019–2023 |
| Basins | Bay of Bengal + Arabian Sea |

The satellite observations are associated with the nearest available IMD
observation using a temporal matching tolerance of approximately one
hour.

---

## 🔗 Satellite / Best-Track Alignment

Each matched observation stores:

-   Storm
-   Label timestamp
-   INSAT timestamp
-   Time difference
-   Latitude
-   Longitude
-   Wind speed
-   Pressure
-   Pixel row
-   Pixel column

The alignment pipeline is:

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

The cyclone location from the IMD best track is currently used to center
the satellite crop.

---

## 🧪 INSAT Preprocessing Pipeline

For each INSAT observation:

**1. HDF5 validation**

The pipeline checks for required datasets:

-   IMG_TIR1
-   IMG_WV
-   X
-   Y

Invalid or incomplete HDF5 files are skipped.

**2. Projection handling**

INSAT-3D image coordinates are interpreted using the product projection
metadata. The current processing uses the INSAT Mercator projection
parameters.

**3. Geographic alignment**

The IMD cyclone latitude/longitude is transformed into image
coordinates.

    Latitude / Longitude
            ↓
    INSAT projection
            ↓
    Pixel row / column

**4. Cyclone-centered crop**

A **128 × 128** region is extracted around the estimated cyclone
position.

**5. Radiance calibration**

Raw digital counts are converted into radiance using the calibration
coefficients stored in the INSAT product metadata.

The calibration follows:

    R = aC² + bC + c

where: - `C` = raw digital count - `R` = calibrated radiance

**6. Normalization**

The calibrated channels are normalized before being passed to the neural
network.

---

## 🧠 Machine Learning Pipeline

### Stage 1— TCIR Intensity Model

The first model learns cyclone intensity characteristics from TCIR
satellite imagery.

Architecture:

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

The model predicts:

-   Maximum sustained wind
-   Minimum sea-level pressure
-   Storm size

### ⏱️ Stage 2— Temporal Intensity Model

Cyclones evolve continuously, so a single image does not contain the
complete temporal context.

VayuNetra therefore introduces a CNN + GRU temporal architecture.

    Frame t-9h ──┐
    Frame t-6h ──┤
    Frame t-3h ──┼──► CNN Encoder ──► GRU ──► Intensity Heads
    Frame t    ──┘

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

## 📊 TCIR Temporal Model

The TCIR temporal model is a CNN + GRU model operating on four
chronological satellite frames separated by approximately three hours.
The production API exposes this model for real TCIR
demonstration/inference.

The current backend also exposes the TCIR single-frame intensity model
separately. The dashboard therefore distinguishes temporal analysis from
single-frame intensity inference.

> Evaluation numbers shown in the web console are served from the
> project's local evaluation artifacts. This README intentionally does
> not combine results from different checkpoints or test configurations
> into one headline number.

---

## 🛰️ INSAT Domain Adaptation

The TCIR model and INSAT-3D observations differ in:

-   Satellite sensor
-   Channel definitions
-   Radiometric characteristics
-   Spatial characteristics
-   Geographic domain
-   Storm population

Therefore, TCIR performance cannot simply be transferred to INSAT.

The current approach is:

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

The INSAT model is being developed separately rather than treating the
two satellite products as identical.

---

## 🌪️ Current Real INSAT Dataset

The NIO INSAT pipeline has now been expanded to **10 target storms**:

| Storm | Matched INSAT observations |
|---|---|
| FANI | 61 |
| TAUKTAE | 12 |
| AMPHAN | 17 |
| YAAS | 13 |
| ASANI | 19 |
| NISARGA | 9 |
| BULBUL | 12 |
| GULAAB | 10 |
| BIPARJOY | 39 |
| MICHAUNG | 10 |
| **Total** | **202** |

All ten storm manifests passed the project audit: required fields are
present, calibrated TIR1/WV pairs are available for the recorded
samples, and no missing best-track latitude/longitude/time/wind/pressure
fields were reported.

The final calibrated INSAT dataset is therefore **202 matched
observations across 10 NIO storms**.

---

## 🔬 NIO Dataset Status

The planned 10-storm NIO collection has been acquired and audited:

-   FANI
-   TAUKTAE
-   AMPHAN
-   YAAS
-   ASANI
-   NISARGA
-   BULBUL
-   GULAAB
-   BIPARJOY
-   MICHAUNG

The resulting dataset contains **202 calibrated INSAT observations**
linked to IMD best-track labels.

The dataset is split at storm level for INSAT model evaluation. Random
image-level splitting is avoided because frames from the same cyclone
can otherwise appear in both training and testing.

---

## 🧪 NIO Evaluation Strategy

The INSAT modeling work uses storm-level separation and a fixed
five-fold cross-validation protocol. Each fold holds out two complete
storms for testing, while the remaining storms are divided
deterministically into training and validation groups.

A same-fold majority-class baseline was also computed for comparison.

The INSAT modeling phase is now **frozen**. The results are treated as
research evaluation evidence, not as an operational seven-class cyclone
classifier.

Example:

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

This is intended to measure whether the model generalizes to cyclone
systems that were not present during training.

---

## 🗺️ Track Prediction

Track prediction is implemented as a separate trajectory-modeling
component using IMD best-track sequences.

The current learned model predicts short-horizon displacement from
historical track information and provides:

-   +3 hour forecast
-   +6 hour forecast
-   +9 hour forecast

The dashboard displays the observed IMD best track together with the
learned GRU forecast.

The track model is evaluated on held-out storm sequences; it is **not**
a satellite-image-to-track model.

For consecutive observations:

    Current Position
           ↓
    Temporal Model
           ↓
    Predicted ΔLat / ΔLon
           ↓
    Future Position

Track errors will ultimately be converted into geographic distance
rather than relying only on degree-based errors.

Planned evaluation:

-   Latitude MAE
-   Longitude MAE
-   Great-circle distance
-   Track error in km

The track model is integrated and evaluated as a research prototype. Its
forecast is presented separately from the observed IMD track and should
not be interpreted as an operational meteorological forecast.

---

## 🚫 Legacy Synthetic Models

Earlier development included synthetic-data models:

-   `classifier.pt`
-   `predictor.pt`

These models are retained for development history and reference.

However, synthetic-only performance should not be interpreted as
real-world cyclone detection capability. In particular,
out-of-distribution testing showed that the synthetic classifier can
become overconfident on unrelated imagery.

Therefore: **the legacy synthetic classifier is not used as the
project's validated real-world cyclone detector.** The current
development focuses on real satellite datasets.

---

## 🧩 Backend Architecture

The backend is responsible for:

    Frontend Request
          ↓
    FastAPI / Backend
          ↓
    Model Inference
          ↓
    Prediction Processing
          ↓
    JSON Response

The project contains separate components for:

-   TCIR inference
-   Temporal inference
-   INSAT preprocessing
-   Best-track processing
-   Satellite/best-track matching
-   Model training
-   Dataset preparation

---

## 🖥️ End-to-End Application

VayuNetra is now integrated as a browser-based local inference console.

The application provides:

-   Login / console entry
-   TCIR live analysis
-   TCIR temporal prediction
-   TCIR single-frame intensity prediction
-   INSAT real-data analysis
-   Local TIR1 and water-vapour preview rendering
-   Observed IMD track visualization
-   GRU +3h / +6h / +9h track forecasts
-   Data-source status
-   Evaluation / metrics views
-   Synthetic pipeline validation with an explicit validation-only
    warning
-   Backend health and API integration

The frontend communicates with the FastAPI backend through REST
endpoints. The current frontend is plain HTML/CSS/JavaScript rather than
React.

### API surface

``` text
GET  /health
GET  /data_sources
GET  /demo_sequence
GET  /tcir_demo
GET  /insat_demo
GET  /metrics
GET  /evaluation_summary
GET  /track
GET  /insat_preview/tir1
GET  /insat_preview/wv
POST /predict
POST /predict_tcir_sequence
POST /predict_tcir_single
POST /predict_insat
POST /predict_image
```

A smoke test currently passes the core health, source, demo, synthetic
prediction, INSAT preview, metrics, evaluation-summary, and track
endpoints.

---

## 📁 Project Structure

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

---

## 💾 Model Checkpoints

The current project includes the following relevant checkpoints:

| File | Purpose | Status |
|---|---|---|
| `tcir_intensity_best.pt` | TCIR single-frame intensity model | Available |
| `tcir_temporal_best.pt` | TCIR CNN + GRU temporal intensity model | Available |
| `insat_v2_best.pt` | INSAT real-data multitask inference model used by the current API | Available / frozen |
| `insat_v3_ordinal_best.pt` | INSAT ordinal-intensity research model used for fixed-split evaluation | Frozen evaluation artifact |
| `track_gru_multi_best.pt` | Multi-horizon GRU track model | Available / integrated |
| `classifier.pt` | Legacy synthetic classifier | Legacy |
| `predictor.pt` | Legacy synthetic predictor | Legacy |

Additional evaluation artifacts are stored under
`backend/data/processed/`, including INSAT V3 cross-validation and
baseline results.

The current API intentionally uses the **INSAT V2 checkpoint** for
uploaded/local INSAT inference, while V3 is retained as the frozen
research-evaluation artifact.

Large raw datasets are intentionally not stored in the Git repository.

---

## 🔐 Data & Credentials

MOSDAC authentication information is stored locally and must never be
committed.

For example, `mosdac/api/config.json` is excluded from Git.

**Never commit:**

-   Passwords
-   API credentials
-   Tokens
-   Private keys
-   Large raw datasets

---

## ⚙️ Technologies

**Machine Learning** - Python - PyTorch - NumPy - Pandas - Scikit-learn

**Computer Vision** - CNNs - Image preprocessing - Satellite-image
cropping - Geospatial coordinate transformation

**Temporal Modeling** - GRU - Sequence modeling - Multi-frame cyclone
analysis

**Satellite Data** - INSAT-3D - MOSDAC - TCIR - HDF5

**Meteorological Data** - IMD best-track observations

**Backend** - Python - FastAPI - REST APIs

**Frontend** - JavaScript - HTML - CSS - Leaflet.js - OpenStreetMap

---

## 🚀 Development Roadmap

### Phase 1— Dataset & Baseline

-   [x] Obtain TCIR data
-   [x] Inspect TCIR structure
-   [x] Remove exact duplicates
-   [x] Train initial intensity model
-   [x] Evaluate wind/pressure/size
-   [x] Build temporal model

### Phase 2— INSAT Integration

-   [x] Obtain MOSDAC access
-   [x] Download INSAT-3D products
-   [x] Inspect HDF5 structure
-   [x] Understand projection metadata
-   [x] Implement geographic alignment
-   [x] Implement cyclone-centered cropping
-   [x] Implement radiance calibration
-   [x] Build storm-level manifests
-   [x] Process initial FANI observations
-   [x] Process initial TAUKTAE observations
-   [x] Process initial BULBUL observations

### Phase 3— NIO Dataset Expansion

-   [x] Build 10-storm IMD best-track collection
-   [x] Establish NIO storm list
-   [x] Acquire INSAT observations for all target storms
-   [x] Calibrate and crop INSAT observations
-   [x] Generate storm-level manifests
-   [x] Audit the complete NIO dataset
-   [x] Generate the 202-sample calibrated dataset

### Phase 4— NIO Model

-   [x] Build storm-level training/validation/test splits
-   [x] Establish same-fold majority baseline
-   [x] Train INSAT V2 multitask model
-   [x] Train INSAT V3 ordinal model
-   [x] Run fixed-split evaluation
-   [x] Run five-fold unseen-storm cross-validation
-   [x] Analyze failure cases
-   [x] Freeze INSAT modeling rather than continuing architecture tuning

### Phase 5— Track Prediction

-   [x] Generate temporal track sequences (storm-consistent
    train/val/test splits with Δlatitude/Δlongitude targets already
    built)
-   [x] Train multi-horizon displacement model on real track sequences
-   [x] Predict Δlatitude / Δlongitude from a trained model
-   [x] Convert predictions to geographic positions
-   [x] Calculate great-circle errors
-   [x] Evaluate on held-out storms
-   [x] Constant-velocity motion baseline integrated as an interim
    +3h / +6h / +9h projection (explicitly labeled non-neural in the
    API response, not a trained model's output)

### Phase 6— End-to-End System

-   [x] Integrate INSAT inference into backend
-   [x] Add temporal prediction endpoint
-   [x] Add track prediction endpoint
-   [x] Build browser visualization
-   [x] Add storm trajectory map
-   [x] Add satellite TIR1/WV previews
-   [x] Add metrics and evaluation views
-   [x] Add synthetic pipeline validation with explicit labeling
-   [x] Run backend smoke tests
-   [x] Run full browser E2E test
-   [ ] Prepare final reports / documentation
-   [ ] Prepare final presentation

> The project is currently in the **documentation and presentation
> phase**. Further model architecture changes are intentionally frozen.

---

## 📊 Evaluation Philosophy

VayuNetra separates:

**Training performance**— Performance measured on data used to
optimize the model.

**Validation performance**— Performance used for model selection and
hyperparameter tuning.

**Test performance**— Performance on data withheld from training.

**Storm-level generalization**— Performance on an entire cyclone that
was not represented in training.

The final system will prioritize the last category for NIO evaluation.

---

## ⚠️ Current Limitations

**1. Limited real INSAT sample size**

The final calibrated NIO collection contains **202 observations across
10 storms**. This is substantially broader than the initial dataset, but
still small for training a robust operational satellite classifier.

**2. Domain gap**

TCIR and INSAT-3D are different satellite products. Their channels,
radiometric characteristics and spatial properties differ.

**3. Best-track-centered crops**

The INSAT crop is centered using IMD best-track coordinates. This is
appropriate for supervised satellite-intensity analysis but does not
constitute independent cyclone localization.

**4. INSAT generalization**

Five-fold unseen-storm evaluation shows that the INSAT model learns some
signal relative to the same-fold majority baseline, but absolute
seven-class classification performance remains weak. INSAT is therefore
presented as an independent satellite observation/intensity-analysis
component rather than as a validated operational classifier.

**5. INSAT presence detection**

The real INSAT dataset is cyclone-positive. The current INSAT presence
head is therefore not used as a no-cyclone detector.

**6. Track prediction scope**

The GRU track model forecasts from historical best-track sequences. It
is not a satellite-image-to-track model and should not be interpreted as
an operational forecast.

**7. Synthetic validation**

Synthetic data is used only to verify the software/model pipeline.
Synthetic results are not evidence of real-world cyclone performance.

**8. Operational deployment**

VayuNetra is a research prototype and should not be considered an
operational meteorological forecasting or warning system.

---

## 📚 References

**TCIR** - Tropical Cyclone Image Recognition:
<https://www.csie.ntu.edu.tw/~htlin/program/TCIR/>

**TCRISI** - Temporal cyclone intensity estimation using satellite
imagery: <https://www.csie.ntu.edu.tw/~htlin/paper/doc/ecml20tcrisi.pdf>

**MOSDAC** - INSAT-3D: <https://mosdac.gov.in/insat-3d> - INSAT-3DS:
<https://mosdac.gov.in/insat-3ds> - INSAT-3D L1C Asia MER:
<https://mosdac.gov.in/doi/123/> - INSAT-3D product documentation:
<https://mosdac.gov.in/docs/INSAT3D_Products.pdf> - MOSDAC API:
<https://mosdac.gov.in/downloadapi-manual> - MOSDAC FAQ:
<https://mosdac.gov.in/faq-page> - Cyclone information:
<https://www.mosdac.gov.in/cyclone>

---

## 👥 Project

**VayuNetra**

Developed for: **Smart India Hackathon 2026— SIH26070**

Problem statement: Cyclone Identification / Classification / Prediction

Organization: Ministry of Earth Sciences (MoES)

---

## 📌 Current Status

``` text
┌─────────────────────────────────────────────┐
│             VAYUNETRA STATUS                │
├─────────────────────────────────────────────┤
│ TCIR Dataset                  COMPLETE      │
│ TCIR Intensity Model          COMPLETE      │
│ TCIR Temporal Model           COMPLETE      │
│ IMD NIO Best Tracks            COMPLETE     │
│ MOSDAC Integration             COMPLETE     │
│ INSAT Calibration              COMPLETE     │
│ NIO Storm Collection           10 STORMS    │
│ Calibrated INSAT Samples       202          │
│ INSAT V2 Inference              COMPLETE    │
│ INSAT Baseline Comparison       COMPLETE    │
│ Track GRU (real data)           NOT STARTED │
│ Track Constant-Velocity Interim COMPLETE    │
│ Backend Integration             COMPLETE    │
│ Frontend E2E Testing            COMPLETE    │
│ Documentation                   IN PROGRESS │
│ Presentation                    NEXT        │
└─────────────────────────────────────────────┘
```

Intensity modeling and backend/frontend integration are complete. The
real-data track GRU is the one component in the original plan that has
not been trained yet — the dashboard currently shows a constant-velocity
projection, honestly labeled as such in the API response. Remaining
work: train and evaluate the track GRU on the existing
`tcir_track_sequences` splits, then documentation and presentation.

---

## 🔥 Core Principle

**Real data first. Honest evaluation always.**

VayuNetra distinguishes between synthetic experiments, TCIR-based
pretraining, and real INSAT-3D/NIO validation rather than presenting
results from one dataset as performance on another.

The ultimate goal is to develop a reproducible satellite-based cyclone
analysis pipeline capable of learning from historical NIO cyclone
systems and generalizing to previously unseen storms.