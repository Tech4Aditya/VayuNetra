"""
VayuNetra — Cyclone Intelligence Console backend.

Real model/data paths used by this console:
- TCIR HDF5 temporal sequence + temporal checkpoint
- TCIR single-frame intensity checkpoint
- INSAT real-data classifier checkpoint
- Existing synthetic pipeline (kept explicitly as validation/demo only)

Run:
    python -m uvicorn backend.app:app --reload --port 8000
"""

import io
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
DATA_DIR = ROOT / "data"

TCIR_H5 = DATA_DIR / "raw" / "tcir" / "Cyclone_Images.h5"
TCIR_TEST = DATA_DIR / "labels" / "tcir_temporal_test.csv"
TCIR_TEMPORAL_CKPT = CKPT_DIR / "tcir_temporal_best.pt"
TCIR_INTENSITY_CKPT = CKPT_DIR / "tcir_intensity_best.pt"

INSAT_V2_CKPT = CKPT_DIR / "insat_v2_best.pt"
INSAT_CKPT = CKPT_DIR / "classifier_insat.pt"  # legacy fallback
INSAT_PROCESSED = DATA_DIR / "processed"

SYNTH_CLASSIFIER_CKPT = CKPT_DIR / "classifier.pt"
SYNTH_PREDICTOR_CKPT = CKPT_DIR / "predictor.pt"
TRACK_MULTI_CKPT = CKPT_DIR / "track_gru_multi_best.pt"

EVAL_DIR = DATA_DIR / "evaluation"

sys.path.insert(0, str(ROOT))

from models.classifier import CycloneCNN
from models.predictor import CycloneTrendLSTM
from models.tcir_temporal import TCIRTemporalModel
from models.tcir_intensity import TCIRIntensityCNN
from models.insat_complete import INSATMultiTask
from data.generate_synthetic import make_sequence, CATEGORIES
from models.track_gru_multi import TrackGRUMulti

TREND_LABELS = {0: "Weakening", 1: "Steady", 2: "Intensifying"}
TCIR_CHANNEL_INDEX = {"IR": 0, "WV": 1, "VIS": 2, "PMW": 3}

app = FastAPI(title="VayuNetra — Cyclone Intelligence Console")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = None
predictor = None
tcir_temporal_model = None
tcir_temporal_checkpoint = {}
tcir_intensity_model = None
tcir_intensity_checkpoint = {}
insat_model = None
insat_checkpoint = {}
track_multi_model = None
track_multi_checkpoint = {}
model_errors: dict[str, str] = {}


def _load_state(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj if isinstance(obj, dict) else {"model_state_dict": obj}


def _safe_load_models():
    global classifier, predictor
    global tcir_temporal_model, tcir_temporal_checkpoint
    global tcir_intensity_model, tcir_intensity_checkpoint
    global insat_model, insat_checkpoint
    global track_multi_model, track_multi_checkpoint

    # Existing synthetic pipeline.
    try:
        classifier = CycloneCNN()
        ck = _load_state(SYNTH_CLASSIFIER_CKPT)
        classifier.load_state_dict(ck["model_state_dict"], strict=True)
        classifier.eval()
        predictor = CycloneTrendLSTM()
        pk = _load_state(SYNTH_PREDICTOR_CKPT)
        predictor.load_state_dict(pk["model_state_dict"], strict=True)
        predictor.eval()
    except Exception as exc:
        model_errors["synthetic"] = str(exc)
        classifier = None
        predictor = None

    # Real TCIR temporal model.
    try:
        tcir_temporal_checkpoint = _load_state(TCIR_TEMPORAL_CKPT)
        channels = tuple(tcir_temporal_checkpoint.get("channels", ("IR", "PMW")))
        tcir_temporal_model = TCIRTemporalModel(in_channels=len(channels))
        tcir_temporal_model.load_state_dict(
            tcir_temporal_checkpoint["model_state_dict"], strict=True
        )
        tcir_temporal_model.eval()
    except Exception as exc:
        model_errors["tcir_temporal"] = str(exc)
        tcir_temporal_model = None
        tcir_temporal_checkpoint = {}

    # Real TCIR single-frame model.
    try:
        tcir_intensity_checkpoint = _load_state(TCIR_INTENSITY_CKPT)
        channels = tuple(tcir_intensity_checkpoint.get("channels", ("IR", "PMW")))
        tcir_intensity_model = TCIRIntensityCNN(in_channels=len(channels))
        tcir_intensity_model.load_state_dict(
            tcir_intensity_checkpoint["model_state_dict"], strict=True
        )
        tcir_intensity_model.eval()
    except Exception as exc:
        model_errors["tcir_intensity"] = str(exc)
        tcir_intensity_model = None
        tcir_intensity_checkpoint = {}

    # Real INSAT classifier. Prefer the corrected V2 checkpoint with
    # training-set normalization and weighted classification loss.
    try:
        insat_path = INSAT_V2_CKPT if INSAT_V2_CKPT.exists() else INSAT_CKPT
        insat_checkpoint = _load_state(insat_path)
        categories = insat_checkpoint["category_names"]
        if insat_path == INSAT_V2_CKPT:
            insat_model = INSATMultiTask(
                in_channels=int(insat_checkpoint.get("in_channels", 2)),
                n_classes=len(categories),
                predict_size=bool(insat_checkpoint.get("predict_size", False)),
            )
            insat_model.load_state_dict(insat_checkpoint["model_state_dict"], strict=True)
            insat_checkpoint["model_version"] = "V2"
        else:
            insat_model = CycloneCNN(
                num_categories=len(categories),
                in_channels=int(insat_checkpoint.get("in_channels", 2)),
            )
            insat_model.load_state_dict(insat_checkpoint["model_state_dict"], strict=True)
            insat_checkpoint["model_version"] = "legacy"
        insat_checkpoint["checkpoint_path"] = str(insat_path)
        insat_model.eval()
    except Exception as exc:
        model_errors["insat"] = str(exc)
        insat_model = None
        insat_checkpoint = {}

    # Learned multi-horizon trajectory model.
    try:
        track_multi_checkpoint = _load_state(TRACK_MULTI_CKPT)
        track_multi_model = TrackGRUMulti(
            in_features=int(track_multi_checkpoint.get("input_features", 5)),
            horizons=len(track_multi_checkpoint.get("horizons", [3, 6, 9])),
        )
        track_multi_model.load_state_dict(track_multi_checkpoint["model_state_dict"], strict=True)
        track_multi_model.eval()
    except Exception as exc:
        model_errors["track_multi"] = str(exc)
        track_multi_model = None
        track_multi_checkpoint = {}



_safe_load_models()


class FramesIn(BaseModel):
    frames: list


def imd_category(wind_kt: float) -> str:
    if wind_kt < 17:
        return "Low Pressure Area"
    if wind_kt <= 27:
        return "Depression"
    if wind_kt <= 33:
        return "Deep Depression"
    if wind_kt <= 47:
        return "Cyclonic Storm"
    if wind_kt <= 63:
        return "Severe Cyclonic Storm"
    if wind_kt <= 90:
        return "Very Severe Cyclonic Storm"
    if wind_kt <= 119:
        return "Extremely Severe Cyclonic Storm"
    return "Super Cyclonic Storm"


def _denorm(value: float, stats: Any, name: str) -> float:
    cfg = stats.get(name, (0.0, 1.0))
    if isinstance(cfg, dict):
        mean, std = cfg.get("mean", 0.0), cfg.get("std", 1.0)
    else:
        mean, std = cfg
    return float(value) * float(std) + float(mean)


def _tcir_tensor(frames: list, channels=("IR", "PMW")) -> torch.Tensor:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 4 or arr.shape != (4, 128, 128, 4):
        raise ValueError(
            f"Expected 4 frames of 128x128x4; received {arr.shape}"
        )
    selected = []
    for ch in channels:
        selected.append(arr[..., TCIR_CHANNEL_INDEX[ch]])
    arr = np.stack(selected, axis=1)
    if np.nanmax(arr) > 1.0:
        arr = arr / 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.from_numpy(arr.copy()).unsqueeze(0).float()


def _tcir_single_tensor(frame: list, channels=("IR", "PMW")) -> torch.Tensor:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.shape != (128, 128, 4):
        raise ValueError(f"Expected one 128x128x4 frame; received {arr.shape}")
    selected = [arr[..., TCIR_CHANNEL_INDEX[ch]] for ch in channels]
    arr = np.stack(selected, axis=0)
    if np.nanmax(arr) > 1.0:
        arr = arr / 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.from_numpy(arr.copy()).unsqueeze(0).float()


def _find_insat_sample():
    candidates = sorted(INSAT_PROCESSED.glob("**/tir1_radiance.npy"))
    for tir_path in candidates:
        wv_path = tir_path.parent / "wv_radiance.npy"
        if wv_path.exists():
            return tir_path, wv_path
    return None, None


def _insat_predict(tir: np.ndarray, wv: np.ndarray):
    if insat_model is None:
        raise RuntimeError("INSAT real-data model is not loaded.")
    tir = np.asarray(tir, dtype=np.float32)
    wv = np.asarray(wv, dtype=np.float32)
    if tir.shape != (128, 128) or wv.shape != (128, 128):
        raise ValueError(f"INSAT expects two 128x128 arrays; got {tir.shape} and {wv.shape}")

    version = insat_checkpoint.get("model_version", "legacy")
    category_names = insat_checkpoint["category_names"]

    if version == "V2":
        st = insat_checkpoint["stats"]
        x = np.stack([
            (tir - float(st["tir_mean"])) / float(st["tir_std"]),
            (wv - float(st["wv_mean"])) / float(st["wv_std"]),
        ], axis=0)
        x = torch.from_numpy(x).unsqueeze(0).float()
        with torch.no_grad():
            out = insat_model(x)
        probs = torch.softmax(out["category"], dim=1)[0].cpu().numpy()
        category_index = int(np.argmax(probs))
        wind = float(out["wind"].item()) * float(st["wind_std"]) + float(st["wind_mean"])
        pressure = float(out["pressure"].item()) * float(st["pressure_std"]) + float(st["pressure_mean"])
        result = {
            "status": "success",
            "source": "INSAT / processed real sample",
            "model": {
                "name": "INSAT Real-Data Multitask V2",
                "version": "V2",
                "checkpoint_epoch": insat_checkpoint.get("epoch"),
                "checkpoint": "insat_v2_best.pt",
            },
            "classification": {
                "predicted_category": category_names[category_index],
                "category_probabilities": {name: round(float(p), 4) for name, p in zip(category_names, probs)},
            },
            "intensity": {
                "wind_kt": round(float(wind), 2),
                "pressure_hpa": round(float(pressure), 2),
            },
            "presence": None,
            "note": (
                "V2 uses training-set image normalization and normalized multitask regression. "
                "The dataset is cyclone-positive, so the presence head is not used as a no-cyclone detector."
            ),
        }
        if "size" in out and "size_mean" in st:
            size = float(out["size"].item()) * float(st["size_std"]) + float(st["size_mean"])
            result["intensity"]["size_nmi"] = round(float(size), 2)
        return result

    # Legacy checkpoint fallback.
    tir_mean = float(insat_checkpoint["tir_mean"]); tir_std = float(insat_checkpoint["tir_std"])
    wv_mean = float(insat_checkpoint["wv_mean"]); wv_std = float(insat_checkpoint["wv_std"])
    wind_min = float(insat_checkpoint.get("wind_min", 20.0)); wind_max = float(insat_checkpoint.get("wind_max", 130.0))
    x = np.stack([(tir - tir_mean) / (tir_std + 1e-8), (wv - wv_mean) / (wv_std + 1e-8)], axis=0)
    x = torch.from_numpy(x).unsqueeze(0).float()
    with torch.no_grad():
        _, category_logits, intensity = insat_model(x)
    probs = torch.softmax(category_logits, dim=1)[0].cpu().numpy()
    category_index = int(np.argmax(probs))
    wind_norm = float(intensity.squeeze().item())
    wind = np.clip(wind_norm, 0.0, 1.0) * (wind_max - wind_min) + wind_min
    return {
        "status": "success", "source": "INSAT / processed real sample",
        "model": {"name": "INSAT Real-Data Classifier", "version": "legacy", "checkpoint": "classifier_insat.pt", "checkpoint_epoch": insat_checkpoint.get("epoch")},
        "classification": {"predicted_category": category_names[category_index], "category_probabilities": {name: round(float(p), 4) for name, p in zip(category_names, probs)}},
        "intensity": {"wind_kt": round(float(wind), 2)}, "presence": None,
        "note": "Legacy fallback checkpoint. V2 is preferred when insat_v2_best.pt exists.",
    }


def _image_to_128(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("L").resize((128, 128))
    return np.asarray(img, dtype=np.float32)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "models": {
            "synthetic": classifier is not None and predictor is not None,
            "tcir_temporal": tcir_temporal_model is not None,
            "tcir_intensity": tcir_intensity_model is not None,
            "insat": insat_model is not None,
            "track_gru_multi": track_multi_model is not None,
        },
        "errors": model_errors,
    }


@app.get("/data_sources")
def data_sources():
    tir_path, wv_path = _find_insat_sample()
    sources = {
        "TCIR": {
            "status": "READY" if TCIR_H5.exists() and TCIR_TEST.exists() else "MISSING",
            "dataset": str(TCIR_H5.name),
            "temporal_model": tcir_temporal_model is not None,
            "single_frame_model": tcir_intensity_model is not None,
            "manifest": TCIR_TEST.exists(),
        },
        "INSAT": {
            "status": "READY" if insat_model is not None else "MODEL MISSING",
            "classifier_model": insat_model is not None,
            "model_version": insat_checkpoint.get("model_version"),
            "real_sample": tir_path is not None and wv_path is not None,
            "sample": str(tir_path.parent.name) if tir_path else None,
        },
        "SYNTHETIC": {
            "status": "VALIDATION ONLY",
            "classifier_model": classifier is not None,
            "predictor_model": predictor is not None,
            "warning": "Synthetic data is not evidence of real-world cyclone performance.",
        },
        "LABELS": {
            "status": "READY" if (DATA_DIR / "labels").exists() else "MISSING",
            "directory": str(DATA_DIR / "labels"),
        },
    }
    return {"status": "success", "sources": sources}


@app.get("/metrics")
def metrics():
    out = {"status": "success", "overall": [], "intensity_bins": []}
    overall = EVAL_DIR / "overall_metrics.csv"
    bins = EVAL_DIR / "intensity_bin_metrics.csv"
    if overall.exists():
        out["overall"] = pd.read_csv(overall).replace({np.nan: None}).to_dict("records")
    if bins.exists():
        out["intensity_bins"] = pd.read_csv(bins).replace({np.nan: None}).to_dict("records")
    return out


@app.get("/evaluation_summary")
def evaluation_summary():
    """Expose frozen, locally generated evaluation artifacts without inventing metrics."""
    files = {
        "tcir": EVAL_DIR / "overall_metrics.csv",
        "insat": DATA_DIR / "processed" / "insat_v2_evaluation.json",
        "track": DATA_DIR / "processed" / "track_multi_evaluation.json",
        "insat_baseline": DATA_DIR / "processed" / "insat_baseline_evaluation.json",
        "insat_failures": DATA_DIR / "processed" / "insat_failure_analysis.json",
    }
    out = {"status": "success", "artifacts": {}}
    for name, path in files.items():
        if path.exists():
            try:
                if path.suffix.lower() == ".json":
                    out["artifacts"][name] = __import__("json").loads(path.read_text())
                else:
                    out["artifacts"][name] = pd.read_csv(path).replace({np.nan: None}).to_dict("records")
            except Exception as exc:
                out["artifacts"][name] = {"error": str(exc)}
        else:
            out["artifacts"][name] = {"status": "missing", "path": str(path)}
    return out


@app.get("/demo_sequence")
def demo_sequence():
    if classifier is None or predictor is None:
        return {"status": "error", "error": "Synthetic validation models are not loaded."}
    seq = make_sequence()
    trend_index = int(seq["trend_label"]) + 1
    return {
        "status": "success",
        "frames": seq["frames"].tolist(),
        "true_category": CATEGORIES[int(seq["category_label"])],
        "true_trend": TREND_LABELS[trend_index],
        "source": "synthetic_validation",
    }


@app.get("/tcir_demo")
def tcir_demo():
    if not TCIR_H5.exists() or not TCIR_TEST.exists():
        return {"status": "error", "error": "TCIR HDF5 or temporal test manifest not found."}
    df = pd.read_csv(TCIR_TEST)
    if df.empty:
        return {"status": "error", "error": "TCIR temporal test manifest is empty."}
    row = df.iloc[0]
    frame_cols = ["frame_0_h5_index", "frame_1_h5_index", "frame_2_h5_index", "frame_3_h5_index"]
    indices = [int(row[c]) for c in frame_cols]
    with h5py.File(TCIR_H5, "r") as h5:
        frames = [h5["Images"][i].astype(np.float32).tolist() for i in indices]
    wind = float(row["target_wind_kt"])
    return {
        "status": "success",
        "source": {
            "dataset": "TCIR",
            "file": TCIR_H5.name,
            "split": "test",
            "cyclone_id": str(row.get("cyclone_id", "TCIR TEST SAMPLE")),
            "target_timestamp": str(row.get("target_timestamp", "")),
        },
        "cyclone_id": str(row.get("cyclone_id", "TCIR TEST SAMPLE")),
        "target_timestamp": str(row.get("target_timestamp", "")),
        "temporal_context": {
            "frames": 4,
            "interval_hours": 3,
            "history_hours": 9,
            "chronological_order": "oldest_to_newest",
            "h5_indices": indices,
        },
        "ground_truth": {
            "wind_kt": float(row["target_wind_kt"]),
            "pressure_hpa": float(row["target_pressure_hpa"]),
            "size_nmi": float(row["target_size_nmi"]),
            "category": imd_category(wind),
        },
        "frames": frames,
    }



def _find_col(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    for col in df.columns:
        lc = str(col).strip().lower()
        if any(c in lc for c in candidates):
            return col
    return None


def _extract_track_from_manifest(df, cyclone_id):
    id_col = _find_col(df, ["cyclone_id", "storm_id", "storm", "name"])
    lat_col = _find_col(df, ["lat", "latitude", "target_lat", "target_latitude"])
    lon_col = _find_col(df, ["lon", "longitude", "target_lon", "target_longitude"])
    time_col = _find_col(df, ["target_timestamp", "timestamp", "datetime", "date_time", "time"])

    if lat_col is None or lon_col is None:
        return [], {"available": False, "reason": "The temporal manifest does not contain latitude/longitude columns."}

    work = df.copy()
    if id_col is not None and cyclone_id:
        mask = work[id_col].astype(str).str.strip().str.upper() == str(cyclone_id).strip().upper()
        if mask.any():
            work = work.loc[mask].copy()

    if time_col is not None:
        parsed = pd.to_datetime(work[time_col], errors="coerce")
        work = work.assign(_parsed_time=parsed).sort_values("_parsed_time", na_position="last")

    points = []
    for _, r in work.iterrows():
        try:
            lat = float(r[lat_col]); lon = float(r[lon_col])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            points.append({
                "timestamp": str(r[time_col]) if time_col is not None else "",
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "wind_kt": round(float(r["target_wind_kt"]), 2) if "target_wind_kt" in r and pd.notna(r["target_wind_kt"]) else None
            })
        except Exception:
            continue

    unique, seen = [], set()
    for p in points:
        key = (p["timestamp"], p["lat"], p["lon"])
        if key not in seen:
            unique.append(p); seen.add(key)
    return unique, {"available": bool(unique), "lat_column": str(lat_col), "lon_column": str(lon_col), "time_column": str(time_col) if time_col else None}


def _compass_direction(deg):
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[int((deg + 22.5) // 45) % 8]


def _project_motion(points, hours=(3, 6, 9)):
    if len(points) < 2:
        return [], {"available": False, "reason": "At least two observed positions are required for motion estimation."}
    a, b = points[-2], points[-1]
    try:
        ta = pd.to_datetime(a["timestamp"]); tb = pd.to_datetime(b["timestamp"])
        dt_h = max((tb - ta).total_seconds() / 3600.0, 1e-6)
    except Exception:
        dt_h = 3.0
    dlat = b["lat"] - a["lat"]; dlon = b["lon"] - a["lon"]
    lat_rad = np.deg2rad(b["lat"])
    east_nm = dlon * np.cos(lat_rad) * 60.0
    north_nm = dlat * 60.0
    speed_kt = float(np.hypot(east_nm, north_nm) / dt_h)
    bearing = (np.degrees(np.arctan2(east_nm, north_nm)) + 360.0) % 360.0
    projections = [
        {"hours_ahead": h, "lat": round(float(b["lat"] + dlat / dt_h * h), 5),
         "lon": round(float(b["lon"] + dlon / dt_h * h), 5)}
        for h in hours
    ]
    return projections, {
        "available": True, "speed_kt": round(speed_kt, 2),
        "bearing_deg": round(float(bearing), 1),
        "direction": _compass_direction(bearing),
        "based_on_hours": round(dt_h, 2)
    }


def _prepare_track_features(points, n=5):
    """Build the exact 5-feature history used by TrackGRUMulti.

    Feature order: latitude, longitude, wind_kt, pressure_hpa, size_nmi.
    The trained checkpoint supplies the training-set normalization statistics.
    """
    if len(points) < n:
        return None, {"available": False, "reason": f"Need at least {n} chronological observations for the GRU track model."}

    recent = points[-n:]
    required = ("wind_kt", "pressure_hpa", "size_nmi")
    missing = [
        name for name in required
        if any(p.get(name) is None for p in recent)
    ]
    if missing:
        return None, {"available": False, "reason": f"Track model features missing: {', '.join(missing)}."}

    x = np.asarray([
        [p["lat"], p["lon"], p["wind_kt"], p["pressure_hpa"], p["size_nmi"]]
        for p in recent
    ], dtype=np.float32)
    return x, {"available": True, "observations_used": n, "feature_order": ["lat", "lon", "wind_kt", "pressure_hpa", "size_nmi"]}


def _predict_track_gru(points):
    if track_multi_model is None:
        return [], {"available": False, "reason": model_errors.get("track_multi", "Track GRU checkpoint is not loaded.")}

    x, feature_info = _prepare_track_features(points, n=5)
    if x is None:
        return [], feature_info

    mean = np.asarray(track_multi_checkpoint.get("mean"), dtype=np.float32)
    std = np.asarray(track_multi_checkpoint.get("std"), dtype=np.float32)
    if mean.shape != (5,) or std.shape != (5,):
        return [], {"available": False, "reason": "Track GRU checkpoint has invalid normalization statistics."}

    xn = (x - mean) / np.maximum(std, 1e-6)
    tensor = torch.from_numpy(xn.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        delta = track_multi_model(tensor).squeeze(0).cpu().numpy()

    current = points[-1]
    horizons = track_multi_checkpoint.get("horizons", [3, 6, 9])
    projections = []
    for h, d in zip(horizons, delta):
        projections.append({
            "hours_ahead": int(h),
            "lat": round(float(current["lat"] + d[0]), 5),
            "lon": round(float(current["lon"] + d[1]), 5),
            "delta_lat_deg": round(float(d[0]), 5),
            "delta_lon_deg": round(float(d[1]), 5),
        })
    return projections, {"available": True, **feature_info, "model": "TrackGRUMulti"}


def _load_besttrack_points(storm: str | None = None):
    """Load the same IMD best-track source family used to train TrackGRUMulti."""
    files = sorted((DATA_DIR / "labels").glob("*_besttrack.csv"))
    if not files:
        return [], {"available": False, "reason": "No *_besttrack.csv files found in backend/data/labels."}, None

    requested = (storm or "AMPHAN").strip().upper()
    chosen = None
    for f in files:
        if f.stem.replace("_besttrack", "").upper() == requested:
            chosen = f
            break
    if chosen is None:
        # Accept a storm_id inside the CSV even if the filename differs.
        for f in files:
            try:
                d = pd.read_csv(f, nrows=3)
                col = _find_col(d, ["cyclone_id", "storm_id", "cyclone", "storm", "name"])
                if col is not None and requested in d[col].astype(str).str.upper().tolist():
                    chosen = f
                    break
            except Exception:
                continue
    if chosen is None:
        return [], {"available": False, "reason": f"Best-track file for storm '{requested}' was not found."}, requested

    try:
        df = pd.read_csv(chosen)
        lat_col = _find_col(df, ["latitude", "lat", "lat_deg", "center_lat", "storm_lat"])
        lon_col = _find_col(df, ["longitude", "lon", "long", "lon_deg", "center_lon", "storm_lon"])
        time_col = _find_col(df, ["timestamp", "datetime", "date", "time", "valid_time", "observation_time"])
        wind_col = _find_col(df, ["wind_kt", "wind", "max_wind", "maximum_wind", "vmax"])
        pressure_col = _find_col(df, ["pressure_hpa", "pressure", "mslp", "min_pressure"])
        size_col = _find_col(df, ["size_nmi", "size_nm", "size", "radius_nmi", "radius_nm"])
        if lat_col is None or lon_col is None or time_col is None:
            return [], {"available": False, "reason": f"Best-track file {chosen.name} lacks required latitude/longitude/timestamp columns."}, requested

        work = df.copy()
        work["_parsed_time"] = pd.to_datetime(work[time_col], errors="coerce", utc=True, format="mixed")
        work = work.dropna(subset=["_parsed_time"]).sort_values("_parsed_time")
        points = []
        for _, r in work.iterrows():
            try:
                lat, lon = float(r[lat_col]), float(r[lon_col])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue
                wind = float(r[wind_col]) if wind_col and pd.notna(r[wind_col]) else None
                pressure = float(r[pressure_col]) if pressure_col and pd.notna(r[pressure_col]) else None
                size = float(r[size_col]) if size_col and pd.notna(r[size_col]) else 0.0
                points.append({
                    "timestamp": str(r[time_col]),
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "wind_kt": round(wind, 2) if wind is not None else None,
                    "pressure_hpa": round(pressure, 2) if pressure is not None else None,
                    # Best-track training used 0 when no size column existed.
                    "size_nmi": round(size, 2) if size is not None else 0.0,
                })
            except Exception:
                continue

        unique, seen = [], set()
        for p in points:
            key = (p["timestamp"], p["lat"], p["lon"])
            if key not in seen:
                unique.append(p); seen.add(key)
        return unique, {
            "available": bool(unique),
            "source_file": str(chosen.relative_to(DATA_DIR)),
            "source": "IMD best-track CSV",
            "records": len(unique),
            "lat_column": str(lat_col), "lon_column": str(lon_col),
            "time_column": str(time_col),
        }, requested
    except Exception as exc:
        return [], {"available": False, "reason": str(exc)}, requested


@app.get("/track")
def track(storm: str = "AMPHAN"):
    points, availability, cyclone_id = _load_besttrack_points(storm)
    if not points:
        return {"status": "error", "error": availability.get("reason", "Best-track data unavailable.")}

    # The GRU was trained on 5 chronological observations with
    # [lat, lon, wind, pressure, size] features, using 0 for missing size.
    learned_projection, learned_info = _predict_track_gru(points)
    motion_projection, motion = _project_motion(points)

    return {
        "status": "success",
        "source": "IMD best-track history + learned multi-horizon Track GRU",
        "cyclone_id": cyclone_id,
        "current": points[-1],
        "history": points,
        "projection": learned_projection,
        "motion_baseline": motion_projection,
        "motion": motion,
        "track_model": learned_info,
        "availability": availability,
        "note": "Observed positions and meteorological features come from the IMD best-track CSV family used to construct the Track GRU training sequences. Future positions are produced by the learned GRU at 3h, 6h and 9h; constant-motion projection is retained separately as a baseline."
    }


@app.post("/predict_tcir_sequence")
def predict_tcir_sequence(payload: FramesIn):
    if tcir_temporal_model is None:
        return {"status": "error", "error": "TCIR temporal model is not loaded."}
    try:
        channels = tuple(tcir_temporal_checkpoint.get("channels", ("IR", "PMW")))
        tensor = _tcir_tensor(payload.frames, channels)
        with torch.no_grad():
            outputs = tcir_temporal_model(tensor)
        stats = tcir_temporal_checkpoint.get("target_stats", {})
        wind = max(0.0, _denorm(outputs["wind"].item(), stats, "wind"))
        pressure = max(0.0, _denorm(outputs["pressure"].item(), stats, "pressure"))
        size = max(0.0, _denorm(outputs["size"].item(), stats, "size"))
        return {
            "status": "success",
            "source": "TCIR",
            "model": {
                "name": "TCIR Temporal Intensity Model",
                "channels": list(channels),
                "sequence_length": int(tcir_temporal_checkpoint.get("sequence_length", 4)),
                "interval_hours": int(tcir_temporal_checkpoint.get("interval_hours", 3)),
                "history_hours": 9,
                "checkpoint_epoch": tcir_temporal_checkpoint.get("epoch"),
                "best_validation_loss": tcir_temporal_checkpoint.get("best_val_loss"),
            },
            "intensity": {
                "wind_kt": round(wind, 2),
                "pressure_hpa": round(pressure, 2),
                "size_nmi": round(size, 2),
            },
            "classification": {
                "predicted_category": imd_category(wind),
                "note": "Category is deterministically derived from predicted wind using the North Indian Ocean thresholds used by VayuNetra.",
            },
            "temporal_context": {
                "frames": 4,
                "interval_hours": 3,
                "history_hours": 9,
                "chronological_order": "oldest_to_newest",
            },
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.post("/predict_tcir_single")
def predict_tcir_single(payload: dict):
    if tcir_intensity_model is None:
        return {"status": "error", "error": "TCIR single-frame intensity model is not loaded."}
    try:
        channels = tuple(tcir_intensity_checkpoint.get("channels", ("IR", "PMW")))
        frame = payload["frame"]
        tensor = _tcir_single_tensor(frame, channels)
        with torch.no_grad():
            outputs = tcir_intensity_model(tensor)
        stats = tcir_intensity_checkpoint.get("target_stats", {})
        wind = max(0.0, _denorm(outputs["wind"].item(), stats, "wind"))
        pressure = max(0.0, _denorm(outputs["pressure"].item(), stats, "pressure"))
        size = max(0.0, _denorm(outputs["size"].item(), stats, "size"))
        return {
            "status": "success",
            "source": "TCIR",
            "model": {
                "name": "TCIR Single-Frame Intensity CNN",
                "channels": list(channels),
                "checkpoint_epoch": tcir_intensity_checkpoint.get("epoch"),
                "validation_loss": tcir_intensity_checkpoint.get("val_loss"),
            },
            "intensity": {
                "wind_kt": round(wind, 2),
                "pressure_hpa": round(pressure, 2),
                "size_nmi": round(size, 2),
            },
            "classification": {"predicted_category": imd_category(wind)},
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.get("/insat_demo")
def insat_demo():
    tir_path, wv_path = _find_insat_sample()
    if tir_path is None:
        return {"status": "error", "error": "No processed INSAT TIR1/WV sample found."}
    try:
        result = _insat_predict(np.load(tir_path), np.load(wv_path))
        result["sample"] = {
            "tir1": str(tir_path.relative_to(DATA_DIR)),
            "wv": str(wv_path.relative_to(DATA_DIR)),
        }
        return result
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.post("/predict_insat")
async def predict_insat(
    tir1: UploadFile = File(...),
    wv: UploadFile = File(...),
):
    """Accept raw .npy TIR1/WV arrays produced by the INSAT preprocessing pipeline."""
    try:
        tir = np.load(io.BytesIO(await tir1.read()))
        wv_arr = np.load(io.BytesIO(await wv.read()))
        result = _insat_predict(tir, wv_arr)
        result["source"] = "INSAT / uploaded numpy radiance"
        return result
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.post("/predict_image")
async def predict_image(images: list[UploadFile] = File(...)):
    """Legacy synthetic image route. Keep it clearly marked as validation-only."""
    if classifier is None or predictor is None:
        return {"status": "error", "error": "Synthetic models are not loaded."}
    if not images:
        return {"status": "error", "error": "No images supplied."}
    frames = []
    for upload in images:
        frames.append(_image_to_64(await upload.read()))
    frames = (frames * 4) if len(frames) == 1 else frames
    result = _run_synthetic(np.stack(frames))
    result["frames_received"] = len(images)
    result["source"] = "synthetic_validation"
    result["warning"] = "This route uses the synthetic pipeline and is not evidence of real-world cyclone performance."
    return result


def _image_to_64(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("L").resize((64, 64))
    return np.asarray(img, dtype=np.float32) / 255.0


def _run_synthetic(frames: np.ndarray):
    seq_tensor = torch.from_numpy(frames).unsqueeze(0).unsqueeze(2).float()
    last_frame = torch.from_numpy(frames[-1]).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        presence_logit, category_logits, intensity = classifier(last_frame)
        trend_logits, track_delta = predictor(seq_tensor)
    presence = float(torch.sigmoid(presence_logit).item())
    probs = torch.softmax(category_logits, dim=1)[0].tolist()
    category_idx = int(torch.argmax(category_logits, dim=1).item())
    trend_probs = torch.softmax(trend_logits, dim=1)[0].tolist()
    trend_idx = int(torch.argmax(trend_logits, dim=1).item())
    return {
        "status": "success",
        "identification": {"cyclone_present_probability": round(presence, 4)},
        "classification": {
            "predicted_category": CATEGORIES[category_idx],
            "category_probabilities": {
                CATEGORIES[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
        },
        "intensity": {
            "normalized_intensity": round(float(intensity.item()), 4),
            "note": "Synthetic validation output; do not interpret as wind speed.",
        },
        "prediction": {
            "trend": TREND_LABELS.get(trend_idx, "Unknown"),
            "trend_probabilities": {
                TREND_LABELS[i]: round(float(p), 4) for i, p in enumerate(trend_probs)
            },
            "predicted_next_step_track_delta": [round(float(x), 3) for x in track_delta[0].tolist()],
            "scope_note": "Short-horizon synthetic nowcasting only.",
        },
    }
