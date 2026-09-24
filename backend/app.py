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
import base64
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
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

def _search_roots():
    roots = [ROOT, PROJECT_ROOT, Path.cwd()]
    for base in list(roots):
        roots.extend(list(base.parents)[:3])
    out = []
    seen = set()
    for r in roots:
        r = r.resolve()
        if r not in seen and r.exists():
            seen.add(r); out.append(r)
    return out

SEARCH_ROOTS = _search_roots()

def _pick_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return candidate.resolve()
    # Last resort: search the actual project tree for the exact filename.
    target = candidates[0].name
    for root in SEARCH_ROOTS:
        try:
            hits = list(root.rglob(target))
            if hits:
                return hits[0].resolve()
        except Exception:
            pass
    return candidates[0].resolve()

# Resolve from the file location, the project root, the current working directory,
# and finally by filename search. This makes the app survive launching uvicorn
# from either the repository root or the backend directory.
DATA_DIR = _pick_existing(
    ROOT / "data",
    PROJECT_ROOT / "backend" / "data",
    PROJECT_ROOT / "data",
)
CKPT_DIR = _pick_existing(
    ROOT / "checkpoints",
    PROJECT_ROOT / "backend" / "checkpoints",
    PROJECT_ROOT / "checkpoints",
)

TCIR_H5 = _pick_existing(
    DATA_DIR / "raw" / "tcir" / "Cyclone_Images.h5",
    PROJECT_ROOT / "backend" / "data" / "raw" / "tcir" / "Cyclone_Images.h5",
)
TCIR_TEST = _pick_existing(
    DATA_DIR / "labels" / "tcir_temporal_test.csv",
    PROJECT_ROOT / "backend" / "data" / "labels" / "tcir_temporal_test.csv",
)
TCIR_TEMPORAL_CKPT = CKPT_DIR / "tcir_temporal_best.pt"
TCIR_INTENSITY_CKPT = CKPT_DIR / "tcir_intensity_best.pt"

INSAT_CKPT = CKPT_DIR / "classifier_insat.pt"
INSAT_PROCESSED = DATA_DIR / "processed"

SYNTH_CLASSIFIER_CKPT = CKPT_DIR / "classifier.pt"
SYNTH_PREDICTOR_CKPT = CKPT_DIR / "predictor.pt"

EVAL_DIR = DATA_DIR / "evaluation"
TRACK_CATALOG = ROOT / "track_catalog.json"

sys.path.insert(0, str(ROOT))

from models.classifier import CycloneCNN
from models.predictor import CycloneTrendLSTM
from models.tcir_temporal import TCIRTemporalModel
from models.tcir_intensity import TCIRIntensityCNN
from data.generate_synthetic import make_sequence, CATEGORIES

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
model_errors: dict[str, str] = {}


def _load_state(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj if isinstance(obj, dict) else {"model_state_dict": obj}


def _safe_load_models():
    global classifier, predictor
    global tcir_temporal_model, tcir_temporal_checkpoint
    global tcir_intensity_model, tcir_intensity_checkpoint
    global insat_model, insat_checkpoint

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

    # Real INSAT classifier.
    try:
        insat_checkpoint = _load_state(INSAT_CKPT)
        categories = insat_checkpoint["category_names"]
        insat_model = CycloneCNN(
            num_categories=len(categories),
            in_channels=int(insat_checkpoint.get("in_channels", 2)),
        )
        insat_model.load_state_dict(insat_checkpoint["model_state_dict"], strict=True)
        insat_model.eval()
    except Exception as exc:
        model_errors["insat"] = str(exc)
        insat_model = None
        insat_checkpoint = {}


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
    # Primary layout produced by train_insat_real.py.
    names = ["tir1_radiance.npy", "wv_radiance.npy"]
    for root in SEARCH_ROOTS:
        try:
            for tir_path in root.rglob(names[0]):
                wv_path = tir_path.parent / names[1]
                if wv_path.is_file():
                    return tir_path.resolve(), wv_path.resolve()
        except Exception:
            pass

    # Fallback: inspect INSAT manifests for explicit processed paths.
    for root in SEARCH_ROOTS:
        try:
            manifests = list(root.rglob("*_insat_manifest.csv")) + list(root.rglob("insat_combined_manifest.csv"))
            for manifest in manifests:
                df = pd.read_csv(manifest)
                cols = {str(c).lower(): c for c in df.columns}
                tir_col = next((c for k,c in cols.items() if "tir1" in k and ("path" in k or "file" in k)), None)
                wv_col = next((c for k,c in cols.items() if ("water" in k or k.startswith("wv")) and ("path" in k or "file" in k)), None)
                if tir_col and wv_col:
                    for _, row in df.iterrows():
                        tp = Path(str(row[tir_col])); wp = Path(str(row[wv_col]))
                        if not tp.is_absolute(): tp = (manifest.parent / tp).resolve()
                        if not wp.is_absolute(): wp = (manifest.parent / wp).resolve()
                        if tp.is_file() and wp.is_file(): return tp, wp
        except Exception:
            pass
    return None, None


def _insat_predict(tir: np.ndarray, wv: np.ndarray):
    if insat_model is None:
        raise RuntimeError("INSAT real-data model is not loaded.")
    tir = np.asarray(tir, dtype=np.float32)
    wv = np.asarray(wv, dtype=np.float32)
    if tir.shape != (128, 128) or wv.shape != (128, 128):
        raise ValueError(f"INSAT expects two 128x128 arrays; got {tir.shape} and {wv.shape}")

    tir_mean = float(insat_checkpoint["tir_mean"])
    tir_std = float(insat_checkpoint["tir_std"])
    wv_mean = float(insat_checkpoint["wv_mean"])
    wv_std = float(insat_checkpoint["wv_std"])
    wind_min = float(insat_checkpoint.get("wind_min", 20.0))
    wind_max = float(insat_checkpoint.get("wind_max", 130.0))

    x = np.stack(
        [(tir - tir_mean) / (tir_std + 1e-8),
         (wv - wv_mean) / (wv_std + 1e-8)],
        axis=0,
    )
    x = torch.from_numpy(x).unsqueeze(0).float()

    with torch.no_grad():
        presence_logit, category_logits, intensity = insat_model(x)

    probs = torch.softmax(category_logits, dim=1)[0].cpu().numpy()
    category_names = insat_checkpoint["category_names"]
    category_index = int(np.argmax(probs))
    wind_norm = float(intensity.squeeze().item())
    wind = np.clip(wind_norm, 0.0, 1.0) * (wind_max - wind_min) + wind_min

    return {
        "status": "success",
        "source": "INSAT / processed real sample",
        "model": {
            "name": "INSAT Real-Data Cyclone Classifier",
            "checkpoint_epoch": insat_checkpoint.get("epoch"),
        },
        "classification": {
            "predicted_category": category_names[category_index],
            "category_probabilities": {
                name: round(float(p), 4) for name, p in zip(category_names, probs)
            },
        },
        "intensity": {
            "wind_kt": round(float(wind), 2),
        },
        "presence": None,
        "note": (
            "The current INSAT training pipeline uses cyclone-positive samples; "
            "the presence head is therefore not used as a no-cyclone detector."
        ),
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
            "status": "LIVE SOURCE + MODEL" if insat_model is not None else "LIVE SOURCE",
            "live_source": True,
            "live_url": "https://www.mosdac.gov.in/scorpio/quad/",
            "platform": "INSAT-3DR/3DS",
            "classifier_model": insat_model is not None,
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


def _fallback_synthetic_sequence():
    """Deterministic local synthetic sequence used only when the generator fails.
    It keeps the demo endpoint alive; it is not model evidence or real data.
    """
    rng = np.random.default_rng(42)
    h = w = 128
    yy, xx = np.mgrid[0:h, 0:w]
    frames = []
    for t in range(4):
        cx = 42 + t * 7
        cy = 76 - t * 3
        r = np.sqrt((xx-cx)**2 + (yy-cy)**2)
        ring = np.exp(-((r-18.0)**2)/(2*7.0**2))
        core = np.exp(-((r)**2)/(2*9.0**2))
        swirl = 0.18*np.sin((xx+yy+t*10)/7.0)
        frame = (0.12 + 0.62*ring + 0.35*core + swirl + rng.normal(0,0.035,(h,w))).clip(0,1)
        frames.append(frame.astype(np.float32))
    return {"frames": np.stack(frames), "category_label": 1, "trend_label": 2}

@app.get("/demo_sequence")
def demo_sequence():
    try:
        seq = make_sequence()
        source = "synthetic_validation"
    except Exception as exc:
        seq = _fallback_synthetic_sequence()
        source = "synthetic_validation_fallback"
        model_errors["synthetic_generator"] = str(exc)
    try:
        trend_index = int(seq["trend_label"])
        category_index = int(seq["category_label"])
        frames = np.asarray(seq["frames"], dtype=np.float32)
        return {
            "status": "success",
            "frames": frames.tolist(),
            "true_category": CATEGORIES[category_index],
            "true_trend": TREND_LABELS.get(trend_index, "Intensifying"),
            "source": source,
            "note": "Synthetic validation only; not real-world cyclone evidence.",
        }
    except Exception as exc:
        return {"status":"error", "error":f"Synthetic sequence preparation failed: {exc}"}


@app.get("/debug_paths")
def debug_paths():
    tir_path, wv_path = _find_insat_sample()
    return {
        "project_root": str(PROJECT_ROOT),
        "backend_root": str(ROOT),
        "cwd": os.getcwd(),
        "tcir_h5": {"path": str(TCIR_H5), "exists": TCIR_H5.exists(), "size_gb": round(TCIR_H5.stat().st_size / (1024**3), 3) if TCIR_H5.is_file() else None},
        "tcir_manifest": {"path": str(TCIR_TEST), "exists": TCIR_TEST.exists(), "size_kb": round(TCIR_TEST.stat().st_size / 1024, 1) if TCIR_TEST.is_file() else None},
        "tcir_temporal_checkpoint": {"path": str(TCIR_TEMPORAL_CKPT), "exists": TCIR_TEMPORAL_CKPT.exists()},
        "insat_sample": {"tir1": str(tir_path) if tir_path else None, "wv": str(wv_path) if wv_path else None},
    }


@app.get("/tcir_demo")
def tcir_demo():
    if not TCIR_H5.is_file() or not TCIR_TEST.is_file():
        return {
            "status": "error",
            "error": "TCIR data path unresolved.",
            "paths": {
                "h5": str(TCIR_H5), "h5_exists": TCIR_H5.is_file(),
                "manifest": str(TCIR_TEST), "manifest_exists": TCIR_TEST.is_file(),
                "cwd": os.getcwd(), "backend_root": str(ROOT),
            },
        }
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


@app.get("/track")
def track():
    """Return a real best-track reference plus short-horizon motion projection.

    The TCIR temporal network predicts intensity/pressure/size, not lat/lon.
    For the demo scenario 200301L, the geographic track is sourced from a
    curated best-track catalog derived from NOAA/NHC HURDAT2 records.
    Future points are a transparent constant-velocity motion nowcast from
    the last two observed best-track points; they are not claimed as neural
    network track output.
    """
    catalog = {}
    if TRACK_CATALOG.exists():
        try:
            import json
            catalog = json.loads(TRACK_CATALOG.read_text(encoding="utf-8"))
        except Exception:
            catalog = {}

    cyclone_id = "200301L"
    if TCIR_TEST.exists():
        try:
            df = pd.read_csv(TCIR_TEST)
            if not df.empty:
                row = df.iloc[0]
                id_col = _find_col(df, ["cyclone_id", "storm_id", "storm", "name"])
                if id_col is not None and pd.notna(row[id_col]):
                    candidate = str(row[id_col]).strip()
                    if candidate in catalog:
                        cyclone_id = candidate
        except Exception:
            pass

    entry = catalog.get(cyclone_id)
    if not entry:
        return {
            "status": "error",
            "error": f"No verified geographic track is bundled for cyclone {cyclone_id}.",
            "demo_mode": False,
        }

    # Historical replay cutoff chosen to align the demo scenario with the
    # early TCIR 200301L sample shown on the dashboard (approximately 30 kt).
    # All points up to this timestamp are verified best-track observations.
    cutoff = pd.Timestamp("2003-04-18T18:00:00Z")
    points = [p for p in entry["points"] if pd.Timestamp(p["timestamp"]) <= cutoff]
    # Use the latest observation as the current best-track position.
    # The frontend can animate/replay this track without fabricating coordinates.
    projections, motion = _project_motion(points)
    current = points[-1]

    return {
        "status": "success",
        "source": entry["source"],
        "source_url": entry.get("source_url"),
        "cyclone_id": cyclone_id,
        "storm_id": entry.get("storm_id"),
        "storm_name": entry.get("name"),
        "basin": entry.get("basin"),
        "scenario": "HISTORICAL REPLAY / ANA 2003 / 01L",
        "demo_mode": False,
        "current": current,
        "history": points,
        "motion": motion,
        "projection": projections,
        "availability": {"available": True, "source": "verified_best_track"},
        "note": "Historical replay: observed coordinates are best-track observations through 2003-04-18 18:00 UTC. Future points are a constant-velocity geodesic motion baseline from the latest observed displacement, not neural-network latitude/longitude output.",
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


def _insat_png_data_url(path: Path) -> str:
    """Convert a processed INSAT .npy array into a browser-safe PNG data URL."""
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"INSAT array must be 2-D, got shape {arr.shape}")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("INSAT sample contains no finite values")
    lo, hi = np.percentile(finite, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if hi <= lo:
            hi = lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
    img = img.resize((768, 768), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/insat_preview/{channel}")
def insat_preview(channel: str):
    tir_path, wv_path = _find_insat_sample()
    path = tir_path if channel.lower() == "tir1" else wv_path if channel.lower() == "wv" else None
    if path is None:
        return {"status": "error", "error": "No local INSAT processed sample available."}
    arr = np.asarray(np.load(path), dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"status": "error", "error": "INSAT sample contains no finite values."}
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0, 1)
    img = Image.fromarray((norm * 255).astype(np.uint8), mode="L").resize((768, 768))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png", headers={"Cache-Control": "no-store"})


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
        result["preview"] = {
            "tir1": _insat_png_data_url(tir_path),
            "wv": _insat_png_data_url(wv_path),
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


@app.post("/predict")
def predict_synthetic(payload: dict[str, Any]):
    """Synthetic validation endpoint. Uses trained checkpoints when available;
    otherwise returns an explicitly labelled deterministic validation fallback.
    """
    frames = np.asarray(payload.get("frames", []), dtype=np.float32)
    if frames.ndim != 3 or frames.shape[0] < 1:
        return {"status": "error", "error": "Expected frames with shape [T,H,W]."}
    if classifier is not None and predictor is not None:
        try:
            out = _run_synthetic(frames)
            out["mode"] = "trained_model"
            return out
        except Exception as exc:
            model_errors["synthetic_inference"] = str(exc)
    # Explicit fallback for a reliable demo path; never presented as trained-model performance.
    mean_level = float(np.mean(frames[-1]))
    delta = float(np.mean(frames[-1]) - np.mean(frames[0])) if frames.shape[0] > 1 else 0.0
    trend = "Intensifying" if delta > 0.005 else ("Weakening" if delta < -0.005 else "Steady")
    return {
        "status": "success",
        "mode": "validation_fallback",
        "identification": {"cyclone_present_probability": round(float(np.clip(0.70 + mean_level*0.20, 0, 1)), 4)},
        "classification": {"predicted_category": "Deep Depression", "category_probabilities": {}},
        "intensity": {"normalized_intensity": round(mean_level, 4), "note": "Synthetic validation fallback; not a wind-speed estimate."},
        "prediction": {
            "trend": trend,
            "trend_probabilities": {},
            "predicted_next_step_track_delta": [round(delta, 3), round(-delta*0.6, 3)],
            "scope_note": "Synthetic validation only; fallback path used because trained synthetic inference was unavailable.",
        },
    }

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
