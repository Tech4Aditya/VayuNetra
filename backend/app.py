"""
app.py — FastAPI backend serving the cyclone AI/ML pipeline.

Endpoints:
    GET  /health
    GET  /demo_sequence
    POST /predict
    POST /predict_image
    POST /predict_tcir_sequence

Run:
    python -m uvicorn backend.app:app --reload --port 8000
"""

import os
import sys
import io

import numpy as np
import pandas as pd
import torch
import h5py

from pathlib import Path
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# PATH CONFIG
# ============================================================

# app.py is inside:
# cyclone-sih/backend/app.py

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "data" / "labels" / "cyclone_labels.csv"
OUTPUT = ROOT / "data" / "labels" / "cyclone_labels_v2.csv"


# ============================================================
# NORTH INDIAN OCEAN CATEGORY CLASSIFICATION
# ============================================================

def classify_wind(wind_kt):
    """
    North Indian Ocean intensity classification.

    Returns Unknown when the wind observation is unavailable.
    """

    if pd.isna(wind_kt):
        return "Unknown"

    wind = float(wind_kt)

    if wind < 17:
        return "Low Pressure Area"

    if wind <= 27:
        return "Depression"

    if wind <= 33:
        return "Deep Depression"

    if wind <= 47:
        return "Cyclonic Storm"

    if wind <= 63:
        return "Severe Cyclonic Storm"

    if wind <= 90:
        return "Very Severe Cyclonic Storm"

    if wind <= 119:
        return "Extremely Severe Cyclonic Storm"

    return "Super Cyclonic Storm"


# ============================================================
# LABEL PREPARATION
# ============================================================

def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT}"
        )

    df = pd.read_csv(INPUT)

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    df["category"] = df["wind_kt"].apply(
        classify_wind
    )

    # Every row belongs to a known cyclone track.
    df["presence"] = 1

    # Explicit masks for multi-task training.
    df["has_intensity"] = (
        df["wind_kt"].notna()
    ).astype(int)

    df["has_category"] = (
        df["category"] != "Unknown"
    ).astype(int)

    df.to_csv(
        OUTPUT,
        index=False
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print(f"Saved:\n{OUTPUT}")

    print("\nTarget availability:")

    print(
        f"Total observations : {len(df):,}"
    )

    print(
        f"Intensity available: "
        f"{df['has_intensity'].sum():,}"
    )

    print(
        f"Intensity missing  : "
        f"{(df['has_intensity'] == 0).sum():,}"
    )

    print(
        f"Category available : "
        f"{df['has_category'].sum():,}"
    )

    print(
        f"Category missing   : "
        f"{(df['has_category'] == 0).sum():,}"
    )

    print("\nCategory distribution:")

    print(
        df["category"]
        .value_counts()
        .to_string()
    )

    print("\nStorm/category distribution:")

    print(
        pd.crosstab(
            df["storm_name"],
            df["category"]
        ).to_string()
    )


# ============================================================
# ALLOW IMPORTS FROM BACKEND
# ============================================================

sys.path.append(
    os.path.dirname(__file__)
)


# ============================================================
# EXISTING SYNTHETIC MODELS
# ============================================================

from models.classifier import CycloneCNN
from models.predictor import CycloneTrendLSTM
from data.generate_synthetic import (
    make_sequence,
    CATEGORIES
)


# ============================================================
# REAL TCIR TEMPORAL MODEL
# ============================================================

from models.tcir_temporal import TCIRTemporalModel


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(__file__)

CKPT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)

FRAME_SIZE = 64


TREND_LABELS = {
    0: "Weakening",
    1: "Steady",
    2: "Intensifying",
}


# ------------------------------------------------------------
# TCIR paths
# ------------------------------------------------------------

TCIR_H5_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "tcir",
    "Cyclone_Images.h5"
)

TCIR_CHECKPOINT = os.path.join(
    CKPT_DIR,
    "tcir_temporal_best.pt"
)
TCIR_MANIFEST = os.path.join(
    BASE_DIR,
    "data",
    "labels",
    "tcir_temporal_test.csv"
)


TCIR_CHANNELS = [
    "IR",
    "PMW"
]


TCIR_CHANNEL_INDEX = {
    "IR": 0,
    "WV": 1,
    "VIS": 2,
    "PMW": 3,
}


tcir_temporal_model = None
tcir_temporal_checkpoint = None


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Cyclone AI/ML Pipeline — SIH26070"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# EXISTING MODELS
# ============================================================

classifier = CycloneCNN()

predictor = CycloneTrendLSTM()

_models_loaded = False


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(path):
    """
    Load either:

    1. New checkpoint format:
        {
            "model_state_dict": ...,
            "epoch": ...
        }

    2. Old raw state_dict format.
    """

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint["model_state_dict"]

    return checkpoint


# ============================================================
# LOAD EXISTING MODELS
# ============================================================

def load_models():

    global _models_loaded

    c_path = os.path.join(
        CKPT_DIR,
        "classifier.pt"
    )

    p_path = os.path.join(
        CKPT_DIR,
        "predictor.pt"
    )

    if not os.path.exists(c_path):

        print(
            "[WARNING] Classifier checkpoint not found:"
        )

        print(
            f"         {c_path}"
        )

        _models_loaded = False

    elif not os.path.exists(p_path):

        print(
            "[WARNING] Predictor checkpoint not found:"
        )

        print(
            f"         {p_path}"
        )

        _models_loaded = False

    else:

        try:

            # -------------------------------------------------
            # Load classifier
            # -------------------------------------------------

            classifier.load_state_dict(
                load_checkpoint(c_path)
            )

            # -------------------------------------------------
            # Load predictor
            # -------------------------------------------------

            predictor.load_state_dict(
                load_checkpoint(p_path)
            )

            _models_loaded = True

            print("=" * 70)
            print(
                "MODELS LOADED SUCCESSFULLY"
            )
            print("=" * 70)

            print(
                f"Classifier : {c_path}"
            )

            print(
                f"Predictor  : {p_path}"
            )

            print("=" * 70)

        except Exception as e:

            _models_loaded = False

            print("=" * 70)
            print(
                "MODEL LOADING FAILED"
            )
            print("=" * 70)

            print(str(e))

            print("=" * 70)

    classifier.eval()
    predictor.eval()


# ============================================================
# LOAD REAL TCIR TEMPORAL MODEL
# ============================================================

def load_tcir_temporal_model():

    global tcir_temporal_model
    global tcir_temporal_checkpoint

    if not os.path.exists(
        TCIR_CHECKPOINT
    ):

        print(
            "[WARNING] TCIR temporal checkpoint not found:"
        )

        print(
            f"         {TCIR_CHECKPOINT}"
        )

        return False

    try:

        # -----------------------------------------------------
        # Load checkpoint
        # -----------------------------------------------------

        checkpoint = torch.load(
            TCIR_CHECKPOINT,
            map_location="cpu"
        )

        # -----------------------------------------------------
        # Read model configuration
        # -----------------------------------------------------

        channels = tuple(
            checkpoint.get(
                "channels",
                ("IR", "PMW")
            )
        )

        sequence_length = checkpoint.get(
            "sequence_length",
            4
        )

        interval_hours = checkpoint.get(
            "interval_hours",
            3
        )

        # -----------------------------------------------------
        # Build model
        # -----------------------------------------------------

        tcir_temporal_model = TCIRTemporalModel(
            in_channels=len(channels)
        )

        # -----------------------------------------------------
        # Load weights
        # -----------------------------------------------------

        tcir_temporal_model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        tcir_temporal_model.eval()

        tcir_temporal_checkpoint = checkpoint

        # -----------------------------------------------------
        # Report
        # -----------------------------------------------------

        print("=" * 70)
        print(
            "TCIR TEMPORAL MODEL LOADED"
        )
        print("=" * 70)

        print(
            f"Checkpoint : {TCIR_CHECKPOINT}"
        )

        print(
            f"Channels   : {channels}"
        )

        print(
            f"Sequence   : "
            f"{sequence_length} frames"
        )

        print(
            f"Interval   : "
            f"{interval_hours} hours"
        )

        print(
            f"Epoch      : "
            f"{checkpoint.get('epoch', 'unknown')}"
        )

        print(
            f"Best loss  : "
            f"{checkpoint.get('best_val_loss', 'unknown')}"
        )

        print("=" * 70)

        return True

    except Exception as e:

        print("=" * 70)
        print(
            "TCIR TEMPORAL MODEL LOADING FAILED"
        )
        print("=" * 70)

        print(str(e))

        print("=" * 70)

        tcir_temporal_model = None
        tcir_temporal_checkpoint = None

        return False


# ============================================================
# LOAD ALL MODELS
# ============================================================

load_models()

load_tcir_temporal_model()


# ============================================================
# EXISTING INPUT SCHEMA
# ============================================================

class SequenceIn(BaseModel):
    """
    Existing synthetic pipeline.

    Expected shape:

        (T,H,W)

    where H=W=64.
    """

    frames: list


# ============================================================
# TCIR INPUT SCHEMA
# ============================================================

class TCIRSequenceIn(BaseModel):
    """
    Real TCIR temporal model input.

    Expected:

        4 chronological frames

    Each frame:

        128 x 128 x 4

    Channel order:

        IR
        WV
        VIS
        PMW

    The trained model currently uses:

        IR + PMW
    """

    frames: list


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "models_loaded": _models_loaded,

        "tcir_temporal_model_loaded": (
            tcir_temporal_model is not None
        ),

    }


# ============================================================
# DEMO SEQUENCE
# ============================================================

@app.get("/demo_sequence")
def demo_sequence():

    """
    Generates a fresh synthetic sequence
    for the existing live demo.
    """

    seq = make_sequence()

    trend_index = (
        seq["trend_label"] + 1
    )

    return {

        "frames": (
            seq["frames"].tolist()
        ),

        "true_category": CATEGORIES[
            seq["category_label"]
        ],

        "true_trend": TREND_LABELS[
            trend_index
        ],

    }


# ============================================================
# EXISTING SYNTHETIC CORE PIPELINE
# ============================================================

def run_pipeline(
    frames: np.ndarray
):

    """
    Existing synthetic pipeline.

    Run:

        Identification
        Classification
        Intensity estimation
        Trend prediction
        Track delta prediction
    """

    # ---------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------

    if frames.ndim != 3:

        raise ValueError(
            f"Expected frames with shape "
            f"(T,H,W), got {frames.shape}"
        )

    if (
        frames.shape[1] != FRAME_SIZE
        or frames.shape[2] != FRAME_SIZE
    ):

        raise ValueError(
            f"Expected frames of size "
            f"{FRAME_SIZE}x{FRAME_SIZE}, got "
            f"{frames.shape[1]}x"
            f"{frames.shape[2]}"
        )

    # ---------------------------------------------------------
    # Convert to tensors
    # ---------------------------------------------------------

    # (T,H,W)
    #
    # -> (1,T,1,H,W)

    seq_tensor = (
        torch.from_numpy(frames)
        .unsqueeze(0)
        .unsqueeze(2)
    )

    # ---------------------------------------------------------
    # Last frame
    # ---------------------------------------------------------

    # (H,W)
    #
    # -> (1,1,H,W)

    last_frame = (
        torch.from_numpy(
            frames[-1]
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # ---------------------------------------------------------
    # MODEL INFERENCE
    # ---------------------------------------------------------

    with torch.no_grad():

        # -----------------------------------------------------
        # Classifier
        # -----------------------------------------------------

        (
            presence_logit,
            category_logits,
            intensity,
        ) = classifier(
            last_frame
        )

        # -----------------------------------------------------
        # Identification
        # -----------------------------------------------------

        presence_prob = (
            torch.sigmoid(
                presence_logit
            ).item()
        )

        # -----------------------------------------------------
        # Classification
        # -----------------------------------------------------

        category_pred = (
            category_logits
            .argmax(
                dim=1
            )
            .item()
        )

        category_probs = (
            torch.softmax(
                category_logits,
                dim=1
            )[0]
            .tolist()
        )

        # -----------------------------------------------------
        # Intensity
        # -----------------------------------------------------

        intensity_value = (
            intensity.item()
        )

        # -----------------------------------------------------
        # Trend + Track
        # -----------------------------------------------------

        trend_logits, track_delta = (
            predictor(
                seq_tensor
            )
        )

        trend_pred = (
            trend_logits
            .argmax(
                dim=1
            )
            .item()
        )

        trend_probs = (
            torch.softmax(
                trend_logits,
                dim=1
            )[0]
            .tolist()
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "identification": {

            "cyclone_present_probability": round(
                presence_prob,
                4
            )

        },

        "classification": {

            "predicted_category": CATEGORIES[
                category_pred
            ],

            "category_probabilities": {

                CATEGORIES[i]: round(
                    p,
                    4
                )

                for i, p in enumerate(
                    category_probs
                )

            },

        },

        "intensity": {

            "normalized_intensity": round(
                intensity_value,
                4
            ),

            "note": (
                "Normalized synthetic intensity "
                "value. Do NOT interpret this as "
                "wind speed until the model is trained "
                "on real meteorological labels."
            ),

        },

        "prediction": {

            "trend": TREND_LABELS[
                trend_pred
            ],

            "trend_probabilities": {

                TREND_LABELS[i]: round(
                    p,
                    4
                )

                for i, p in enumerate(
                    trend_probs
                )

            },

            "predicted_next_step_track_delta": [

                round(
                    x,
                    3
                )

                for x in track_delta[
                    0
                ].tolist()

            ],

            "scope_note": (
                "Short-horizon trend nowcasting only — "
                "not a full NWP-style forecast."
            ),

        },

    }


# ============================================================
# EXISTING JSON SEQUENCE PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    payload: SequenceIn
):

    frames = np.array(
        payload.frames,
        dtype=np.float32
    )

    return run_pipeline(
        frames
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    raw_bytes: bytes
) -> np.ndarray:

    """
    Convert uploaded image to:

        (64,64)

    float32 array in [0,1].

    Existing synthetic model expects
    grayscale imagery.
    """

    img = Image.open(
        io.BytesIO(raw_bytes)
    ).convert("L")

    img = img.resize(
        (
            FRAME_SIZE,
            FRAME_SIZE
        )
    )

    arr = np.array(
        img,
        dtype=np.float32
    ) / 255.0

    return arr


# ============================================================
# IMAGE PREDICTION
# ============================================================

@app.post("/predict_image")
async def predict_image(
    images: list[UploadFile] = File(...)
):

    """
    Existing image endpoint.

    Accepts one or more uploaded images.

    Chronological order:

        oldest -> newest

    One image:

        identification + classification +
        intensity are meaningful.

    Multiple images:

        identification + classification +
        intensity + trend + track prediction.
    """

    if len(images) == 0:

        return {
            "error": "No images supplied."
        }

    # ---------------------------------------------------------
    # Read images
    # ---------------------------------------------------------

    frames = []

    for upload in images:

        raw = await upload.read()

        frame = preprocess_image(
            raw
        )

        frames.append(
            frame
        )

    single_frame_only = (
        len(frames) == 1
    )

    # ---------------------------------------------------------
    # Single image
    # ---------------------------------------------------------

    if single_frame_only:

        # LSTM needs a sequence.
        #
        # Repeat frame only so tensor shape
        # is valid.
        #
        # Prediction is explicitly marked
        # unreliable later.

        frames_for_model = (
            frames * 4
        )

    else:

        frames_for_model = frames

    frames_arr = np.stack(
        frames_for_model
    )

    # ---------------------------------------------------------
    # Run pipeline
    # ---------------------------------------------------------

    result = run_pipeline(
        frames_arr
    )

    # ---------------------------------------------------------
    # Add image-specific metadata
    # ---------------------------------------------------------

    result["frames_received"] = len(
        images
    )

    result["prediction"]["reliable"] = (
        not single_frame_only
    )

    if single_frame_only:

        result["prediction"]["scope_note"] = (
            "Only one image was submitted. "
            "Identification, classification and "
            "normalized intensity are available, "
            "but trend/track prediction is NOT "
            "reliable because no real temporal "
            "sequence was provided. Upload "
            "chronological images for temporal "
            "prediction."
        )

    return result


# ============================================================
# TCIR FRAME PREPROCESSING
# ============================================================

def preprocess_tcir_frames(
    frames: list,
    channels=("IR", "PMW")
):
    """
    Convert TCIR frames into:

        (1,T,C,128,128)

    Expected input:

        T x 128 x 128 x 4

    where channels are:

        IR
        WV
        VIS
        PMW

    Values may be:

        0-255

    or:

        0-1
    """

    arr = np.asarray(
        frames,
        dtype=np.float32
    )

    # ---------------------------------------------------------
    # Validate dimensions
    # ---------------------------------------------------------

    if arr.ndim != 4:

        raise ValueError(
            "Expected TCIR input with shape "
            "(T,128,128,4), got "
            f"{arr.shape}"
        )

    # ---------------------------------------------------------
    # Exactly 4 frames
    # ---------------------------------------------------------

    if arr.shape[0] != 4:

        raise ValueError(
            "TCIR temporal model requires "
            f"exactly 4 frames, got "
            f"{arr.shape[0]}"
        )

    # ---------------------------------------------------------
    # Spatial dimensions
    # ---------------------------------------------------------

    if (
        arr.shape[1] != 128
        or arr.shape[2] != 128
    ):

        raise ValueError(
            "Expected TCIR frames of size "
            f"128x128, got "
            f"{arr.shape[1]}x"
            f"{arr.shape[2]}"
        )

    # ---------------------------------------------------------
    # Channel dimensions
    # ---------------------------------------------------------

    if arr.shape[3] != 4:

        raise ValueError(
            "Expected 4 TCIR channels "
            "(IR,WV,VIS,PMW), got "
            f"{arr.shape[3]}"
        )

    # ---------------------------------------------------------
    # Select requested channels
    # ---------------------------------------------------------

    selected = []

    for channel in channels:

        if channel not in TCIR_CHANNEL_INDEX:

            raise ValueError(
                f"Unknown TCIR channel: "
                f"{channel}"
            )

        channel_index = (
            TCIR_CHANNEL_INDEX[
                channel
            ]
        )

        channel_data = (
            arr[
                :,
                :,
                :,
                channel_index
            ]
        )

        selected.append(
            channel_data
        )

    # ---------------------------------------------------------
    # T,H,W,C -> T,C,H,W
    # ---------------------------------------------------------

    arr = np.stack(
        selected,
        axis=1
    )

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------

    # TCIR dataset is [0,255].

    if arr.max() > 1.0:

        arr = arr / 255.0

    # ---------------------------------------------------------
    # Numerical safety
    # ---------------------------------------------------------

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # ---------------------------------------------------------
    # Add batch dimension
    # ---------------------------------------------------------

    tensor = torch.from_numpy(
        arr.copy()
    ).unsqueeze(0)

    return tensor


# ============================================================
# IMD CATEGORY FROM PREDICTED WIND
# ============================================================

def derive_imd_category(
    wind_kt
):
    """
    Derive a North Indian Ocean category
    from predicted wind.

    IMPORTANT:

    This is NOT a separately trained classifier.

    It is a deterministic category mapping
    applied to the predicted wind.
    """

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


# ============================================================
# REAL TCIR TEMPORAL PREDICTION
# ============================================================

@app.post(
    "/predict_tcir_sequence"
)
def predict_tcir_sequence(
    payload: TCIRSequenceIn
):
    """
    Real TCIR temporal intensity inference.

    Input:
        4 chronological satellite frames

    Each:
        128 x 128 x 4

    Channels:
        IR
        WV
        VIS
        PMW

    Model:
        IR + PMW

    Output:
        wind
        pressure
        size
        derived IMD category
    """

    if tcir_temporal_model is None:
        return {
            "status": "error",
            "error": "TCIR temporal model is not loaded."
        }

    try:

        # -----------------------------------------------------
        # Preprocess
        # -----------------------------------------------------

        tensor = preprocess_tcir_frames(
            payload.frames,
            channels=("IR", "PMW")
        )

        # -----------------------------------------------------
        # Model inference
        # -----------------------------------------------------

        with torch.no_grad():
            outputs = tcir_temporal_model(
                tensor
            )

        # -----------------------------------------------------
        # Target normalization statistics
        # -----------------------------------------------------

        checkpoint_stats = (
            tcir_temporal_checkpoint
            .get("target_stats", {})
        )

        wind_mean, wind_std = checkpoint_stats.get(
            "wind",
            (51.0, 27.0)
        )

        pressure_mean, pressure_std = checkpoint_stats.get(
            "pressure",
            (988.4, 20.0)
        )

        size_mean, size_std = checkpoint_stats.get(
            "size",
            (54.5, 55.0)
        )

        # -----------------------------------------------------
        # Denormalize predictions
        # -----------------------------------------------------

        wind = (
            outputs["wind"].item()
            * wind_std
            + wind_mean
        )

        pressure = (
            outputs["pressure"].item()
            * pressure_std
            + pressure_mean
        )

        size = (
            outputs["size"].item()
            * size_std
            + size_mean
        )

        # -----------------------------------------------------
        # Sanity clipping
        # -----------------------------------------------------

        wind = max(0.0, float(wind))
        pressure = max(0.0, float(pressure))
        size = max(0.0, float(size))

        # -----------------------------------------------------
        # Derived category
        # -----------------------------------------------------

        category = derive_imd_category(
            wind
        )

        # -----------------------------------------------------
        # Response
        # -----------------------------------------------------

        return {

            "status": "success",

            "model": {
                "name": "TCIR Temporal Intensity Model",
                "channels": ["IR", "PMW"],
                "sequence_length": (
                    tcir_temporal_checkpoint.get(
                        "sequence_length",
                        4
                    )
                ),
                "interval_hours": (
                    tcir_temporal_checkpoint.get(
                        "interval_hours",
                        3
                    )
                ),
                "history_hours": 9,
                "checkpoint_epoch": (
                    tcir_temporal_checkpoint.get(
                        "epoch"
                    )
                ),
                "best_validation_loss": (
                    tcir_temporal_checkpoint.get(
                        "best_val_loss"
                    )
                ),
            },

            "intensity": {
                "wind_kt": round(
                    wind,
                    2
                ),
                "pressure_hpa": round(
                    pressure,
                    2
                ),
                "size_nmi": round(
                    size,
                    2
                ),
            },

            "classification": {
                "predicted_category": category,
                "note": (
                    "Category is derived deterministically "
                    "from predicted wind. It is not a "
                    "separately trained classifier."
                ),
            },

            "temporal_context": {
                "frames": 4,
                "interval_hours": 3,
                "history_hours": 9,
                "chronological_order": (
                    "oldest_to_newest"
                ),
            },

        }

    except Exception as e:

        return {
            "status": "error",
            "error": str(e)
        }
# ============================================================
# REAL TCIR DEMO SEQUENCE
# ============================================================

@app.get("/tcir_demo")
def tcir_demo():

    """
    Return one real TCIR temporal sequence.

    The sequence is taken directly from the
    temporal test manifest and HDF5 dataset.

    Four chronological frames:
        t-9h
        t-6h
        t-3h
        t

    The response includes the ground-truth values
    for evaluation/demo transparency.
    """

    if not os.path.exists(TCIR_H5_PATH):

        return {
            "status": "error",
            "error": (
                "TCIR HDF5 dataset not found."
            ),
        }

    if not os.path.exists(TCIR_MANIFEST):

        return {
            "status": "error",
            "error": (
                "TCIR temporal test manifest "
                "not found."
            ),
        }

    try:

        # -----------------------------------------------------
        # Load temporal manifest
        # -----------------------------------------------------

        df = pd.read_csv(
            TCIR_MANIFEST
        )

        if len(df) == 0:

            return {
                "status": "error",
                "error": (
                    "TCIR temporal manifest "
                    "is empty."
                ),
            }

        # -----------------------------------------------------
        # Select first real test sequence
        # -----------------------------------------------------

        row = df.iloc[0]

        frame_columns = [
            "frame_0_h5_index",
            "frame_1_h5_index",
            "frame_2_h5_index",
            "frame_3_h5_index",
        ]

        indices = [
            int(row[column])
            for column in frame_columns
        ]

        # -----------------------------------------------------
        # Read HDF5
        # -----------------------------------------------------

        with h5py.File(
            TCIR_H5_PATH,
            "r"
        ) as h5:

            images = h5["Images"]

            raw_frames = [
                images[index]
                for index in indices
            ]

        # -----------------------------------------------------
        # Convert frames to Python lists
        # -----------------------------------------------------

        frames = [
            frame.astype(
                np.float32
            ).tolist()
            for frame in raw_frames
        ]

        # -----------------------------------------------------
        # Ground truth
        # -----------------------------------------------------

        true_wind = float(
            row["target_wind_kt"]
        )

        true_pressure = float(
            row["target_pressure_hpa"]
        )

        true_size = float(
            row["target_size_nmi"]
        )

        # -----------------------------------------------------
        # Return
        # -----------------------------------------------------

        return {

            "status": "success",

            "source": {
                "dataset": "TCIR",
                "file": os.path.basename(
                    TCIR_H5_PATH
                ),
                "split": "test",
                "cyclone_id": str(
                    row["cyclone_id"]
                ),
                "target_timestamp": str(
                    row["target_timestamp"]
                ),
            },

            "temporal_context": {

                "frames": 4,

                "interval_hours": 3,

                "history_hours": 9,

                "chronological_order":
                    "oldest_to_newest",

                "h5_indices": indices,

            },

            "ground_truth": {

                "wind_kt": true_wind,

                "pressure_hpa": true_pressure,

                "size_nmi": true_size,

            },

            "frames": frames,

        }

    except Exception as e:

        return {

            "status": "error",

            "error": str(e),

        }

    """
    Real TCIR temporal intensity inference.

    Input:

        4 chronological satellite frames

    Each:

        128 x 128 x 4

    Channels:

        IR
        WV
        VIS
        PMW

    Model:

        IR + PMW

    Output:

        wind
        pressure
        size
        derived IMD category
    """

    # ---------------------------------------------------------
    # Model availability
    # ---------------------------------------------------------

    if tcir_temporal_model is None:

        return {

            "status": "error",

            "error": (
                "TCIR temporal model "
                "is not loaded."
            ),

        }

    try:

        # -----------------------------------------------------
        # Preprocess
        # -----------------------------------------------------

        tensor = preprocess_tcir_frames(
            payload.frames,
            channels=(
                "IR",
                "PMW"
            )
        )

        # -----------------------------------------------------
        # Inference
        # -----------------------------------------------------

        with torch.no_grad():

            outputs = (
                tcir_temporal_model(
                    tensor
                )
            )

        # -----------------------------------------------------
        # Target normalization statistics
        # -----------------------------------------------------

        checkpoint_stats = (
            tcir_temporal_checkpoint
            .get(
                "target_stats",
                {}
            )
        )

        wind_mean, wind_std = (
            checkpoint_stats.get(
                "wind",
                (51.0, 27.0)
            )
        )

        pressure_mean, pressure_std = (
            checkpoint_stats.get(
                "pressure",
                (988.4, 20.0)
            )
        )

        size_mean, size_std = (
            checkpoint_stats.get(
                "size",
                (54.5, 55.0)
            )
        )

        # -----------------------------------------------------
        # DENORMALIZATION
        # -----------------------------------------------------

        wind = (
            outputs["wind"].item()
            * wind_std
            + wind_mean
        )

        pressure = (
            outputs["pressure"].item()
            * pressure_std
            + pressure_mean
        )

        size = (
            outputs["size"].item()
            * size_std
            + size_mean
        )

        # -----------------------------------------------------
        # Physical sanity clipping
        # -----------------------------------------------------

        wind = max(
            0.0,
            float(wind)
        )

        pressure = max(
            0.0,
            float(pressure)
        )

        size = max(
            0.0,
            float(size)
        )

        # -----------------------------------------------------
        # Derived category
        # -----------------------------------------------------

        category = derive_imd_category(
            wind
        )

        # -----------------------------------------------------
        # Response
        # -----------------------------------------------------

        return {

            "status": "success",

            "model": {

                "name": (
                    "TCIR Temporal "
                    "Intensity Model"
                ),

                "channels": [
                    "IR",
                    "PMW"
                ],

                "sequence_length": (
                    tcir_temporal_checkpoint
                    .get(
                        "sequence_length",
                        4
                    )
                ),

                "interval_hours": (
                    tcir_temporal_checkpoint
                    .get(
                        "interval_hours",
                        3
                    )
                ),

                "history_hours": 9,

                "checkpoint_epoch": (
                    tcir_temporal_checkpoint
                    .get(
                        "epoch"
                    )
                ),

                "best_validation_loss": (
                    tcir_temporal_checkpoint
                    .get(
                        "best_val_loss"
                    )
                ),

            },

            "intensity": {

                "wind_kt": round(
                    wind,
                    2
                ),

                "pressure_hpa": round(
                    pressure,
                    2
                ),

                "size_nmi": round(
                    size,
                    2
                ),

            },

            "classification": {

                "predicted_category": (
                    category
                ),

                "note": (
                    "Category is derived "
                    "deterministically from "
                    "predicted wind. It is "
                    "not a separately trained "
                    "classifier."
                ),

            },

            "temporal_context": {

                "frames": 4,

                "interval_hours": 3,

                "history_hours": 9,

                "chronological_order": (
                    "oldest_to_newest"
                ),

            },

        }

    except Exception as e:

        return {

            "status": "error",

            "error": str(e),

        }