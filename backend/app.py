"""
app.py — FastAPI backend serving the cyclone AI/ML pipeline.

Endpoints:
    GET  /health
    GET  /demo_sequence
    POST /predict
    POST /predict_image

Run:
    uvicorn backend.app:app --reload --port 8000
"""

import os
import sys
import io

import numpy as np
import torch

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from pathlib import Path
import pandas as pd


# ============================================================
# PATH CONFIG
# ============================================================

# app.py is inside:
# cyclone-sih/backend/app.py
#
# Therefore .parent = cyclone-sih/backend

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "data" / "labels" / "cyclone_labels.csv"
OUTPUT = ROOT / "data" / "labels" / "cyclone_labels_v2.csv"



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


if __name__ == "__main__":
    main()

# Allow imports from backend/
sys.path.append(os.path.dirname(__file__))

from models.classifier import CycloneCNN
from models.predictor import CycloneTrendLSTM
from data.generate_synthetic import make_sequence, CATEGORIES


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(__file__)
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

FRAME_SIZE = 64

TREND_LABELS = {
    0: "Weakening",
    1: "Steady",
    2: "Intensifying",
}


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
# MODELS
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
               "epoch": ...,
               ...
           }

        2. Old raw state_dict format.
    """

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    return checkpoint


# ============================================================
# LOAD MODELS
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
        print(f"[WARNING] Classifier checkpoint not found:")
        print(f"         {c_path}")
        _models_loaded = False

    elif not os.path.exists(p_path):
        print(f"[WARNING] Predictor checkpoint not found:")
        print(f"         {p_path}")
        _models_loaded = False

    else:
        try:
            # Load classifier
            classifier.load_state_dict(
                load_checkpoint(c_path)
            )

            # Load predictor
            predictor.load_state_dict(
                load_checkpoint(p_path)
            )

            _models_loaded = True

            print("=" * 70)
            print("MODELS LOADED SUCCESSFULLY")
            print("=" * 70)
            print(f"Classifier : {c_path}")
            print(f"Predictor  : {p_path}")
            print("=" * 70)

        except Exception as e:
            _models_loaded = False

            print("=" * 70)
            print("MODEL LOADING FAILED")
            print("=" * 70)
            print(str(e))
            print("=" * 70)

    classifier.eval()
    predictor.eval()


load_models()


# ============================================================
# INPUT SCHEMA
# ============================================================

class SequenceIn(BaseModel):
    """
    Expected shape:

        frames = [
            [[...], [...], ...],   # frame 1
            [[...], [...], ...],   # frame 2
            ...
        ]

    Shape:
        (T, H, W)
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
    }


# ============================================================
# DEMO SEQUENCE
# ============================================================

@app.get("/demo_sequence")
def demo_sequence():
    """
    Generates a fresh synthetic sequence for the live demo.
    """

    seq = make_sequence()

    trend_index = seq["trend_label"] + 1

    return {
        "frames": seq["frames"].tolist(),

        "true_category": CATEGORIES[
            seq["category_label"]
        ],

        "true_trend": TREND_LABELS[
            trend_index
        ],
    }


# ============================================================
# CORE PREDICTION FUNCTION
# ============================================================

def run_pipeline(frames: np.ndarray):
    """
    Run:

        Identification
        Classification
        Intensity estimation
        Trend prediction
        Track delta prediction
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if frames.ndim != 3:
        raise ValueError(
            f"Expected frames with shape (T,H,W), got {frames.shape}"
        )

    if frames.shape[1] != FRAME_SIZE or frames.shape[2] != FRAME_SIZE:
        raise ValueError(
            f"Expected frames of size "
            f"{FRAME_SIZE}x{FRAME_SIZE}, got "
            f"{frames.shape[1]}x{frames.shape[2]}"
        )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    # (T,H,W)
    #
    # -> (1,T,1,H,W)
    seq_tensor = (
        torch.from_numpy(frames)
        .unsqueeze(0)
        .unsqueeze(2)
    )

    # Last frame:
    #
    # (H,W)
    #
    # -> (1,1,H,W)

    last_frame = (
        torch.from_numpy(frames[-1])
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        # NEW CLASSIFIER:
        #
        # presence
        # category
        # intensity

        (
            presence_logit,
            category_logits,
            intensity,
        ) = classifier(last_frame)

        # ----------------------------------------------------
        # Identification
        # ----------------------------------------------------

        presence_prob = (
            torch.sigmoid(
                presence_logit
            ).item()
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        category_pred = (
            category_logits
            .argmax(dim=1)
            .item()
        )

        category_probs = (
            torch.softmax(
                category_logits,
                dim=1
            )[0]
            .tolist()
        )

        # ----------------------------------------------------
        # Intensity
        # ----------------------------------------------------

        intensity_value = intensity.item()

        # ----------------------------------------------------
        # Trend + Track
        # ----------------------------------------------------

        trend_logits, track_delta = predictor(
            seq_tensor
        )

        trend_pred = (
            trend_logits
            .argmax(dim=1)
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
                for i, p in enumerate(category_probs)
            },
        },

        "intensity": {
            "normalized_intensity": round(
                intensity_value,
                4
            ),

            "note": (
                "Normalized synthetic intensity value. "
                "Do NOT interpret this as wind speed until "
                "the model is trained on real meteorological labels."
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
                for i, p in enumerate(trend_probs)
            },

            "predicted_next_step_track_delta": [
                round(x, 3)
                for x in track_delta[0].tolist()
            ],

            "scope_note": (
                "Short-horizon trend nowcasting only — "
                "not a full NWP-style forecast."
            ),
        },
    }


# ============================================================
# JSON SEQUENCE PREDICTION
# ============================================================

@app.post("/predict")
def predict(payload: SequenceIn):

    frames = np.array(
        payload.frames,
        dtype=np.float32
    )

    return run_pipeline(frames)


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

    Current synthetic model expects grayscale imagery.
    """

    img = Image.open(
        io.BytesIO(raw_bytes)
    ).convert("L")

    img = img.resize(
        (FRAME_SIZE, FRAME_SIZE)
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
    Accepts one or more uploaded images.

    Chronological order:
        oldest -> newest

    One image:
        identification + classification + intensity
        are meaningful.

        Trend/track prediction is NOT meaningful because
        there is no temporal sequence.

    Multiple images:
        identification + classification + intensity
        + trend + track prediction.
    """

    if len(images) == 0:
        return {
            "error": "No images supplied."
        }

    # --------------------------------------------------------
    # Read images
    # --------------------------------------------------------

    frames = []

    for upload in images:

        raw = await upload.read()

        frame = preprocess_image(
            raw
        )

        frames.append(frame)

    single_frame_only = (
        len(frames) == 1
    )

    # --------------------------------------------------------
    # Single image
    # --------------------------------------------------------

    if single_frame_only:

        # LSTM needs a sequence.
        #
        # Repeat frame only so tensor shape is valid.
        #
        # We explicitly mark prediction as unreliable later.

        frames_for_model = frames * 4

    else:

        frames_for_model = frames

    frames_arr = np.stack(
        frames_for_model
    )

    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    result = run_pipeline(
        frames_arr
    )

    # --------------------------------------------------------
    # Add image-specific metadata
    # --------------------------------------------------------

    result["frames_received"] = len(
        images
    )

    result["prediction"]["reliable"] = (
        not single_frame_only
    )

    if single_frame_only:

        result["prediction"]["scope_note"] = (
            "Only one image was submitted. "
            "Identification, classification and normalized "
            "intensity are available, but trend/track prediction "
            "is NOT reliable because no real temporal sequence "
            "was provided. Upload chronological images for "
            "temporal prediction."
        )

    return result