import os
import sys
import numpy as np
import torch

# ============================================================
# IMPORT MODEL
# ============================================================

# Add backend directory to Python path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from models.classifier import CycloneCNN


# ============================================================
# PATHS
# ============================================================

CHECKPOINT = "backend/checkpoints/classifier_insat.pt"

STATS = "backend/data/processed/insat_training_stats.npz"

SAMPLE_DIR = (
    "backend/data/processed/"
    "insat_amphan_calibrated/"
    "3DIMG_17MAY2020_1500_L1C_ASIA_MER_V01R00"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

categories = checkpoint["categories"]
in_channels = checkpoint["in_channels"]

model = CycloneCNN(
    num_categories=len(categories),
    in_channels=in_channels
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded.")
print("Categories:", categories)
print("Input channels:", in_channels)


# ============================================================
# LOAD NORMALIZATION STATS
# ============================================================

print("\n" + "=" * 60)
print("LOADING NORMALIZATION")
print("=" * 60)

stats = np.load(STATS)

# Actual keys in your file:
# mean
# std
# wind_min
# wind_max

mean = stats["mean"]
std = stats["std"]

tir_mean = float(mean[0])
wv_mean = float(mean[1])

tir_std = float(std[0])
wv_std = float(std[1])

wind_min = float(stats["wind_min"])
wind_max = float(stats["wind_max"])

print("TIR1 mean/std:", tir_mean, tir_std)
print("WV   mean/std:", wv_mean, wv_std)
print("Wind range:", wind_min, "-", wind_max, "kt")


# ============================================================
# LOAD REAL INSAT SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("LOADING INSAT SAMPLE")
print("=" * 60)

tir_path = os.path.join(
    SAMPLE_DIR,
    "tir1_radiance.npy"
)

wv_path = os.path.join(
    SAMPLE_DIR,
    "wv_radiance.npy"
)

if not os.path.exists(tir_path):
    raise FileNotFoundError(
        f"TIR1 file not found:\n{tir_path}"
    )

if not os.path.exists(wv_path):
    raise FileNotFoundError(
        f"WV file not found:\n{wv_path}"
    )

tir = np.load(tir_path).astype(np.float32)
wv = np.load(wv_path).astype(np.float32)

print("TIR1 file:", tir_path)
print("WV file  :", wv_path)

print("\nRaw shapes:")
print("TIR1:", tir.shape)
print("WV  :", wv.shape)


# ============================================================
# NORMALIZATION
# ============================================================

print("\n" + "=" * 60)
print("PREPARING MODEL INPUT")
print("=" * 60)

tir = (
    tir - tir_mean
) / (
    tir_std + 1e-8
)

wv = (
    wv - wv_mean
) / (
    wv_std + 1e-8
)

# Channel order:
# C0 = TIR1
# C1 = WV

x = np.stack(
    [tir, wv],
    axis=0
)

# Add batch dimension
# (2,128,128)
#       ↓
# (1,2,128,128)

x = torch.from_numpy(
    x
).unsqueeze(0).float()

print("Model input:", tuple(x.shape))


# ============================================================
# MODEL INFERENCE
# ============================================================

print("\n" + "=" * 60)
print("RUNNING INFERENCE")
print("=" * 60)

with torch.no_grad():

    # Your actual CycloneCNN.forward() returns:
    #
    # presence_logit
    # category_logits
    # intensity
    #
    presence_logit, category_logits, intensity = model(x)

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    category_prob = torch.softmax(
        category_logits,
        dim=1
    )

    predicted_class = int(
        torch.argmax(
            category_prob,
            dim=1
        ).item()
    )

    predicted_category = categories[
        predicted_class
    ]

    confidence = float(
        category_prob[
            0,
            predicted_class
        ].item()
    )

    # --------------------------------------------------------
    # WIND
    # --------------------------------------------------------

    wind_normalized = float(
        intensity.item()
    )

    wind_kt = (
        wind_min
        + wind_normalized
        * (wind_max - wind_min)
    )

    # --------------------------------------------------------
    # PRESENCE
    # --------------------------------------------------------

    presence_probability = float(
        torch.sigmoid(
            presence_logit
        ).item()
    )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("INSAT REAL-IMAGE INFERENCE RESULT")
print("=" * 60)

print("\nSAMPLE")
print("Storm           : AMPHAN")
print("Timestamp       : 2020-05-17 15:00 UTC")

print("\nACTUAL GROUND TRUTH")
print("Category        : Very Severe Cyclonic Storm")
print("Wind            : 75 kt")

print("\nMODEL PREDICTION")
print("Category        :", predicted_category)
print(
    "Confidence      :",
    f"{confidence * 100:.2f}%"
)
print(
    "Wind            :",
    f"{wind_kt:.2f} kt"
)

print(
    "Presence score  :",
    f"{presence_probability * 100:.2f}%"
)

print("\nCATEGORY PROBABILITIES")

for i, category in enumerate(categories):

    probability = (
        category_prob[0, i].item()
        * 100
    )

    print(
        f"{category:35s} "
        f"{probability:6.2f}%"
    )


# ============================================================
# SIMPLE COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

actual_category = "Very Severe Cyclonic Storm"
actual_wind = 75.0

category_correct = (
    predicted_category
    == actual_category
)

wind_error = abs(
    wind_kt - actual_wind
)

print(
    "Category correct :",
    "YES" if category_correct else "NO"
)

print(
    "Actual wind      :",
    f"{actual_wind:.2f} kt"
)

print(
    "Predicted wind   :",
    f"{wind_kt:.2f} kt"
)

print(
    "Wind error       :",
    f"{wind_error:.2f} kt"
)

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)