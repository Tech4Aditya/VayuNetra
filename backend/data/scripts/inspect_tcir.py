import h5py
import numpy as np
from collections import Counter

IMAGE_PATH = "backend/data/raw/tcir/Cyclone_Images.h5"
LABEL_PATH = "backend/data/raw/tcir/Cyclone_Labels h5.npy"

print("=" * 70)
print("TCIR DATASET INSPECTION")
print("=" * 70)

# -----------------------------
# Labels
# -----------------------------

labels = np.load(LABEL_PATH, allow_pickle=True)

print("\nLABELS")
print("Shape:", labels.shape)
print("Dtype:", labels.dtype)

print("\nFirst 5 rows:")
for row in labels[:5]:
    print(row)

print("\nUnique basins:")
print(Counter(labels[:, 0]))

print("\nUnique cyclones:")
print("Count:", len(set(labels[:, 1])))

# -----------------------------
# Images
# -----------------------------

with h5py.File(IMAGE_PATH, "r") as f:

    images = f["Images"]

    print("\nIMAGES")
    print("Shape:", images.shape)
    print("Dtype:", images.dtype)

    # Inspect several frames instead of loading everything
    indices = [0, 1, 100, 1000, 10000, 20000]

    for idx in indices:

        if idx >= len(images):
            continue

        x = images[idx]

        print(f"\nFrame {idx}")

        for c, name in enumerate(["IR", "WV", "VIS", "PMW"]):

            channel = x[:, :, c]

            finite = channel[np.isfinite(channel)]

            print(
                f"  {name}: "
                f"min={finite.min():.4f}, "
                f"max={finite.max():.4f}, "
                f"mean={finite.mean():.4f}, "
                f"std={finite.std():.4f}, "
                f"NaN={np.isnan(channel).sum()}"
            )

# -----------------------------
# Label statistics
# -----------------------------

winds = np.array(
    [float(x) for x in labels[:, 5] if str(x) != "nan"],
    dtype=np.float32
)

pressures = np.array(
    [float(x) for x in labels[:, 7] if str(x) != "nan"],
    dtype=np.float32
)

sizes = np.array(
    [float(x) for x in labels[:, 6] if str(x) != "nan"],
    dtype=np.float32
)

print("\nLABEL STATISTICS")

print(
    f"Wind: "
    f"min={winds.min():.2f}, "
    f"max={winds.max():.2f}, "
    f"mean={winds.mean():.2f}"
)

print(
    f"Pressure: "
    f"min={pressures.min():.2f}, "
    f"max={pressures.max():.2f}, "
    f"mean={pressures.mean():.2f}"
)

print(
    f"Size: "
    f"min={sizes.min():.2f}, "
    f"max={sizes.max():.2f}, "
    f"mean={sizes.mean():.2f}"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)