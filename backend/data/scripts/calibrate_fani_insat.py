import os
import numpy as np
import pandas as pd


ROOT = "backend/data/processed/insat_fani"
MANIFEST = "backend/data/labels/fani_insat_manifest_strict.csv"

# INSAT-3D laboratory radiance calibration
CAL = {
    "tir1": {
        "a": -3.8403e-7,
        "b":  0.00160569,
        "c": -0.0209949,
    },
    "wv": {
        "a": -6.24616e-8,
        "b":  0.00084133,
        "c": -0.0036705,
    },
}


def calibrate(counts, p):
    x = counts.astype(np.float32)

    # Fill value
    mask = x >= 1023

    radiance = (
        p["a"] * x * x
        + p["b"] * x
        + p["c"]
    )

    radiance[mask] = np.nan

    return radiance.astype(np.float32)


df = pd.read_csv(MANIFEST)

print("Samples:", len(df))

for _, row in df.iterrows():

    stem = os.path.splitext(row["filename"])[0]
    folder = os.path.join(ROOT, stem)

    for channel in ["tir1", "wv"]:

        src = os.path.join(
            folder,
            f"{channel}.npy"
        )

        dst = os.path.join(
            folder,
            f"{channel}_radiance.npy"
        )

        counts = np.load(src)

        radiance = calibrate(
            counts,
            CAL[channel]
        )

        np.save(dst, radiance)

        valid = np.isfinite(radiance)

        print(
            f"{stem} | {channel.upper()} | "
            f"min={np.nanmin(radiance):.6f} | "
            f"max={np.nanmax(radiance):.6f} | "
            f"mean={np.nanmean(radiance):.6f}"
        )

print("\nDONE")