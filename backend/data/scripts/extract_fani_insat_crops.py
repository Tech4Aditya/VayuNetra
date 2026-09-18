import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image


MANIFEST = "backend/data/labels/fani_insat_manifest_strict.csv"

OUTPUT_ROOT = "backend/data/processed/insat_fani"

CROP_SIZE = 128


def normalize_uint16(arr):
    arr = arr.astype(np.float32)

    # Remove obvious fill value
    arr[arr >= 1023] = np.nan

    valid = np.isfinite(arr)

    if not valid.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    lo = np.nanpercentile(arr, 1)
    hi = np.nanpercentile(arr, 99)

    arr = np.clip(arr, lo, hi)

    arr = (arr - lo) / (hi - lo + 1e-8)
    arr = (arr * 255).astype(np.uint8)

    return arr


def extract_crop(arr, row, col, size=128):

    half = size // 2

    r0 = row - half
    r1 = r0 + size

    c0 = col - half
    c1 = c0 + size

    # Safety check
    if r0 < 0 or c0 < 0:
        return None

    if r1 > arr.shape[0] or c1 > arr.shape[1]:
        return None

    return arr[r0:r1, c0:c1]


df = pd.read_csv(MANIFEST)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("Pairs:", len(df))

for idx, item in df.iterrows():

    filename = item["filename"]

    matches = []

    for root, _, files in os.walk(
        "backend/data/raw/insat/3DIMG_L1C_ASIA_MER/2019"
    ):
        if filename in files:
            matches.append(
                os.path.join(root, filename)
            )

    if not matches:
        print("MISSING:", filename)
        continue

    path = matches[0]

    row = int(item["pixel_row"])
    col = int(item["pixel_col"])

    with h5py.File(path, "r") as h:

        tir1 = np.asarray(
            h["IMG_TIR1"][0]
        )

        wv = np.asarray(
            h["IMG_WV"][0]
        )

    tir_crop = extract_crop(
        tir1,
        row,
        col
    )

    wv_crop = extract_crop(
        wv,
        row,
        col
    )

    if tir_crop is None or wv_crop is None:
        print("CROP ERROR:", filename)
        continue

    tir_img = normalize_uint16(tir_crop)
    wv_img = normalize_uint16(wv_crop)

    stem = os.path.splitext(filename)[0]

    out_dir = os.path.join(
        OUTPUT_ROOT,
        stem
    )

    os.makedirs(out_dir, exist_ok=True)

    Image.fromarray(tir_img).save(
        os.path.join(
            out_dir,
            "tir1.png"
        )
    )

    Image.fromarray(wv_img).save(
        os.path.join(
            out_dir,
            "wv.png"
        )
    )

    # Save raw arrays too — these are more useful for ML
    np.save(
        os.path.join(
            out_dir,
            "tir1.npy"
        ),
        tir_crop
    )

    np.save(
        os.path.join(
            out_dir,
            "wv.npy"
        ),
        wv_crop
    )

    print(
        f"[{idx + 1}/{len(df)}] "
        f"{stem} | "
        f"wind={item['wind_kt']} kt | "
        f"crop=128x128 | "
        f"pixel=({row},{col})"
    )


print()
print("DONE")
print("Output:", OUTPUT_ROOT)