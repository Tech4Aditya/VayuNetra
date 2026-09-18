from pathlib import Path
import pandas as pd
import numpy as np
import h5py

BASE = Path(__file__).resolve().parents[2]

MANIFEST = BASE / "data/labels/bulbul_insat_manifest.csv"
OUT_ROOT = BASE / "data/processed/insat_bulbul"

CROP = 128
HALF = CROP // 2

df = pd.read_csv(MANIFEST)

print("BULBUL samples:", len(df))

processed = 0

for _, r in df.iterrows():

    path = Path(r["file"])

    if not path.exists():
        print("Missing:", path)
        continue

    stem = path.stem
    out = OUT_ROOT / stem
    out.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(path, "r") as f:
            tir = np.asarray(f["IMG_TIR1"][0])
            wv = np.asarray(f["IMG_WV"][0])

        row = int(r["pixel_row"])
        col = int(r["pixel_col"])

        r0 = row - HALF
        r1 = row + HALF
        c0 = col - HALF
        c1 = col + HALF

        if r0 < 0 or c0 < 0 or r1 > tir.shape[0] or c1 > tir.shape[1]:
            print("Crop outside image:", stem)
            continue

        tir_crop = tir[r0:r1, c0:c1]
        wv_crop = wv[r0:r1, c0:c1]

        np.save(out / "tir1.npy", tir_crop)
        np.save(out / "wv.npy", wv_crop)

        pd.DataFrame([{
            "cyclone": "BULBUL",
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "wind_kt": r["wind_kt"],
            "pressure_hpa": r["pressure_hpa"],
            "category": r["category"],
            "label_timestamp": r["label_timestamp"],
            "insat_timestamp": r["insat_timestamp"],
            "time_diff_hours": r["time_diff_hours"],
            "pixel_row": row,
            "pixel_col": col
        }]).to_csv(out / "metadata.csv", index=False)

        processed += 1

    except Exception as e:
        print("ERROR:", stem, e)

print()
print("========== BULBUL CROPS ==========")
print("Processed:", processed)
print("Skipped  :", len(df) - processed)
print("Output   :", OUT_ROOT)