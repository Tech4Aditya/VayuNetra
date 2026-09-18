from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]

INPUT = BASE / "data/processed/insat_bulbul"
OUTPUT = BASE / "data/processed/insat_bulbul_calibrated"

TIR1 = (-3.8403e-7, 0.00160569, -0.0209949)
WV = (-6.24616e-8, 0.00084133, -0.0036705)


def calibrate(x, coeff):
    a, b, c = coeff
    return (a * x.astype(np.float32) ** 2
            + b * x.astype(np.float32)
            + c)


processed = 0

for d in INPUT.iterdir():

    if not d.is_dir():
        continue

    tir_path = d / "tir1.npy"
    wv_path = d / "wv.npy"

    if not tir_path.exists() or not wv_path.exists():
        continue

    out = OUTPUT / d.name
    out.mkdir(parents=True, exist_ok=True)

    tir = np.load(tir_path)
    wv = np.load(wv_path)

    tir_rad = calibrate(tir, TIR1)
    wv_rad = calibrate(wv, WV)

    np.save(out / "tir1_radiance.npy", tir_rad)
    np.save(out / "wv_radiance.npy", wv_rad)

    # Preserve raw arrays
    np.save(out / "tir1.npy", tir)
    np.save(out / "wv.npy", wv)

    processed += 1

print()
print("========== BULBUL CALIBRATION ==========")
print("Processed:", processed)
print("Skipped  :", 12 - processed)
print("Output   :", OUTPUT)