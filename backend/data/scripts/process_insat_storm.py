from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import h5py
from pyproj import CRS, Transformer

BASE = Path(__file__).resolve().parents[2]

MAX_DIFF_HOURS = 1.0
CROP = 128
HALF = CROP // 2

TIR1 = (-3.8403e-7, 0.00160569, -0.0209949)
WV   = (-6.24616e-8, 0.00084133, -0.0036705)

CRS_INSAT = CRS.from_proj4(
    "+proj=merc +lon_0=75 +lat_ts=17.75 +datum=WGS84"
)

TRANSFORMER = Transformer.from_crs(
    "EPSG:4326", CRS_INSAT, always_xy=True
)


def calibrate(x, coeff):
    a, b, c = coeff
    x = x.astype(np.float32)
    return a*x*x + b*x + c


def get_time(path):
    try:
        parts = path.name.split("_")
        return pd.to_datetime(
            parts[1] + parts[2],
            format="%d%b%Y%H%M",
            utc=True
        )
    except Exception:
        return pd.NaT


def valid_h5(path):
    try:
        with h5py.File(path, "r") as f:
            return all(
                k in f for k in
                ["IMG_TIR1", "IMG_WV", "X", "Y"]
            )
    except Exception:
        return False


def pixel_from_latlon(path, lat, lon):
    with h5py.File(path, "r") as f:
        x = np.asarray(f["X"][:])
        y = np.asarray(f["Y"][:])

    px, py = TRANSFORMER.transform(lon, lat)

    col = int(np.argmin(np.abs(x - px)))
    row = int(np.argmin(np.abs(y - py)))

    return row, col


def process(storm):

    best = BASE / f"data/labels/{storm.lower()}_besttrack.csv"
    insat_root = BASE / "data/raw/insat/3DIMG_L1C_ASIA_MER"

    out_root = BASE / f"data/processed/insat_{storm.lower()}_calibrated"
    manifest_out = BASE / f"data/labels/{storm.lower()}_insat_manifest.csv"

    print(f"\n========== {storm.upper()} ==========")

    bt = pd.read_csv(best)
    bt["timestamp"] = pd.to_datetime(
        bt["timestamp"], utc=True
    )

    print("Best-track:", len(bt))

    files = []

    for p in insat_root.rglob("*.h5"):

        t = get_time(p)

        if pd.isna(t):
            continue

        if valid_h5(p):
            files.append((p, t))

    print("Valid INSAT:", len(files))

    used = set()
    rows = []

    for _, label in bt.iterrows():

        target = label["timestamp"]

        candidates = sorted(
            files,
            key=lambda z:
            abs((z[1] - target).total_seconds())
        )

        selected = None

        for path, timestamp in candidates:

            if path in used:
                continue

            diff = abs(
                (timestamp - target).total_seconds()
            ) / 3600

            if diff <= MAX_DIFF_HOURS:
                selected = path, timestamp, diff
                break

        if selected is None:
            continue

        path, insat_time, diff = selected
        used.add(path)

        try:
            row, col = pixel_from_latlon(
                path,
                float(label["latitude"]),
                float(label["longitude"])
            )

            with h5py.File(path, "r") as f:
                tir = np.asarray(f["IMG_TIR1"][0])
                wv = np.asarray(f["IMG_WV"][0])

            r0 = row - HALF
            r1 = row + HALF
            c0 = col - HALF
            c1 = col + HALF

            if (
                r0 < 0 or c0 < 0 or
                r1 > tir.shape[0] or
                c1 > tir.shape[1]
            ):
                continue

            tir_crop = tir[r0:r1, c0:c1]
            wv_crop = wv[r0:r1, c0:c1]

            tir_rad = calibrate(tir_crop, TIR1)
            wv_rad = calibrate(wv_crop, WV)

            out = out_root / path.stem
            out.mkdir(parents=True, exist_ok=True)

            np.save(
                out / "tir1_radiance.npy",
                tir_rad
            )

            np.save(
                out / "wv_radiance.npy",
                wv_rad
            )

            rows.append({
                "cyclone": storm.upper(),
                "label_timestamp": target.isoformat(),
                "insat_timestamp": insat_time.isoformat(),
                "time_diff_hours": diff,
                "latitude": label["latitude"],
                "longitude": label["longitude"],
                "wind_kt": label["wind_kt"],
                "pressure_hpa": label["pressure_hpa"],
                "category": label["category"],
                "pixel_row": row,
                "pixel_col": col,
                "file": str(path)
            })

        except Exception as e:
            print("ERROR:", path.name, e)

    out_df = pd.DataFrame(rows)

    manifest_out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out_df.to_csv(
        manifest_out,
        index=False
    )

    print()
    print("Matched:", len(out_df))
    print(
        "Mean Δt:",
        out_df["time_diff_hours"].mean()
        if len(out_df) else "N/A"
    )
    print(
        "Max Δt:",
        out_df["time_diff_hours"].max()
        if len(out_df) else "N/A"
    )

    print("Output:", out_root)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "storm",
        help="Storm name, e.g. AMPHAN"
    )

    args = parser.parse_args()

    process(args.storm)