from pathlib import Path

import pandas as pd
import numpy as np
import h5py
from pyproj import CRS, Transformer


# ============================================================
# PROJECT PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[2]

MAX_DIFF_HOURS = 1.0

CROP = 128
HALF = CROP // 2


# ============================================================
# INSAT CALIBRATION COEFFICIENTS
# ============================================================

TIR1 = (
    -3.8403e-7,
    0.00160569,
    -0.0209949,
)

WV = (
    -6.24616e-8,
    0.00084133,
    -0.0036705,
)


# ============================================================
# INSAT PROJECTION
# ============================================================

CRS_INSAT = CRS.from_proj4(
    "+proj=merc +lon_0=75 +lat_ts=17.75 +datum=WGS84"
)

TRANSFORMER = Transformer.from_crs(
    "EPSG:4326",
    CRS_INSAT,
    always_xy=True,
)


# ============================================================
# CALIBRATION
# ============================================================

def calibrate(x, coeff):
    """
    Apply quadratic radiometric calibration:

        y = a*x^2 + b*x + c
    """

    a, b, c = coeff

    x = x.astype(np.float32)

    return a * x * x + b * x + c


# ============================================================
# HDF5 TIMESTAMP
# ============================================================

def get_time(path):
    """
    Parse timestamp from filenames such as:

    3DIMG_03MAY2019_0930_L1C_ASIA_MER_V01R00.h5
    """

    try:
        parts = path.name.split("_")

        # parts:
        # 0 -> 3DIMG
        # 1 -> 03MAY2019
        # 2 -> 0930

        return pd.to_datetime(
            parts[1] + parts[2],
            format="%d%b%Y%H%M",
            utc=True,
        )

    except Exception:
        return pd.NaT


# ============================================================
# HDF5 VALIDATION
# ============================================================

def valid_h5(path):
    """
    Check whether the required INSAT datasets exist.
    """

    try:
        with h5py.File(path, "r") as f:

            required = [
                "IMG_TIR1",
                "IMG_WV",
                "X",
                "Y",
            ]

            return all(k in f for k in required)

    except Exception:
        return False


# ============================================================
# LAT/LON -> PIXEL
# ============================================================

def pixel_from_latlon(path, lat, lon):
    """
    Convert geographic latitude/longitude to the nearest
    INSAT pixel using the X/Y projection arrays.
    """

    with h5py.File(path, "r") as f:

        x = np.asarray(f["X"][:])
        y = np.asarray(f["Y"][:])

    px, py = TRANSFORMER.transform(
        lon,
        lat,
    )

    col = int(
        np.argmin(
            np.abs(x - px)
        )
    )

    row = int(
        np.argmin(
            np.abs(y - py)
        )
    )

    return row, col


# ============================================================
# MAIN PROCESSING
# ============================================================

def process():

    storm = "FANI"

    # --------------------------------------------------------
    # INPUT / OUTPUT PATHS
    # --------------------------------------------------------

    best = (
        BASE
        / "data"
        / "labels"
        / "fani_besttrack.csv"
    )

    insat_root = (
        BASE
        / "data"
        / "raw"
        / "mosdac"
        / "fani"
        / "3DIMG_L1C_ASIA_MER"
    )

    out_root = (
        BASE
        / "data"
        / "processed"
        / "insat_fani_calibrated"
    )

    manifest_out = (
        BASE
        / "data"
        / "labels"
        / "fani_insat_manifest.csv"
    )

    print()
    print("========== FANI / MOSDAC ==========")

    print("Raw root:", insat_root)
    print("Best-track:", best)

    # --------------------------------------------------------
    # CHECK INPUTS
    # --------------------------------------------------------

    if not best.exists():
        raise FileNotFoundError(
            f"Missing best-track: {best}"
        )

    if not insat_root.exists():
        raise FileNotFoundError(
            f"Missing MOSDAC directory: {insat_root}"
        )

    # --------------------------------------------------------
    # LOAD BEST TRACK
    # --------------------------------------------------------

    bt = pd.read_csv(best)

    # Your standardized FANI file already contains timestamps
    # in ISO-like format, so dayfirst=True is unnecessary.
    bt["timestamp"] = pd.to_datetime(
        bt["timestamp"],
        utc=True,
    )

    print(
        "Best-track rows:",
        len(bt),
    )

    # --------------------------------------------------------
    # CHECK REQUIRED BEST-TRACK COLUMNS
    # --------------------------------------------------------

    required_bt_columns = [
        "timestamp",
        "lat",
        "lon",
        "wind_kt",
        "pressure_hpa",
        "category",
    ]

    missing_columns = [
        col
        for col in required_bt_columns
        if col not in bt.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required best-track columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # FIND VALID HDF5 FILES
    # --------------------------------------------------------

    files = []

    for p in insat_root.rglob("*.h5"):

        timestamp = get_time(p)

        if pd.isna(timestamp):
            continue

        if valid_h5(p):
            files.append(
                (
                    p,
                    timestamp,
                )
            )

    print(
        "Valid INSAT HDF5:",
        len(files),
    )

    if not files:
        raise RuntimeError(
            "No valid INSAT HDF5 files found."
        )

    # Sort chronologically
    files.sort(
        key=lambda x: x[1]
    )

    # --------------------------------------------------------
    # MATCH BEST-TRACK OBSERVATIONS TO INSAT
    # --------------------------------------------------------

    used = set()

    rows = []

    skipped_no_match = 0
    skipped_crop = 0
    errors = 0

    for _, label in bt.iterrows():

        target = label["timestamp"]

        # Find closest unused INSAT observation
        candidates = sorted(
            files,
            key=lambda z: abs(
                (z[1] - target).total_seconds()
            ),
        )

        selected = None

        for path, timestamp in candidates:

            if path in used:
                continue

            diff = (
                abs(
                    (timestamp - target)
                    .total_seconds()
                )
                / 3600.0
            )

            if diff <= MAX_DIFF_HOURS:

                selected = (
                    path,
                    timestamp,
                    diff,
                )

                break

        # ----------------------------------------------------
        # NO MATCH
        # ----------------------------------------------------

        if selected is None:

            skipped_no_match += 1

            continue

        path, insat_time, diff = selected

        used.add(path)

        # ----------------------------------------------------
        # PROCESS MATCHED IMAGE
        # ----------------------------------------------------

        try:

            # -----------------------------------------------
            # LAT/LON -> PIXEL
            # -----------------------------------------------

            row, col = pixel_from_latlon(
                path,
                float(label["lat"]),
                float(label["lon"]),
            )

            # -----------------------------------------------
            # READ IMAGE CHANNELS
            # -----------------------------------------------

            with h5py.File(path, "r") as f:

                tir = np.asarray(
                    f["IMG_TIR1"][0]
                )

                wv = np.asarray(
                    f["IMG_WV"][0]
                )

            # -----------------------------------------------
            # 128 x 128 CROP
            # -----------------------------------------------

            r0 = row - HALF
            r1 = row + HALF

            c0 = col - HALF
            c1 = col + HALF

            if (
                r0 < 0
                or c0 < 0
                or r1 > tir.shape[0]
                or c1 > tir.shape[1]
            ):

                skipped_crop += 1

                continue

            tir_crop = tir[
                r0:r1,
                c0:c1,
            ]

            wv_crop = wv[
                r0:r1,
                c0:c1,
            ]

            # -----------------------------------------------
            # CALIBRATION
            # -----------------------------------------------

            tir_rad = calibrate(
                tir_crop,
                TIR1,
            )

            wv_rad = calibrate(
                wv_crop,
                WV,
            )

            # -----------------------------------------------
            # SHAPE CHECK
            # -----------------------------------------------

            if (
                tir_rad.shape != (128, 128)
                or wv_rad.shape != (128, 128)
            ):

                skipped_crop += 1

                continue

            # -----------------------------------------------
            # NUMERICAL VALIDATION
            # -----------------------------------------------

            if (
                not np.isfinite(tir_rad).all()
                or not np.isfinite(wv_rad).all()
            ):

                raise ValueError(
                    "Calibrated crop contains NaN/Inf"
                )

            # -----------------------------------------------
            # OUTPUT DIRECTORY
            # -----------------------------------------------

            out = (
                out_root
                / path.stem
            )

            out.mkdir(
                parents=True,
                exist_ok=True,
            )

            # -----------------------------------------------
            # SAVE ARRAYS
            # -----------------------------------------------

            np.save(
                out / "tir1_radiance.npy",
                tir_rad.astype(
                    np.float32
                ),
            )

            np.save(
                out / "wv_radiance.npy",
                wv_rad.astype(
                    np.float32
                ),
            )

            # -----------------------------------------------
            # MANIFEST ROW
            # -----------------------------------------------

            rows.append(
                {
                    "cyclone": storm,

                    "label_timestamp":
                        target.isoformat(),

                    "insat_timestamp":
                        insat_time.isoformat(),

                    "time_diff_hours":
                        diff,

                    "latitude":
                        label["lat"],

                    "longitude":
                        label["lon"],

                    "wind_kt":
                        label["wind_kt"],

                    "pressure_hpa":
                        label["pressure_hpa"],

                    # IMPORTANT:
                    # FANI best-track uses `category`,
                    # NOT `grade`.
                    "category":
                        label["category"],

                    "pixel_row":
                        row,

                    "pixel_col":
                        col,

                    "file":
                        str(path),
                }
            )

            print(
                f"OK  {path.name} "
                f"| dt={diff:.2f}h "
                f"| pixel=({row},{col})"
            )

        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        except Exception as e:

            errors += 1

            print(
                "ERROR:",
                path.name,
                e,
            )

    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(rows)

    df.to_csv(
        manifest_out,
        index=False,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("========== RESULT ==========")

    print(
        "Matched/calibrated:",
        len(rows),
    )

    print(
        "No timestamp match:",
        skipped_no_match,
    )

    print(
        "Invalid crop:",
        skipped_crop,
    )

    print(
        "Errors:",
        errors,
    )

    if rows:

        print(
            "Mean delta-t:",
            df["time_diff_hours"].mean(),
        )

        print(
            "Max delta-t:",
            df["time_diff_hours"].max(),
        )

        print()
        print(
            "Categories:"
        )

        print(
            df["category"].value_counts()
        )

    print()
    print(
        "Output:",
        out_root,
    )

    print(
        "Manifest:",
        manifest_out,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    process()