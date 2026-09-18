from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BESTTRACK = ROOT / "data" / "labels" / "tauktae_besttrack.csv"

INSAT_DIR = ROOT / "data" / "raw" / "insat_tauktae"

OUTPUT = ROOT / "data" / "labels" / "tauktae_insat_manifest.csv"


# ============================================================
# SETTINGS
# ============================================================

MAX_TIME_DIFF_HOURS = 1.0


# ============================================================
# INSAT-3D PROJECTION
# ============================================================

CRS_INSAT = CRS.from_proj4(
    "+proj=merc +lon_0=75 +lat_ts=17.75 +datum=WGS84"
)

TRANSFORMER = Transformer.from_crs(
    "EPSG:4326",
    CRS_INSAT,
    always_xy=True
)


# ============================================================
# TIMESTAMP FROM INSAT FILENAME
# ============================================================

def timestamp_from_filename(filename):
    """
    Extract timestamp from filenames such as:

    3DIMG_16MAY2021_1229_L1C_ASIA_MER_V01R00.h5
    """

    pattern = r"3DIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})"

    match = re.search(pattern, filename)

    if not match:
        return None

    date_part = match.group(1)
    time_part = match.group(2)

    value = f"{date_part}_{time_part}"

    timestamp = pd.to_datetime(
        value,
        format="%d%b%Y_%H%M",
        utc=True,
        errors="coerce"
    )

    return timestamp


# ============================================================
# LAT/LON → INSAT PIXEL
# ============================================================

def latlon_to_pixel(lat, lon, x, y):
    """
    Convert latitude/longitude to nearest INSAT pixel.

    INSAT-3D ASIA_MER:
        Mercator
        longitude of projection origin = 75°
        standard parallel = 17.75°
    """

    target_x, target_y = TRANSFORMER.transform(
        float(lon),
        float(lat)
    )

    col = int(np.argmin(np.abs(x - target_x)))
    row = int(np.argmin(np.abs(y - target_y)))

    return row, col


# ============================================================
# VALIDATE INSAT FILE
# ============================================================

def inspect_insat_file(path):
    """
    Validate an INSAT HDF5 file and extract its timestamp,
    X/Y projection arrays.
    """

    try:

        filename = path.name

        timestamp = timestamp_from_filename(filename)

        if timestamp is None:
            return None

        with h5py.File(path, "r") as f:

            required = [
                "IMG_TIR1",
                "IMG_WV",
                "X",
                "Y",
                "time"
            ]

            for key in required:
                if key not in f:
                    return None

            tir1_shape = f["IMG_TIR1"].shape
            wv_shape = f["IMG_WV"].shape

            if tir1_shape != (1, 1616, 1618):
                return None

            if wv_shape != (1, 1616, 1618):
                return None

            x = f["X"][:]
            y = f["Y"][:]

            if len(x) != 1618:
                return None

            if len(y) != 1616:
                return None

        return {
            "filename": filename,
            "path": str(path),
            "timestamp": timestamp,
            "x": x,
            "y": y
        }

    except Exception:
        return None


# ============================================================
# LOAD BEST TRACK
# ============================================================

def load_besttrack():

    if not BESTTRACK.exists():
        raise FileNotFoundError(
            f"Best-track file not found:\n{BESTTRACK}"
        )

    df = pd.read_csv(BESTTRACK)

    print(f"Best-track observations: {len(df)}")

    return df


# ============================================================
# FIND INSAT FILES
# ============================================================

def find_insat_files():

    files = sorted(
        INSAT_DIR.rglob("*.h5")
    )

    print(f"INSAT files found: {len(files)}")

    return files


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TAUKTAE → INSAT MANIFEST BUILDER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load best track
    # --------------------------------------------------------

    besttrack = load_besttrack()

    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    besttrack["timestamp"] = pd.to_datetime(
        besttrack["timestamp"],
        utc=True,
        errors="coerce"
    )

    besttrack = besttrack.dropna(
        subset=["timestamp"]
    ).copy()

    # --------------------------------------------------------
    # Find INSAT files
    # --------------------------------------------------------

    files = find_insat_files()

    # --------------------------------------------------------
    # Validate INSAT files
    # --------------------------------------------------------

    valid_insat = []
    invalid_count = 0

    print()
    print("Validating INSAT files...")

    for path in files:

        info = inspect_insat_file(path)

        if info is None:
            invalid_count += 1
        else:
            valid_insat.append(info)

    print(
        f"Valid INSAT files: {len(valid_insat)}"
    )

    print(
        f"Invalid INSAT files: {invalid_count}"
    )

    if not valid_insat:
        raise RuntimeError(
            "No valid INSAT files found."
        )

    # --------------------------------------------------------
    # Sort by timestamp
    # --------------------------------------------------------

    valid_insat.sort(
        key=lambda x: x["timestamp"]
    )

    # --------------------------------------------------------
    # ONE-TO-ONE MATCHING
    #
    # Each INSAT frame can only be used once.
    # --------------------------------------------------------

    used_insat = set()

    matches = []

    for _, label in besttrack.iterrows():

        label_time = label["timestamp"]

        best_candidate = None
        best_diff = float("inf")
        best_index = None

        for i, insat in enumerate(valid_insat):

            if i in used_insat:
                continue

            diff_hours = abs(
                (
                    insat["timestamp"] -
                    label_time
                ).total_seconds()
            ) / 3600.0

            if diff_hours < best_diff:

                best_diff = diff_hours
                best_candidate = insat
                best_index = i

        # ----------------------------------------------------
        # Accept only <= 1 hour difference
        # ----------------------------------------------------

        if (
            best_candidate is None
            or best_diff > MAX_TIME_DIFF_HOURS
        ):
            continue

        used_insat.add(best_index)

        # ----------------------------------------------------
        # Geographic → pixel mapping
        # ----------------------------------------------------

        lat = float(label["lat"])
        lon = float(label["lon"])

        pixel_row, pixel_col = latlon_to_pixel(
            lat,
            lon,
            best_candidate["x"],
            best_candidate["y"]
        )

        # ----------------------------------------------------
        # Build output record
        # ----------------------------------------------------

        record = {
            "filename":
                best_candidate["filename"],

            "insat_timestamp":
                best_candidate["timestamp"].isoformat(),

            "label_timestamp":
                label_time.isoformat(),

            "time_diff_hours":
                best_diff,

            "lat":
                lat,

            "lon":
                lon,

            "wind_kt":
                float(label["wind_kt"]),

            "pressure_hpa":
                (
                    float(label["pressure_hpa"])
                    if "pressure_hpa" in label
                    and pd.notna(label["pressure_hpa"])
                    else np.nan
                ),

            "category":
                (
                    label["category"]
                    if "category" in label
                    else ""
                ),

            "pixel_row":
                pixel_row,

            "pixel_col":
                pixel_col
        }

        matches.append(record)

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    result = pd.DataFrame(matches)

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    if not result.empty:

        result = result.sort_values(
            "label_timestamp"
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MATCHING COMPLETE")
    print("=" * 60)

    print(
        f"Best-track observations : {len(besttrack)}"
    )

    print(
        f"Valid INSAT frames      : {len(valid_insat)}"
    )

    print(
        f"Matched observations    : {len(result)}"
    )

    print(
        f"Unmatched observations  : "
        f"{len(besttrack) - len(result)}"
    )

    if not result.empty:

        print(
            f"Mean time difference    : "
            f"{result['time_diff_hours'].mean():.3f} h"
        )

        print(
            f"Max time difference     : "
            f"{result['time_diff_hours'].max():.3f} h"
        )

    else:

        print(
            "Mean time difference    : N/A"
        )

        print(
            "Max time difference     : N/A"
        )

    print()
    print(f"Output: {OUTPUT}")

    # --------------------------------------------------------
    # Print matches
    # --------------------------------------------------------

    if not result.empty:

        print()
        print("MATCHES:")

        columns = [
            "label_timestamp",
            "insat_timestamp",
            "time_diff_hours",
            "lat",
            "lon",
            "wind_kt",
            "pixel_row",
            "pixel_col"
        ]

        print(
            result[columns].to_string(
                index=False
            )
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()