import os
import glob
import h5py
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pyproj import CRS, Transformer


# ============================================================
# PATHS
# ============================================================

LABEL_PATH = "backend/data/labels/indian_cyclone_evaluation.csv"

INSAT_ROOT = (
    "backend/data/raw/insat/"
    "3DIMG_L1C_ASIA_MER/2019"
)

OUTPUT_PATH = "backend/data/labels/fani_insat_manifest_strict.csv"


# ============================================================
# FANI LABELS
# ============================================================

df = pd.read_csv(LABEL_PATH)

fani = df[df["storm_name"].eq("FANI")].copy()

fani["timestamp"] = pd.to_datetime(
    fani["timestamp"],
    utc=True
)

fani = fani.sort_values("timestamp").reset_index(drop=True)

print("FANI labels:", len(fani))
print(
    "FANI range:",
    fani["timestamp"].min(),
    "->",
    fani["timestamp"].max()
)


# ============================================================
# INSAT FILES
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(
            INSAT_ROOT,
            "**",
            "*.h5"
        ),
        recursive=True
    )
)

print("INSAT files found:", len(files))


# ============================================================
# PROJECTION
# ============================================================

crs_insat = CRS.from_proj4(
    "+proj=merc +lon_0=75 +lat_ts=17.75 +datum=WGS84"
)

transformer = Transformer.from_crs(
    CRS.from_epsg(4326),
    crs_insat,
    always_xy=True
)


# ============================================================
# READ VALID INSAT FILES
# ============================================================

insat_records = []

for path in files:

    try:

        with h5py.File(path, "r") as h:

            # Validate that required image exists
            shape = h["IMG_TIR1"].shape

            # Acquisition time
            root_attrs = h.attrs

            time_value = root_attrs.get(
                "Acquisition_Start_Time"
            )

            if time_value is None:
                continue

            if isinstance(time_value, bytes):
                time_value = time_value.decode()

            timestamp = pd.to_datetime(
                str(time_value),
                format="%d-%b-%YT%H:%M:%S",
                utc=True,
                errors="coerce"
            )

            if pd.isna(timestamp):
                continue

            X = np.asarray(h["X"][:])
            Y = np.asarray(h["Y"][:])

            insat_records.append(
                {
                    "filename": os.path.basename(path),
                    "path": path,
                    "timestamp": timestamp,
                    "X": X,
                    "Y": Y,
                    "shape": shape,
                }
            )

    except Exception as e:

        print(
            "SKIP BAD:",
            os.path.basename(path),
            "|",
            type(e).__name__,
            e
        )


print("Valid INSAT files:", len(insat_records))


# ============================================================
# STRICT ONE-TO-ONE MATCHING
# ============================================================

used_insat = set()
matches = []

MAX_DIFF_HOURS = 1.0


for _, label in fani.iterrows():

    label_time = label["timestamp"]

    candidates = []

    for i, insat in enumerate(insat_records):

        if i in used_insat:
            continue

        diff_hours = abs(
            (
                insat["timestamp"] - label_time
            ).total_seconds()
        ) / 3600.0

        if diff_hours <= MAX_DIFF_HOURS:

            candidates.append(
                (
                    diff_hours,
                    i,
                    insat
                )
            )

    if not candidates:
        continue

    # Nearest INSAT frame
    candidates.sort(
        key=lambda x: x[0]
    )

    diff_hours, idx, insat = candidates[0]

    used_insat.add(idx)


    # ========================================================
    # GEOLOCATION
    # ========================================================

    lon = float(label["lon"])
    lat = float(label["lat"])

    x_target, y_target = transformer.transform(
        lon,
        lat
    )

    col = int(
        np.argmin(
            np.abs(insat["X"] - x_target)
        )
    )

    row = int(
        np.argmin(
            np.abs(insat["Y"] - y_target)
        )
    )


    matches.append(
        {
            "filename": insat["filename"],
            "insat_timestamp": insat["timestamp"].isoformat(),
            "label_timestamp": label_time.isoformat(),
            "time_diff_hours": diff_hours,

            "lat": float(label["lat"]),
            "lon": float(label["lon"]),

            "wind_kt": float(label["wind_kt"]),

            "pressure_hpa": float(
                label["pressure_hpa"]
            ),

            "category": label["category"],

            "pixel_row": row,
            "pixel_col": col,
        }
    )


# ============================================================
# SAVE
# ============================================================

manifest = pd.DataFrame(matches)

manifest = manifest.sort_values(
    "label_timestamp"
).reset_index(drop=True)

manifest.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 60)
print("STRICT FANI INSAT MANIFEST")
print("=" * 60)

print(
    "Matched labels:",
    len(manifest)
)

if len(manifest) > 0:

    print(
        "Unique INSAT files:",
        manifest["filename"].nunique()
    )

    print(
        "Maximum time difference:",
        manifest["time_diff_hours"].max()
    )

    print(
        "Mean time difference:",
        manifest["time_diff_hours"].mean()
    )

    print()
    print(
        manifest[
            [
                "filename",
                "label_timestamp",
                "time_diff_hours",
                "lat",
                "lon",
                "wind_kt",
                "pixel_row",
                "pixel_col",
            ]
        ].to_string(index=False)
    )

print()
print("Output:")
print(OUTPUT_PATH)