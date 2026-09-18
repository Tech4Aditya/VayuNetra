import os
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT
    / "data"
    / "labels"
    / "tcir_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "labels"
    / "tcir_track_sequences"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

SEQUENCE_LENGTH = 4
INTERVAL_HOURS = 3

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TCIR TRACK SEQUENCE GENERATION")
print("=" * 70)

print(
    f"\nReading:\n{MANIFEST_PATH}"
)

if not MANIFEST_PATH.exists():

    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST_PATH}"
    )


df = pd.read_csv(
    MANIFEST_PATH
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "basin",
    "cyclone_id",
    "lon",
    "lat",
    "timestamp",
    "h5_index",
    "datetime",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing
        )
    )


# ============================================================
# DATETIME
# ============================================================

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

if df["datetime"].isna().any():

    raise ValueError(
        "Invalid datetime values found."
    )


# ============================================================
# NUMERIC COORDINATES
# ============================================================

df["lat"] = pd.to_numeric(
    df["lat"],
    errors="coerce"
)

df["lon"] = pd.to_numeric(
    df["lon"],
    errors="coerce"
)

df["h5_index"] = pd.to_numeric(
    df["h5_index"],
    errors="coerce"
)


if df[
    ["lat", "lon", "h5_index"]
].isna().any().any():

    raise ValueError(
        "Missing lat/lon/h5_index values found."
    )


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "cyclone_id",
        "datetime",
    ]
).reset_index(
    drop=True
)


print(
    f"\nObservations : {len(df):,}"
)

print(
    f"Cyclones     : "
    f"{df['cyclone_id'].nunique():,}"
)


# ============================================================
# LONGITUDE WRAP
# ============================================================

def longitude_delta(
    lon_start,
    lon_end
):
    """
    Return shortest longitudinal displacement.

    Example:

        179 -> -179

    becomes:

        +2 degrees

    rather than:

        -358 degrees
    """

    delta = (
        lon_end -
        lon_start
    )

    return (
        (
            delta + 180.0
        ) % 360.0
    ) - 180.0


# ============================================================
# CREATE CANDIDATE SEQUENCES
# ============================================================

samples = []


for cyclone_id, storm in df.groupby(
    "cyclone_id",
    sort=False
):

    storm = storm.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    if len(storm) < (
        SEQUENCE_LENGTH + 1
    ):
        continue


    for start in range(
        0,
        len(storm) -
        SEQUENCE_LENGTH
    ):

        sequence = storm.iloc[
            start:
            start + SEQUENCE_LENGTH + 1
        ]


        # ----------------------------------------------------
        # Check exact 3-hour cadence
        # ----------------------------------------------------

        gaps = (
            sequence["datetime"]
            .diff()
            .dt.total_seconds()
            .div(3600)
            .iloc[1:]
        )

        if not np.allclose(
            gaps.to_numpy(),
            INTERVAL_HOURS
        ):
            continue


        # ----------------------------------------------------
        # Four input frames
        # ----------------------------------------------------

        input_frames = sequence.iloc[
            :SEQUENCE_LENGTH
        ]

        target = sequence.iloc[
            SEQUENCE_LENGTH
        ]


        # ----------------------------------------------------
        # Current position
        # ----------------------------------------------------

        current = input_frames.iloc[
            -1
        ]

        current_lat = float(
            current["lat"]
        )

        current_lon = float(
            current["lon"]
        )


        # ----------------------------------------------------
        # Future position
        # ----------------------------------------------------

        target_lat = float(
            target["lat"]
        )

        target_lon = float(
            target["lon"]
        )


        # ----------------------------------------------------
        # Displacement
        # ----------------------------------------------------

        delta_lat = (
            target_lat -
            current_lat
        )

        delta_lon = longitude_delta(
            current_lon,
            target_lon
        )


        # ----------------------------------------------------
        # Store sample
        # ----------------------------------------------------

        row = {

            "cyclone_id":
                str(cyclone_id),

            "basin":
                str(
                    input_frames.iloc[
                        0
                    ]["basin"]
                ),

            "target_timestamp":
                str(
                    target["timestamp"]
                ),

            "target_datetime":
                target["datetime"],

            "current_lat":
                current_lat,

            "current_lon":
                current_lon,

            "target_lat":
                target_lat,

            "target_lon":
                target_lon,

            "delta_lat":
                delta_lat,

            "delta_lon":
                delta_lon,
        }


        # ----------------------------------------------------
        # HDF5 frame indices
        # ----------------------------------------------------

        for i in range(
            SEQUENCE_LENGTH
        ):

            row[
                f"frame_{i}_h5_index"
            ] = int(
                input_frames.iloc[
                    i
                ]["h5_index"]
            )


        samples.append(
            row
        )


# ============================================================
# DATAFRAME
# ============================================================

samples_df = pd.DataFrame(
    samples
)


if len(samples_df) == 0:

    raise RuntimeError(
        "No valid track sequences were created."
    )


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(
    samples_df
)

samples_df = samples_df.drop_duplicates(
    subset=[
        "cyclone_id",
        "target_timestamp",
    ]
).reset_index(
    drop=True
)

removed = (
    before -
    len(samples_df)
)


# ============================================================
# STORM-LEVEL SPLIT
# ============================================================

cyclones = np.array(
    sorted(
        samples_df[
            "cyclone_id"
        ].unique()
    )
)

rng = np.random.default_rng(
    RANDOM_SEED
)

rng.shuffle(
    cyclones
)


n_cyclones = len(
    cyclones
)

n_train = int(
    n_cyclones *
    TRAIN_RATIO
)

n_val = int(
    n_cyclones *
    VAL_RATIO
)

train_cyclones = set(
    cyclones[
        :n_train
    ]
)

val_cyclones = set(
    cyclones[
        n_train:
        n_train + n_val
    ]
)

test_cyclones = set(
    cyclones[
        n_train + n_val:
    ]
)


# ============================================================
# APPLY SPLIT
# ============================================================

train_df = samples_df[
    samples_df[
        "cyclone_id"
    ].isin(
        train_cyclones
    )
].copy()

val_df = samples_df[
    samples_df[
        "cyclone_id"
    ].isin(
        val_cyclones
    )
].copy()

test_df = samples_df[
    samples_df[
        "cyclone_id"
    ].isin(
        test_cyclones
    )
].copy()


# ============================================================
# SAVE
# ============================================================

train_path = (
    OUTPUT_DIR /
    "train.csv"
)

val_path = (
    OUTPUT_DIR /
    "val.csv"
)

test_path = (
    OUTPUT_DIR /
    "test.csv"
)

all_path = (
    OUTPUT_DIR /
    "all.csv"
)


train_df.to_csv(
    train_path,
    index=False
)

val_df.to_csv(
    val_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)

samples_df.to_csv(
    all_path,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("TRACK DATASET CREATED")
print("=" * 70)

print(
    f"\nTotal sequences : "
    f"{len(samples_df):,}"
)

print(
    f"Removed duplicates : "
    f"{removed:,}"
)

print(
    f"Cyclones : "
    f"{samples_df['cyclone_id'].nunique():,}"
)


print("\nSplit:")

print(
    f"Train : "
    f"{len(train_df):,} sequences / "
    f"{train_df['cyclone_id'].nunique():,} cyclones"
)

print(
    f"Val   : "
    f"{len(val_df):,} sequences / "
    f"{val_df['cyclone_id'].nunique():,} cyclones"
)

print(
    f"Test  : "
    f"{len(test_df):,} sequences / "
    f"{test_df['cyclone_id'].nunique():,} cyclones"
)


# ============================================================
# CHECK LEAKAGE
# ============================================================

train_ids = set(
    train_df[
        "cyclone_id"
    ]
)

val_ids = set(
    val_df[
        "cyclone_id"
    ]
)

test_ids = set(
    test_df[
        "cyclone_id"
    ]
)


assert not (
    train_ids &
    val_ids
)

assert not (
    train_ids &
    test_ids
)

assert not (
    val_ids &
    test_ids
)


print(
    "\n✓ No cyclone leakage between splits."
)


# ============================================================
# TARGET STATISTICS
# ============================================================

print("\nTarget displacement statistics:")

print(
    samples_df[
        [
            "delta_lat",
            "delta_lon",
        ]
    ].describe().to_string()
)


# ============================================================
# SANITY CHECK
# ============================================================

print(
    "\nSample sequences:"
)

display_columns = [
    "cyclone_id",
    "target_timestamp",
    "current_lat",
    "current_lon",
    "target_lat",
    "target_lon",
    "delta_lat",
    "delta_lon",
    "frame_0_h5_index",
    "frame_1_h5_index",
    "frame_2_h5_index",
    "frame_3_h5_index",
]


print(
    samples_df[
        display_columns
    ].head(10).to_string(
        index=False
    )
)


# ============================================================
# OUTPUT
# ============================================================

print("\nSaved:")

print(
    train_path
)

print(
    val_path
)

print(
    test_path
)

print(
    all_path
)

print(
    "\nDone."
)