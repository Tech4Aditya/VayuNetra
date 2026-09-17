import numpy as np
import pandas as pd

LABEL_PATH = "backend/data/raw/tcir/Cyclone_Labels h5.npy"

labels = np.load(LABEL_PATH, allow_pickle=True)

df = pd.DataFrame(
    labels,
    columns=[
        "basin",
        "cyclone_id",
        "lon",
        "lat",
        "timestamp",
        "wind_kt",
        "size",
        "pressure"
    ]
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"].astype(str),
    format="%Y%m%d%H"
)

df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")
df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
df["size"] = pd.to_numeric(df["size"], errors="coerce")

print("=" * 70)
print("TCIR SEQUENCE ANALYSIS")
print("=" * 70)

print("\nTotal frames:", len(df))
print("Unique cyclones:", df["cyclone_id"].nunique())

print("\nFrames per cyclone:")
counts = df.groupby("cyclone_id").size()

print(counts.describe())

print("\nMinimum frames:", counts.min())
print("Maximum frames:", counts.max())
print("Median frames:", counts.median())

# --------------------------------------------------
# Temporal gaps
# --------------------------------------------------

df = df.sort_values(["cyclone_id", "timestamp"])

df["time_diff_hours"] = (
    df.groupby("cyclone_id")["timestamp"]
      .diff()
      .dt.total_seconds()
      / 3600
)

print("\nTemporal gaps:")
print(df["time_diff_hours"].value_counts().sort_index())

# --------------------------------------------------
# Wind change
# --------------------------------------------------

df["wind_change"] = (
    df.groupby("cyclone_id")["wind_kt"]
      .diff()
)

print("\nWind change statistics:")
print(df["wind_change"].describe())

# --------------------------------------------------
# Basin distribution
# --------------------------------------------------

print("\nFrames by basin:")
print(df["basin"].value_counts())

print("\nCyclones by basin:")
print(df.groupby("basin")["cyclone_id"].nunique())

# --------------------------------------------------
# Example storms
# --------------------------------------------------

print("\nExample cyclone sequences:")

for cyclone_id, group in df.groupby("cyclone_id"):
    if len(group) >= 8:
        print("\nCyclone:", cyclone_id)
        print("Basin:", group["basin"].iloc[0])
        print("Frames:", len(group))
        print(
            group[
                ["timestamp", "wind_kt", "pressure"]
            ].head(8).to_string(index=False)
        )

        break

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)