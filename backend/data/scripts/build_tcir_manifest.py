import numpy as np
import pandas as pd

LABEL_PATH = "backend/data/raw/tcir/Cyclone_Labels h5.npy"
OUTPUT_PATH = "backend/data/labels/tcir_manifest.csv"

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
        "size_nmi",
        "pressure_hpa"
    ]
)

# Keep the FIRST occurrence of every cyclone + timestamp.
# The H5 image index is preserved so we can lazily read it later.
df["h5_index"] = np.arange(len(df))

before = len(df)

df = df.drop_duplicates(
    subset=["cyclone_id", "timestamp"],
    keep="first"
).reset_index(drop=True)

after = len(df)

# Parse timestamp
df["datetime"] = pd.to_datetime(
    df["timestamp"].astype(str),
    format="%Y%m%d%H"
)

# Numeric labels
df["lon"] = pd.to_numeric(df["lon"])
df["lat"] = pd.to_numeric(df["lat"])
df["wind_kt"] = pd.to_numeric(df["wind_kt"])
df["size_nmi"] = pd.to_numeric(df["size_nmi"])
df["pressure_hpa"] = pd.to_numeric(df["pressure_hpa"])

# Sort properly
df = df.sort_values(
    ["cyclone_id", "datetime"]
).reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("TCIR MANIFEST")
print("=" * 70)

print("Original rows :", before)
print("Clean rows    :", after)
print("Removed       :", before - after)

print("Cyclones      :", df["cyclone_id"].nunique())
print("Basins        :", df["basin"].nunique())

print("\nFrames by basin:")
print(df["basin"].value_counts())

print("\nFrames per cyclone:")
print(df.groupby("cyclone_id").size().describe())

print("\nSaved:")
print(OUTPUT_PATH)

print("=" * 70)