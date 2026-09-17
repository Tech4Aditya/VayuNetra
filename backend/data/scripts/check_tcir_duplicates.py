import numpy as np
import pandas as pd
import h5py

LABEL_PATH = "backend/data/raw/tcir/Cyclone_Labels h5.npy"
IMAGE_PATH = "backend/data/raw/tcir/Cyclone_Images.h5"

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

duplicates = df[
    df.duplicated(
        subset=["cyclone_id", "timestamp"],
        keep=False
    )
]

print("=" * 70)
print("TCIR IMAGE DUPLICATE CHECK")
print("=" * 70)

print("Total frames:", len(df))
print("Duplicate-label rows:", len(duplicates))

checked = 0
identical_count = 0
different_count = 0

with h5py.File(IMAGE_PATH, "r") as f:

    images = f["Images"]

    for (cyclone_id, timestamp), group in duplicates.groupby(
        ["cyclone_id", "timestamp"]
    ):

        indices = group.index.tolist()

        if len(indices) != 2:
            continue

        i1, i2 = indices

        image1 = images[i1]
        image2 = images[i2]

        identical = np.array_equal(image1, image2)

        max_diff = np.max(
            np.abs(
                image1.astype(np.float32)
                - image2.astype(np.float32)
            )
        )

        print(
            f"\n{checked + 1:02d}. "
            f"{cyclone_id} | {timestamp}"
        )

        print("    indices:", i1, i2)
        print("    identical:", identical)
        print("    max pixel difference:", max_diff)

        if identical:
            identical_count += 1
        else:
            different_count += 1

        checked += 1

        if checked >= 20:
            break

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("Pairs checked:", checked)
print("Identical image pairs:", identical_count)
print("Different image pairs:", different_count)

print("=" * 70)