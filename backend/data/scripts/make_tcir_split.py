import pandas as pd
import numpy as np

INPUT = "backend/data/labels/tcir_manifest.csv"

TRAIN_OUT = "backend/data/labels/tcir_train.csv"
VAL_OUT = "backend/data/labels/tcir_val.csv"
TEST_OUT = "backend/data/labels/tcir_test.csv"

SEED = 42

df = pd.read_csv(INPUT)

# --------------------------------------------------
# Unique cyclone IDs
# --------------------------------------------------

cyclones = (
    df["cyclone_id"]
    .drop_duplicates()
    .to_numpy()
)

rng = np.random.default_rng(SEED)
rng.shuffle(cyclones)

n = len(cyclones)

n_train = int(0.70 * n)
n_val = int(0.15 * n)

train_ids = set(
    cyclones[:n_train]
)

val_ids = set(
    cyclones[n_train:n_train + n_val]
)

test_ids = set(
    cyclones[n_train + n_val:]
)

# --------------------------------------------------
# Split by cyclone
# --------------------------------------------------

train = df[
    df["cyclone_id"].isin(train_ids)
].copy()

val = df[
    df["cyclone_id"].isin(val_ids)
].copy()

test = df[
    df["cyclone_id"].isin(test_ids)
].copy()

# --------------------------------------------------
# Save
# --------------------------------------------------

train.to_csv(TRAIN_OUT, index=False)
val.to_csv(VAL_OUT, index=False)
test.to_csv(TEST_OUT, index=False)

# --------------------------------------------------
# Verify no leakage
# --------------------------------------------------

assert set(train["cyclone_id"]).isdisjoint(
    set(val["cyclone_id"])
)

assert set(train["cyclone_id"]).isdisjoint(
    set(test["cyclone_id"])
)

assert set(val["cyclone_id"]).isdisjoint(
    set(test["cyclone_id"])
)

print("=" * 70)
print("TCIR STORM-LEVEL SPLIT")
print("=" * 70)

print(
    f"\nTrain: {len(train):,} frames "
    f"| {train['cyclone_id'].nunique()} cyclones"
)

print(
    f"Val  : {len(val):,} frames "
    f"| {val['cyclone_id'].nunique()} cyclones"
)

print(
    f"Test : {len(test):,} frames "
    f"| {test['cyclone_id'].nunique()} cyclones"
)

print("\nFrame distribution:")
print(
    pd.DataFrame({
        "train": [len(train)],
        "val": [len(val)],
        "test": [len(test)]
    }).to_string(index=False)
)

print("\nBasin distribution:")
print("\nTRAIN")
print(train["basin"].value_counts())

print("\nVAL")
print(val["basin"].value_counts())

print("\nTEST")
print(test["basin"].value_counts())

print("\nNo cyclone leakage detected.")

print("\nSaved:")
print(TRAIN_OUT)
print(VAL_OUT)
print(TEST_OUT)

print("=" * 70)