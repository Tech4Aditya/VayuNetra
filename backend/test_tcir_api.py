import h5py
import requests
import pandas as pd


H5_PATH = "backend/data/raw/tcir/Cyclone_Images.h5"
MANIFEST = "backend/data/labels/tcir_temporal_test.csv"

API_URL = "http://127.0.0.1:8000/predict_tcir_sequence"


# ------------------------------------------------------------
# LOAD ONE REAL TEST SEQUENCE
# ------------------------------------------------------------

df = pd.read_csv(MANIFEST)

row = df.iloc[0]

frame_columns = [
    "frame_0_h5_index",
    "frame_1_h5_index",
    "frame_2_h5_index",
    "frame_3_h5_index",
]


indices = [
    int(row[col])
    for col in frame_columns
]


print("=" * 70)
print("TCIR API TEST")
print("=" * 70)

print(f"Cyclone : {row['cyclone_id']}")
print(f"Target  : {row['target_timestamp']}")
print(f"Wind    : {row['target_wind_kt']} kt")
print(f"Pressure: {row['target_pressure_hpa']} hPa")
print(f"Size    : {row['target_size_nmi']} nmi")

print("\nHDF5 indices:")
print(indices)


# ------------------------------------------------------------
# READ ACTUAL SATELLITE FRAMES
# ------------------------------------------------------------

with h5py.File(H5_PATH, "r") as h5:

    images = h5["Images"]

    frames = [
        images[index].tolist()
        for index in indices
    ]


print("\nLoaded frames:")
print("4 × 128 × 128 × 4")


# ------------------------------------------------------------
# SEND TO API
# ------------------------------------------------------------

payload = {
    "frames": frames
}


response = requests.post(
    API_URL,
    json=payload,
    timeout=60
)


print("\nHTTP status:", response.status_code)

print("\nAPI RESPONSE")
print("=" * 70)

print(
    response.json()
)