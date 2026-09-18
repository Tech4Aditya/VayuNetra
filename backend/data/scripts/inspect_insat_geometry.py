import h5py
import numpy as np
import os

path = (
    r"backend\data\raw\insat\3DIMG_L1C_ASIA_MER\2019\03MAY"
    r"\3DIMG_03MAY2019_1600_L1C_ASIA_MER_V01R00.h5"
)

print("FILE:")
print(path)

with h5py.File(path, "r") as f:

    print("\nROOT KEYS:")
    for key in f.keys():
        print(" ", key)

    print("\n--- TIME ---")
    print("time:", f["time"][:])

    print("\n--- COORDINATES ---")
    x = f["X"][:]
    y = f["Y"][:]

    print("X shape:", x.shape)
    print("X range:", x.min(), "->", x.max())

    print("Y shape:", y.shape)
    print("Y range:", y.min(), "->", y.max())

    print("\n--- TIR1 ---")
    tir = f["IMG_TIR1"]

    print("shape:", tir.shape)
    print("dtype:", tir.dtype)
    print("min:", np.min(tir[:]))
    print("max:", np.max(tir[:]))

    print("\nTIR1 ATTRIBUTES:")
    for k, v in tir.attrs.items():
        print(f"  {k}: {v}")

    print("\n--- TIR1 TEMPERATURE ---")

    if "IMG_TIR1_TEMP" in f:
        tir_temp = f["IMG_TIR1_TEMP"]

        print("shape:", tir_temp.shape)
        print("dtype:", tir_temp.dtype)
        print("min:", np.min(tir_temp[:]))
        print("max:", np.max(tir_temp[:]))

        print("\nTIR1_TEMP ATTRIBUTES:")
        for k, v in tir_temp.attrs.items():
            print(f"  {k}: {v}")

    print("\n--- WV ---")

    if "IMG_WV" in f:
        wv = f["IMG_WV"]

        print("shape:", wv.shape)
        print("dtype:", wv.dtype)
        print("min:", np.min(wv[:]))
        print("max:", np.max(wv[:]))

        print("\nWV ATTRIBUTES:")
        for k, v in wv.attrs.items():
            print(f"  {k}: {v}")

    print("\n--- PROJECTION ---")

    if "Projection_Information" in f:
        p = f["Projection_Information"]

        print("Projection data:", p[:])

        print("\nProjection attributes:")
        for k, v in p.attrs.items():
            print(f"  {k}: {v}")

    print("\n--- ROOT ATTRIBUTES ---")
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")