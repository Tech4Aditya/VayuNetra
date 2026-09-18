import glob
import os
import h5py

roots = [
    r"backend\data\raw\insat\3DIMG_L1C_ASIA_MER\2019\03MAY",
    r"backend\data\raw\insat\3DIMG_L1C_ASIA_MER\2019\04MAY",
]

files = []
for root in roots:
    files.extend(glob.glob(os.path.join(root, "*.h5")))

print(f"Found {len(files)} files\n")

ok = 0
bad = 0

for path in sorted(files):
    try:
        with h5py.File(path, "r") as f:
            required = ["IMG_TIR1", "IMG_WV", "X", "Y", "time"]

            missing = [k for k in required if k not in f]

            if missing:
                print(f"BAD  {os.path.basename(path)} | missing: {missing}")
                bad += 1
                continue

            shape = f["IMG_TIR1"].shape

            if len(shape) != 3:
                print(f"BAD  {os.path.basename(path)} | shape={shape}")
                bad += 1
                continue

            print(f"OK   {os.path.basename(path)} | TIR1={shape}")
            ok += 1

    except Exception as e:
        print(f"BAD  {os.path.basename(path)} | {type(e).__name__}: {e}")
        bad += 1

print("\nSUMMARY")
print("OK :", ok)
print("BAD:", bad)