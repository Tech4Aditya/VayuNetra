from pathlib import Path
import pandas as pd
import h5py
import numpy as np
from pyproj import CRS, Transformer

BASE = Path(__file__).resolve().parents[2]

BEST = BASE / "data/labels/bulbul_besttrack.csv"
INSAT_ROOT = BASE / "data/raw/insat/3DIMG_L1C_ASIA_MER"
OUT = BASE / "data/labels/bulbul_insat_manifest.csv"

MAX_DIFF_HOURS = 1.0

crs = CRS.from_proj4(
    "+proj=merc +lon_0=75 +lat_ts=17.75 +datum=WGS84"
)
transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def valid_h5(path):
    try:
        with h5py.File(path, "r") as f:
            return "IMG_TIR1" in f and "IMG_WV" in f and "X" in f and "Y" in f
    except Exception:
        return False


def get_time(path):
    name = path.name
    try:
        # 3DIMG_06NOV2019_2330_L1C...
        date_part = name.split("_")[1]
        time_part = name.split("_")[2]

        return pd.to_datetime(
            date_part + time_part,
            format="%d%b%Y%H%M",
            utc=True
        )
    except Exception:
        return pd.NaT


def pixel_from_latlon(path, lat, lon):
    with h5py.File(path, "r") as f:
        x = np.asarray(f["X"][:])
        y = np.asarray(f["Y"][:])

    px, py = transformer.transform(lon, lat)

    col = int(np.argmin(np.abs(x - px)))
    row = int(np.argmin(np.abs(y - py)))

    return row, col


print("Loading BULBUL best track...")
bt = pd.read_csv(BEST)
bt["timestamp"] = pd.to_datetime(bt["timestamp"], utc=True)

print(f"Best-track rows: {len(bt)}")

print("Scanning INSAT files...")

files = sorted(INSAT_ROOT.rglob("*.h5"))

valid = []
for p in files:
    t = get_time(p)

    if pd.isna(t):
        continue

    if valid_h5(p):
        valid.append((p, t))

print(f"INSAT files found : {len(files)}")
print(f"Valid INSAT files : {len(valid)}")

used = set()
rows = []

for _, label in bt.iterrows():

    target = label["timestamp"]

    candidates = sorted(
        valid,
        key=lambda x: abs((x[1] - target).total_seconds())
    )

    selected = None

    for path, t in candidates:

        if path in used:
            continue

        diff = abs((t - target).total_seconds()) / 3600

        if diff <= MAX_DIFF_HOURS:
            selected = (path, t, diff)
            break

    if selected is None:
        continue

    path, insat_time, diff = selected
    used.add(path)

    try:
        row, col = pixel_from_latlon(
            path,
            float(label["latitude"]),
            float(label["longitude"])
        )
    except Exception as e:
        print("Pixel error:", path.name, e)
        continue

    rows.append({
        "cyclone": "BULBUL",
        "label_timestamp": target.isoformat(),
        "insat_timestamp": insat_time.isoformat(),
        "time_diff_hours": diff,
        "latitude": label["latitude"],
        "longitude": label["longitude"],
        "wind_kt": label["wind_kt"],
        "pressure_hpa": label["pressure_hpa"],
        "category": label["category"],
        "pixel_row": row,
        "pixel_col": col,
        "file": str(path)
    })

out = pd.DataFrame(rows)

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print()
print("========== BULBUL MANIFEST ==========")
print("Best-track rows :", len(bt))
print("Matched         :", len(out))
print("Unique INSAT     :", out["file"].nunique() if len(out) else 0)

if len(out):
    print("Mean diff (h)   :", out["time_diff_hours"].mean())
    print("Max diff (h)    :", out["time_diff_hours"].max())

print("Saved:", OUT)