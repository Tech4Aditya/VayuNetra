from pathlib import Path
import pandas as pd
import numpy as np
import h5py
from PIL import Image


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "data" / "labels" / "tauktae_insat_manifest.csv"
INPUT_DIR = ROOT / "data" / "raw" / "insat_tauktae"
OUTPUT_DIR = ROOT / "data" / "processed" / "insat_tauktae"


# ============================================================
# SETTINGS
# ============================================================

CROP_SIZE = 128
HALF = CROP_SIZE // 2


# ============================================================
# HELPERS
# ============================================================

def find_h5(filename):
    matches = list(INPUT_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(f"Could not find: {filename}")

    return matches[0]


def extract_crop(array, row, col):
    """
    Extract centered 128x128 crop.
    """
    h, w = array.shape

    r0 = int(row) - HALF
    r1 = int(row) + HALF
    c0 = int(col) - HALF
    c1 = int(col) + HALF

    # Keep crop inside image boundaries
    r0 = max(0, min(r0, h - CROP_SIZE))
    c0 = max(0, min(c0, w - CROP_SIZE))

    r1 = r0 + CROP_SIZE
    c1 = c0 + CROP_SIZE

    crop = array[r0:r1, c0:c1]

    if crop.shape != (CROP_SIZE, CROP_SIZE):
        raise ValueError(f"Unexpected crop shape: {crop.shape}")

    return crop


def save_visualization(arr, path):
    """
    Save simple 8-bit visualization.
    Does NOT modify the raw data.
    """
    arr = arr.astype(np.float32)

    valid = arr[np.isfinite(arr)]

    if len(valid) == 0:
        raise ValueError("No finite values for visualization")

    lo = np.percentile(valid, 1)
    hi = np.percentile(valid, 99)

    if hi <= lo:
        hi = lo + 1.0

    scaled = np.clip((arr - lo) / (hi - lo), 0, 1)
    image = (scaled * 255).astype(np.uint8)

    Image.fromarray(image).save(path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TAUKTAE → INSAT CROP EXTRACTION")
    print("=" * 60)

    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST)

    print(f"Manifest rows: {len(df)}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print()

    processed = 0
    skipped = 0

    for idx, row in df.iterrows():

        filename = str(row["filename"])

        pixel_row = int(row["pixel_row"])
        pixel_col = int(row["pixel_col"])

        wind = float(row["wind_kt"])

        print(
            f"[{idx + 1}/{len(df)}] "
            f"{filename} | "
            f"wind={wind:.1f} kt | "
            f"pixel=({pixel_row},{pixel_col})"
        )

        try:
            h5_path = find_h5(filename)

            sample_name = Path(filename).stem
            sample_dir = OUTPUT_DIR / sample_name
            sample_dir.mkdir(parents=True, exist_ok=True)

            with h5py.File(h5_path, "r") as f:

                if "IMG_TIR1" not in f:
                    raise KeyError("IMG_TIR1 missing")

                if "IMG_WV" not in f:
                    raise KeyError("IMG_WV missing")

                tir1 = f["IMG_TIR1"][0]
                wv = f["IMG_WV"][0]

            # Extract centered crops
            tir1_crop = extract_crop(
                tir1,
                pixel_row,
                pixel_col
            )

            wv_crop = extract_crop(
                wv,
                pixel_row,
                pixel_col
            )

            # Validate
            if not np.isfinite(tir1_crop).all():
                raise ValueError("TIR1 contains NaN/Inf")

            if not np.isfinite(wv_crop).all():
                raise ValueError("WV contains NaN/Inf")

            # Check fill values
            tir1_fill = np.sum(tir1_crop >= 1023)
            wv_fill = np.sum(wv_crop >= 1023)

            if tir1_fill > 0:
                print(f"  WARNING: TIR1 fill pixels: {tir1_fill}")

            if wv_fill > 0:
                print(f"  WARNING: WV fill pixels: {wv_fill}")

            # Save raw arrays
            np.save(
                sample_dir / "tir1.npy",
                tir1_crop.astype(np.float32)
            )

            np.save(
                sample_dir / "wv.npy",
                wv_crop.astype(np.float32)
            )

            # Save visualization only
            save_visualization(
                tir1_crop,
                sample_dir / "tir1.png"
            )

            save_visualization(
                wv_crop,
                sample_dir / "wv.png"
            )

            print(
                f"  TIR1 mean={tir1_crop.mean():.3f}, "
                f"std={tir1_crop.std():.3f}"
            )

            print(
                f"  WV   mean={wv_crop.mean():.3f}, "
                f"std={wv_crop.std():.3f}"
            )

            processed += 1

        except Exception as e:

            skipped += 1

            print(f"  ERROR: {e}")

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Input samples : {len(df)}")
    print(f"Processed      : {processed}")
    print(f"Skipped        : {skipped}")
    print()
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()