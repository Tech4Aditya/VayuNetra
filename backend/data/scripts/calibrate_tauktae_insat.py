from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT
    / "data"
    / "labels"
    / "tauktae_insat_manifest.csv"
)

INPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "insat_tauktae"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "insat_tauktae_calibrated"
)


# ============================================================
# INSAT-3D LAB CALIBRATION COEFFICIENTS
#
# Radiance = a*C^2 + b*C + c
#
# C = raw digital count
# ============================================================

TIR1_COEFFS = {
    "a": -3.8403e-7,
    "b":  1.60569e-3,
    "c": -2.09949e-2,
}

WV_COEFFS = {
    "a": -6.24616e-8,
    "b":  8.4133e-4,
    "c": -3.6705e-3,
}


# ============================================================
# CALIBRATION FUNCTION
# ============================================================

def calibrate_counts(counts, coeffs):
    """
    Convert INSAT-3D raw digital counts to LAB radiance.

    R = a*C^2 + b*C + c
    """

    counts = counts.astype(np.float32)

    a = coeffs["a"]
    b = coeffs["b"]
    c = coeffs["c"]

    radiance = (
        a * counts * counts
        + b * counts
        + c
    )

    return radiance.astype(np.float32)


# ============================================================
# FIND HDF5
# ============================================================

def find_h5(filename, root_dir):
    matches = list(root_dir.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find HDF5 file: {filename}"
        )

    return matches[0]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("TAUKTAE → INSAT RADIANCE CALIBRATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST}"
        )

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found:\n{INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    df = pd.read_csv(MANIFEST)

    print(f"Manifest samples : {len(df)}")
    print(f"Input directory  : {INPUT_DIR}")
    print(f"Output directory : {OUTPUT_DIR}")
    print()

    processed = 0
    skipped = 0

    # --------------------------------------------------------
    # Process every matched sample
    # --------------------------------------------------------

    for idx, row in df.iterrows():

        filename = str(row["filename"])

        print(
            f"[{idx + 1}/{len(df)}] "
            f"{filename} | "
            f"wind={float(row['wind_kt']):.1f} kt"
        )

        try:

            # ------------------------------------------------
            # Locate original HDF5
            # ------------------------------------------------

            h5_path = find_h5(
                filename,
                ROOT / "data" / "raw" / "insat_tauktae"
            )

            # ------------------------------------------------
            # Locate extracted crop
            # ------------------------------------------------

            sample_name = Path(filename).stem

            sample_dir = (
                INPUT_DIR
                / sample_name
            )

            tir1_path = sample_dir / "tir1.npy"
            wv_path = sample_dir / "wv.npy"

            if not tir1_path.exists():
                raise FileNotFoundError(
                    f"Missing TIR1 crop: {tir1_path}"
                )

            if not wv_path.exists():
                raise FileNotFoundError(
                    f"Missing WV crop: {wv_path}"
                )

            # ------------------------------------------------
            # Load raw crops
            # ------------------------------------------------

            tir1_counts = np.load(
                tir1_path
            )

            wv_counts = np.load(
                wv_path
            )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            if tir1_counts.shape != (128, 128):
                raise ValueError(
                    f"Invalid TIR1 shape: "
                    f"{tir1_counts.shape}"
                )

            if wv_counts.shape != (128, 128):
                raise ValueError(
                    f"Invalid WV shape: "
                    f"{wv_counts.shape}"
                )

            if not np.isfinite(tir1_counts).all():
                raise ValueError(
                    "TIR1 contains NaN/Inf"
                )

            if not np.isfinite(wv_counts).all():
                raise ValueError(
                    "WV contains NaN/Inf"
                )

            # ------------------------------------------------
            # Mask fill values
            #
            # INSAT-3D fill value = 1023
            # ------------------------------------------------

            tir1_valid = tir1_counts < 1023
            wv_valid = wv_counts < 1023

            if not tir1_valid.all():
                print(
                    "  WARNING: "
                    f"TIR1 fill pixels = "
                    f"{np.sum(~tir1_valid)}"
                )

            if not wv_valid.all():
                print(
                    "  WARNING: "
                    f"WV fill pixels = "
                    f"{np.sum(~wv_valid)}"
                )

            # ------------------------------------------------
            # Calibrate
            # ------------------------------------------------

            tir1_radiance = calibrate_counts(
                tir1_counts,
                TIR1_COEFFS
            )

            wv_radiance = calibrate_counts(
                wv_counts,
                WV_COEFFS
            )

            # ------------------------------------------------
            # Set fill pixels to NaN
            # ------------------------------------------------

            tir1_radiance = tir1_radiance.copy()
            wv_radiance = wv_radiance.copy()

            tir1_radiance[~tir1_valid] = np.nan
            wv_radiance[~wv_valid] = np.nan

            # ------------------------------------------------
            # Validate calibrated values
            # ------------------------------------------------

            if not np.isfinite(
                tir1_radiance[tir1_valid]
            ).all():
                raise ValueError(
                    "Invalid TIR1 radiance values"
                )

            if not np.isfinite(
                wv_radiance[wv_valid]
            ).all():
                raise ValueError(
                    "Invalid WV radiance values"
                )

            # ------------------------------------------------
            # Create output directory
            # ------------------------------------------------

            output_sample_dir = (
                OUTPUT_DIR
                / sample_name
            )

            output_sample_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Save calibrated radiance
            # ------------------------------------------------

            np.save(
                output_sample_dir
                / "tir1_radiance.npy",
                tir1_radiance.astype(np.float32)
            )

            np.save(
                output_sample_dir
                / "wv_radiance.npy",
                wv_radiance.astype(np.float32)
            )

            # ------------------------------------------------
            # Preserve raw arrays too
            # ------------------------------------------------

            np.save(
                output_sample_dir
                / "tir1_counts.npy",
                tir1_counts.astype(np.float32)
            )

            np.save(
                output_sample_dir
                / "wv_counts.npy",
                wv_counts.astype(np.float32)
            )

            # ------------------------------------------------
            # Save metadata
            # ------------------------------------------------

            metadata = {
                "filename":
                    filename,

                "label_timestamp":
                    str(row["label_timestamp"]),

                "insat_timestamp":
                    str(row["insat_timestamp"]),

                "time_diff_hours":
                    float(row["time_diff_hours"]),

                "lat":
                    float(row["lat"]),

                "lon":
                    float(row["lon"]),

                "wind_kt":
                    float(row["wind_kt"]),

                "pressure_hpa":
                    (
                        float(row["pressure_hpa"])
                        if pd.notna(row["pressure_hpa"])
                        else None
                    ),

                "pixel_row":
                    int(row["pixel_row"]),

                "pixel_col":
                    int(row["pixel_col"]),

                "tir1_calibration":
                    "R = -3.8403e-7*C^2 + "
                    "1.60569e-3*C - 2.09949e-2",

                "wv_calibration":
                    "R = -6.24616e-8*C^2 + "
                    "8.4133e-4*C - 3.6705e-3",
            }

            metadata_df = pd.DataFrame(
                [metadata]
            )

            metadata_df.to_csv(
                output_sample_dir / "metadata.csv",
                index=False
            )

            # ------------------------------------------------
            # Print statistics
            # ------------------------------------------------

            print(
                "  TIR1 radiance: "
                f"min={np.nanmin(tir1_radiance):.6f}, "
                f"max={np.nanmax(tir1_radiance):.6f}, "
                f"mean={np.nanmean(tir1_radiance):.6f}"
            )

            print(
                "  WV radiance:   "
                f"min={np.nanmin(wv_radiance):.6f}, "
                f"max={np.nanmax(wv_radiance):.6f}, "
                f"mean={np.nanmean(wv_radiance):.6f}"
            )

            processed += 1

        except Exception as e:

            skipped += 1

            print(
                f"  ERROR: {e}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)

    print(
        f"Input samples : {len(df)}"
    )

    print(
        f"Processed      : {processed}"
    )

    print(
        f"Skipped        : {skipped}"
    )

    print()
    print(
        f"Output: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()