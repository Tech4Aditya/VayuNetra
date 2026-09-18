from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

MANIFEST = Path(
    "backend/data/labels/fani_insat_manifest_strict.csv"
)

INPUT_ROOT = Path(
    "backend/data/processed/insat_fani"
)

OUTPUT_ROOT = Path(
    "backend/data/processed/insat_fani_model"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHANNELS
# ============================================================

CHANNELS = [
    "tir1_radiance",
    "wv_radiance",
]


# ============================================================
# TEMPORARY NORMALIZATION
# ============================================================
#
# These values are ONLY provisional FANI-derived statistics.
#
# They must NOT be treated as the final normalization statistics
# for the eventual NIO training dataset.
#
# Final model:
#   calculate statistics from NIO TRAIN split only
#   -> freeze them
#   -> use same values for validation/test
#
# ============================================================

NORM = {
    "tir1_radiance": {
        "mean": 1.055,
        "std": 0.075,
    },

    "wv_radiance": {
        "mean": 0.740,
        "std": 0.020,
    },
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(x, mean, std):
    return (
        x.astype(np.float32) - mean
    ) / std


# ============================================================
# MAIN
# ============================================================

def main():

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST}"
        )

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Input directory not found:\n{INPUT_ROOT}"
        )

    print("=" * 70)
    print("INSAT MODEL INPUT PREPARATION")
    print("=" * 70)

    print(f"Manifest : {MANIFEST}")
    print(f"Input    : {INPUT_ROOT}")
    print(f"Output   : {OUTPUT_ROOT}")
    print()

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    df = pd.read_csv(MANIFEST)

    print(f"Manifest rows: {len(df)}")
    print("Columns:")
    print(df.columns.tolist())
    print()

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "filename",
        "insat_timestamp",
        "label_timestamp",
        "time_diff_hours",
        "lat",
        "lon",
        "wind_kt",
        "pressure_hpa",
        "category",
        "pixel_row",
        "pixel_col",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Manifest is missing columns: {missing}"
        )

    samples = []

    # --------------------------------------------------------
    # Process every matched INSAT frame
    # --------------------------------------------------------

    for index, row in df.iterrows():

        filename = str(row["filename"])

        stem = Path(filename).stem

        sample_dir = INPUT_ROOT / stem

        tir_path = (
            sample_dir /
            "tir1_radiance.npy"
        )

        wv_path = (
            sample_dir /
            "wv_radiance.npy"
        )

        print(
            f"[{index + 1}/{len(df)}] {stem}"
        )

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not tir_path.exists():
            print(
                f"  SKIP: missing {tir_path.name}"
            )
            continue

        if not wv_path.exists():
            print(
                f"  SKIP: missing {wv_path.name}"
            )
            continue

        # ----------------------------------------------------
        # Load calibrated radiance
        # ----------------------------------------------------

        tir = np.load(tir_path).astype(
            np.float32
        )

        wv = np.load(wv_path).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if tir.shape != (128, 128):
            print(
                f"  SKIP: bad TIR1 shape {tir.shape}"
            )
            continue

        if wv.shape != (128, 128):
            print(
                f"  SKIP: bad WV shape {wv.shape}"
            )
            continue

        # ----------------------------------------------------
        # Validate finite values
        # ----------------------------------------------------

        if not np.isfinite(tir).all():
            print(
                "  SKIP: TIR1 contains NaN/Inf"
            )
            continue

        if not np.isfinite(wv).all():
            print(
                "  SKIP: WV contains NaN/Inf"
            )
            continue

        # ----------------------------------------------------
        # Normalize channels
        # ----------------------------------------------------

        tir_normalized = normalize(
            tir,
            NORM["tir1_radiance"]["mean"],
            NORM["tir1_radiance"]["std"],
        )

        wv_normalized = normalize(
            wv,
            NORM["wv_radiance"]["mean"],
            NORM["wv_radiance"]["std"],
        )

        # ----------------------------------------------------
        # Stack channels
        #
        # PyTorch convention:
        #
        #       C x H x W
        #
        #       2 x 128 x 128
        #
        # Channel 0 = TIR1
        # Channel 1 = WV
        # ----------------------------------------------------

        image = np.stack(
            [
                tir_normalized,
                wv_normalized,
            ],
            axis=0,
        ).astype(np.float32)

        # ----------------------------------------------------
        # Output path
        # ----------------------------------------------------

        out_path = (
            OUTPUT_ROOT /
            f"{stem}.npz"
        )

        # ----------------------------------------------------
        # Save model-ready sample
        # ----------------------------------------------------

        np.savez_compressed(
            out_path,

            # Model input
            image=image,

            # INSAT metadata
            latitude=float(row["lat"]),
            longitude=float(row["lon"]),

            # Ground-truth labels
            wind_kt=float(row["wind_kt"]),
            pressure_hpa=float(
                row["pressure_hpa"]
            ),

            category=str(
                row["category"]
            ),

            # Time information
            insat_timestamp=str(
                row["insat_timestamp"]
            ),

            label_timestamp=str(
                row["label_timestamp"]
            ),

            time_diff_hours=float(
                row["time_diff_hours"]
            ),

            # Original spatial location
            pixel_row=int(
                row["pixel_row"]
            ),

            pixel_col=int(
                row["pixel_col"]
            ),
        )

        # ----------------------------------------------------
        # Manifest entry
        # ----------------------------------------------------

        samples.append(
            {
                "filename": filename,

                "npz_path": str(
                    out_path
                ),

                "insat_timestamp": str(
                    row["insat_timestamp"]
                ),

                "label_timestamp": str(
                    row["label_timestamp"]
                ),

                "time_diff_hours": float(
                    row["time_diff_hours"]
                ),

                "lat": float(
                    row["lat"]
                ),

                "lon": float(
                    row["lon"]
                ),

                "wind_kt": float(
                    row["wind_kt"]
                ),

                "pressure_hpa": float(
                    row["pressure_hpa"]
                ),

                "category": str(
                    row["category"]
                ),

                "pixel_row": int(
                    row["pixel_row"]
                ),

                "pixel_col": int(
                    row["pixel_col"]
                ),
            }
        )

        print(
            f"  shape = {image.shape}"
        )

        print(
            f"  TIR1 normalized: "
            f"min={tir_normalized.min():.4f} "
            f"max={tir_normalized.max():.4f} "
            f"mean={tir_normalized.mean():.4f}"
        )

        print(
            f"  WV normalized:   "
            f"min={wv_normalized.min():.4f} "
            f"max={wv_normalized.max():.4f} "
            f"mean={wv_normalized.mean():.4f}"
        )

        print(
            f"  wind = {row['wind_kt']} kt"
        )

        print()

    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    output_manifest = (
        OUTPUT_ROOT /
        "manifest.csv"
    )

    output_df = pd.DataFrame(
        samples
    )

    output_df.to_csv(
        output_manifest,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Input samples : {len(df)}"
    )

    print(
        f"Processed      : {len(samples)}"
    )

    print(
        f"Skipped        : {len(df) - len(samples)}"
    )

    print(
        f"Output         : {OUTPUT_ROOT}"
    )

    print(
        f"Manifest       : {output_manifest}"
    )

    print()

    if len(samples) > 0:

        print(
            "Tensor contract:"
        )

        print(
            "  shape  = (2, 128, 128)"
        )

        print(
            "  dtype  = float32"
        )

        print(
            "  C0     = TIR1 radiance"
        )

        print(
            "  C1     = WV radiance"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()