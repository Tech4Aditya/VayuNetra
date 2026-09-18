from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FANI_DIR = (
    ROOT
    / "data"
    / "processed"
    / "insat_fani_model"
)

TAUKTAE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "insat_tauktae_calibrated"
)

TAUKTAE_MANIFEST = (
    ROOT
    / "data"
    / "labels"
    / "tauktae_insat_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "insat_combined"
)

OUTPUT_MANIFEST = (
    ROOT
    / "data"
    / "labels"
    / "insat_combined_manifest.csv"
)

STATS_FILE = (
    OUTPUT_DIR
    / "normalization_stats.npz"
)


# ============================================================
# OLD FANI NORMALIZATION
#
# These are the statistics used when the existing FANI
# model-input .npz files were created.
# ============================================================

FANI_MEAN = np.array(
    [1.055, 0.740],
    dtype=np.float32
)

FANI_STD = np.array(
    [0.075, 0.020],
    dtype=np.float32
)


# ============================================================
# LOAD FANI
# ============================================================

def load_fani():

    files = sorted(
        FANI_DIR.glob("*.npz")
    )

    print()
    print("-" * 60)
    print("Loading FANI")
    print("-" * 60)

    if not files:
        raise FileNotFoundError(
            f"No FANI NPZ files found in:\n{FANI_DIR}"
        )

    samples = []

    for path in files:

        data = np.load(
            path,
            allow_pickle=True
        )

        image = data["image"].astype(
            np.float32
        )

        if image.shape != (2, 128, 128):
            raise ValueError(
                f"Bad FANI shape in {path.name}: "
                f"{image.shape}"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"Non-finite FANI values: "
                f"{path.name}"
            )

        # ----------------------------------------------------
        # Reverse old normalization
        #
        # normalized = (radiance - mean) / std
        #
        # radiance = normalized * std + mean
        # ----------------------------------------------------

        radiance = (
            image
            * FANI_STD[:, None, None]
            + FANI_MEAN[:, None, None]
        )

        samples.append({
            "storm": "FANI",
            "filename": path.name,
            "image": radiance,
            "latitude": float(data["latitude"]),
            "longitude": float(data["longitude"]),
            "wind_kt": float(data["wind_kt"]),
            "pressure_hpa": float(
                data["pressure_hpa"]
            ),
            "category": str(
                data["category"]
            ),
            "insat_timestamp": str(
                data["insat_timestamp"]
            ),
            "label_timestamp": str(
                data["label_timestamp"]
            ),
            "time_diff_hours": float(
                data["time_diff_hours"]
            ),
            "pixel_row": int(
                data["pixel_row"]
            ),
            "pixel_col": int(
                data["pixel_col"]
            )
        })

    print(
        f"Loaded {len(samples)} FANI samples"
    )

    return samples


# ============================================================
# LOAD TAUKTAE
# ============================================================

def load_tauktae():

    if not TAUKTAE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{TAUKTAE_MANIFEST}"
        )

    df = pd.read_csv(
        TAUKTAE_MANIFEST
    )

    print()
    print("-" * 60)
    print("Loading TAUKTAE")
    print("-" * 60)

    samples = []

    for _, row in df.iterrows():

        filename = str(
            row["filename"]
        )

        sample_name = Path(
            filename
        ).stem

        sample_dir = (
            TAUKTAE_DIR
            / sample_name
        )

        tir1_path = (
            sample_dir
            / "tir1_radiance.npy"
        )

        wv_path = (
            sample_dir
            / "wv_radiance.npy"
        )

        if not tir1_path.exists():
            print(
                f"SKIP missing TIR1: "
                f"{filename}"
            )
            continue

        if not wv_path.exists():
            print(
                f"SKIP missing WV: "
                f"{filename}"
            )
            continue

        tir1 = np.load(
            tir1_path
        ).astype(np.float32)

        wv = np.load(
            wv_path
        ).astype(np.float32)

        if tir1.shape != (128, 128):
            raise ValueError(
                f"Bad TIR1 shape: "
                f"{tir1.shape}"
            )

        if wv.shape != (128, 128):
            raise ValueError(
                f"Bad WV shape: "
                f"{wv.shape}"
            )

        if not np.isfinite(tir1).all():
            raise ValueError(
                f"Invalid TIR1 values: "
                f"{filename}"
            )

        if not np.isfinite(wv).all():
            raise ValueError(
                f"Invalid WV values: "
                f"{filename}"
            )

        image = np.stack(
            [tir1, wv],
            axis=0
        ).astype(np.float32)

        samples.append({
            "storm": "TAUKTAE",
            "filename": filename,
            "image": image,
            "latitude": float(
                row["lat"]
            ),
            "longitude": float(
                row["lon"]
            ),
            "wind_kt": float(
                row["wind_kt"]
            ),
            "pressure_hpa": (
                float(row["pressure_hpa"])
                if pd.notna(
                    row["pressure_hpa"]
                )
                else np.nan
            ),
            "category": (
                str(row["category"])
                if "category" in row
                else ""
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
            "pixel_row": int(
                row["pixel_row"]
            ),
            "pixel_col": int(
                row["pixel_col"]
            )
        })

    print(
        f"Loaded {len(samples)} TAUKTAE samples"
    )

    return samples


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FANI + TAUKTAE → COMBINED INSAT DATASET")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    fani = load_fani()

    tauktae = load_tauktae()

    samples = fani + tauktae

    if not samples:
        raise RuntimeError(
            "No samples loaded."
        )

    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"FANI       : {len(fani)}"
    )

    print(
        f"TAUKTAE    : {len(tauktae)}"
    )

    print(
        f"TOTAL      : {len(samples)}"
    )

    # ========================================================
    # STACK RADIANCE
    # ========================================================

    radiance_images = np.stack(
        [
            sample["image"]
            for sample in samples
        ],
        axis=0
    ).astype(np.float32)

    print()
    print(
        f"Radiance tensor: "
        f"{radiance_images.shape}"
    )

    # ========================================================
    # RADIANCE STATISTICS
    # ========================================================

    channel_mean = radiance_images.mean(
        axis=(0, 2, 3)
    )

    channel_std = radiance_images.std(
        axis=(0, 2, 3)
    )

    channel_std = np.maximum(
        channel_std,
        1e-6
    )

    print()
    print("=" * 60)
    print("COMBINED RADIANCE NORMALIZATION")
    print("=" * 60)

    print(
        f"TIR1 mean : "
        f"{channel_mean[0]:.8f}"
    )

    print(
        f"TIR1 std  : "
        f"{channel_std[0]:.8f}"
    )

    print(
        f"WV mean   : "
        f"{channel_mean[1]:.8f}"
    )

    print(
        f"WV std    : "
        f"{channel_std[1]:.8f}"
    )

    # --------------------------------------------------------
    # Save stats
    # --------------------------------------------------------

    np.savez(
        STATS_FILE,
        mean=channel_mean.astype(
            np.float32
        ),
        std=channel_std.astype(
            np.float32
        )
    )

    # ========================================================
    # NORMALIZE
    # ========================================================

    normalized = (
        radiance_images
        - channel_mean[
            None, :, None, None
        ]
    ) / channel_std[
        None, :, None, None
    ]

    normalized = normalized.astype(
        np.float32
    )

    print()
    print(
        f"Normalized tensor: "
        f"{normalized.shape}"
    )

    print(
        f"Normalized min: "
        f"{normalized.min():.4f}"
    )

    print(
        f"Normalized max: "
        f"{normalized.max():.4f}"
    )

    # ========================================================
    # SAVE MODEL INPUTS
    # ========================================================

    manifest_rows = []

    for i, sample in enumerate(samples):

        sample_id = (
            f"{i:04d}_"
            f"{sample['storm'].lower()}_"
            f"{Path(sample['filename']).stem}"
        )

        output_file = (
            OUTPUT_DIR
            / f"{sample_id}.npz"
        )

        np.savez_compressed(
            output_file,
            image=normalized[i],
            storm=sample["storm"],
            filename=sample["filename"],
            latitude=sample["latitude"],
            longitude=sample["longitude"],
            wind_kt=sample["wind_kt"],
            pressure_hpa=sample["pressure_hpa"],
            category=sample["category"],
            insat_timestamp=sample[
                "insat_timestamp"
            ],
            label_timestamp=sample[
                "label_timestamp"
            ],
            time_diff_hours=sample[
                "time_diff_hours"
            ],
            pixel_row=sample[
                "pixel_row"
            ],
            pixel_col=sample[
                "pixel_col"
            ]
        )

        manifest_rows.append({
            "sample_id":
                sample_id,

            "file":
                output_file.name,

            "storm":
                sample["storm"],

            "filename":
                sample["filename"],

            "latitude":
                sample["latitude"],

            "longitude":
                sample["longitude"],

            "wind_kt":
                sample["wind_kt"],

            "pressure_hpa":
                sample["pressure_hpa"],

            "category":
                sample["category"],

            "insat_timestamp":
                sample["insat_timestamp"],

            "label_timestamp":
                sample["label_timestamp"],

            "time_diff_hours":
                sample["time_diff_hours"],

            "pixel_row":
                sample["pixel_row"],

            "pixel_col":
                sample["pixel_col"]
        })

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        OUTPUT_MANIFEST,
        index=False
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL VALIDATION")
    print("=" * 60)

    print(
        f"Images saved : {len(manifest)}"
    )

    print(
        f"Tensor shape : "
        f"{normalized.shape[1:]}"
    )

    print(
        f"Dtype        : "
        f"{normalized.dtype}"
    )

    print(
        f"Finite       : "
        f"{np.isfinite(normalized).all()}"
    )

    print()
    print("Channel contract:")
    print("  C0 = TIR1 radiance")
    print("  C1 = WV radiance")

    print()
    print(
        f"Stats: {STATS_FILE}"
    )

    print(
        f"Manifest: {OUTPUT_MANIFEST}"
    )

    print()
    print("=" * 60)
    print("COMBINED DATASET READY")
    print("=" * 60)


if __name__ == "__main__":
    main()