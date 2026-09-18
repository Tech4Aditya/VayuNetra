import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from backend.models.tcir_temporal import TCIRTemporalModel


# ============================================================
# PATHS
# ============================================================

H5_PATH = Path(
    "backend/data/raw/tcir/Cyclone_Images.h5"
)

DEFAULT_MANIFEST = Path(
    "backend/data/labels/tcir_temporal_test.csv"
)

DEFAULT_CHECKPOINT = Path(
    "backend/checkpoints/tcir_temporal_best.pt"
)

DEFAULT_OUTPUT = Path(
    "backend/data/predictions/tcir_temporal_test_predictions.csv"
)


# ============================================================
# CHANNEL MAP
# ============================================================

CHANNEL_MAP = {
    "IR": 0,
    "WV": 1,
    "VIS": 2,
    "PMW": 3,
}


# ============================================================
# LOAD SINGLE FRAME
# ============================================================

def load_frame(h5_file, h5_index, channels):

    image = h5_file["Images"][int(h5_index)]

    channel_indices = [
        CHANNEL_MAP[channel]
        for channel in channels
    ]

    # H,W,C
    image = image[:, :, channel_indices].astype(
        np.float32
    )

    # [0,255] → [0,1]
    image /= 255.0

    # Handle invalid values
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    # H,W,C → C,H,W
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    return image


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    rmse = np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot > 0:
        r2 = 1.0 - (ss_res / ss_tot)
    else:
        r2 = 0.0

    return mae, rmse, r2


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate predictions using the trained TCIR temporal model."
    )

    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Temporal test manifest CSV",
    )

    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Trained temporal checkpoint",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output prediction CSV",
    )

    args = parser.parse_args()

    manifest_path = Path(
        args.manifest
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    output_path = Path(
        args.output
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("VAYUNETRA — TCIR TEMPORAL PREDICTION")
    print("=" * 70)

    print(f"Device     : {device}")
    print(f"Manifest   : {manifest_path}")
    print(f"Checkpoint : {checkpoint_path}")

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    channels = tuple(
        checkpoint.get(
            "channels",
            ("IR", "PMW"),
        )
    )

    sequence_length = int(
        checkpoint.get(
            "sequence_length",
            4,
        )
    )

    interval_hours = checkpoint.get(
        "interval_hours",
        3,
    )

    target_stats = checkpoint["target_stats"]

    print(f"Channels   : {channels}")
    print(f"Sequence   : {sequence_length} frames")
    print(f"Interval   : {interval_hours} hours")

    # ========================================================
    # LOAD NORMALIZATION STATISTICS
    # ========================================================

    wind_mean, wind_std = target_stats["wind"]
    pressure_mean, pressure_std = target_stats["pressure"]
    size_mean, size_std = target_stats["size"]

    print("\nTarget normalization:")
    print(
        f"Wind     : mean={wind_mean}, std={wind_std}"
    )
    print(
        f"Pressure : mean={pressure_mean}, std={pressure_std}"
    )
    print(
        f"Size     : mean={size_mean}, std={size_std}"
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = TCIRTemporalModel(
        in_channels=len(channels)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    # ========================================================
    # LOAD MANIFEST
    # ========================================================

    df = pd.read_csv(
        manifest_path
    )

    frame_columns = [
        f"frame_{i}_h5_index"
        for i in range(sequence_length)
    ]

    required_columns = (
        frame_columns
        + [
            "cyclone_id",
            "target_timestamp",
            "target_wind_kt",
            "target_pressure_hpa",
            "target_size_nmi",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns in manifest: "
            f"{missing_columns}"
        )

    # ========================================================
    # GENERATE PREDICTIONS
    # ========================================================

    predictions = []

    print(
        f"\nGenerating predictions for "
        f"{len(df)} sequences..."
    )

    with h5py.File(
        H5_PATH,
        "r",
    ) as h5_file:

        with torch.no_grad():

            for idx, row in df.iterrows():

                frames = []

                # --------------------------------------------
                # Load sequence
                # --------------------------------------------

                for column in frame_columns:

                    frame = load_frame(
                        h5_file,
                        row[column],
                        channels,
                    )

                    frames.append(
                        frame
                    )

                # T,C,H,W
                frames = np.stack(
                    frames
                )

                # T,C,H,W
                # →
                # 1,T,C,H,W

                frames = torch.from_numpy(
                    frames
                ).unsqueeze(
                    0
                ).to(device)

                # --------------------------------------------
                # Model inference
                # --------------------------------------------

                output = model(
                    frames
                )

                # --------------------------------------------
                # NORMALIZED OUTPUTS
                # --------------------------------------------

                pred_wind_norm = (
                    output["wind"]
                    .cpu()
                    .item()
                )

                pred_pressure_norm = (
                    output["pressure"]
                    .cpu()
                    .item()
                )

                pred_size_norm = (
                    output["size"]
                    .cpu()
                    .item()
                )

                # --------------------------------------------
                # DENORMALIZE
                # --------------------------------------------

                pred_wind = (
                    pred_wind_norm
                    * wind_std
                    + wind_mean
                )

                pred_pressure = (
                    pred_pressure_norm
                    * pressure_std
                    + pressure_mean
                )

                pred_size = (
                    pred_size_norm
                    * size_std
                    + size_mean
                )

                # --------------------------------------------
                # SAVE RESULT
                # --------------------------------------------

                predictions.append(
                    {
                        "cyclone_id":
                            row["cyclone_id"],

                        "timestamp":
                            row["target_timestamp"],

                        "true_wind_kt":
                            row["target_wind_kt"],

                        "pred_wind_kt":
                            pred_wind,

                        "true_pressure_hpa":
                            row["target_pressure_hpa"],

                        "pred_pressure_hpa":
                            pred_pressure,

                        "true_size_nmi":
                            row["target_size_nmi"],

                        "pred_size_nmi":
                            pred_size,
                    }
                )

                # --------------------------------------------
                # Progress
                # --------------------------------------------

                if (idx + 1) % 100 == 0:

                    print(
                        f"Processed "
                        f"{idx + 1}/{len(df)}"
                    )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    result = pd.DataFrame(
        predictions
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print("\nPredictions saved:")
    print(output_path)

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    print("\n" + "-" * 70)
    print("TEST METRICS")
    print("-" * 70)

    # --------------------------------------------------------
    # Wind
    # --------------------------------------------------------

    wind_true = result[
        "true_wind_kt"
    ].values

    wind_pred = result[
        "pred_wind_kt"
    ].values

    wind_mae, wind_rmse, wind_r2 = (
        calculate_metrics(
            wind_true,
            wind_pred,
        )
    )

    # --------------------------------------------------------
    # Pressure
    # --------------------------------------------------------

    pressure_true = result[
        "true_pressure_hpa"
    ].values

    pressure_pred = result[
        "pred_pressure_hpa"
    ].values

    pressure_mae, pressure_rmse, pressure_r2 = (
        calculate_metrics(
            pressure_true,
            pressure_pred,
        )
    )

    # --------------------------------------------------------
    # Size
    # --------------------------------------------------------

    size_true = result[
        "true_size_nmi"
    ].values

    size_pred = result[
        "pred_size_nmi"
    ].values

    size_mae, size_rmse, size_r2 = (
        calculate_metrics(
            size_true,
            size_pred,
        )
    )

    print(
        f"Wind       MAE={wind_mae:.2f} | "
        f"RMSE={wind_rmse:.2f} | "
        f"R²={wind_r2:.4f}"
    )

    print(
        f"Pressure   MAE={pressure_mae:.2f} | "
        f"RMSE={pressure_rmse:.2f} | "
        f"R²={pressure_r2:.4f}"
    )

    print(
        f"Size       MAE={size_mae:.2f} | "
        f"RMSE={size_rmse:.2f} | "
        f"R²={size_r2:.4f}"
    )

    # ========================================================
    # PER-CYCLONE WIND METRICS
    # ========================================================

    print("\n" + "-" * 70)
    print("PER-CYCLONE WIND PERFORMANCE")
    print("-" * 70)

    storm_rows = []

    for cyclone_id, group in result.groupby(
        "cyclone_id"
    ):

        true = group[
            "true_wind_kt"
        ].values

        pred = group[
            "pred_wind_kt"
        ].values

        mae, rmse, r2 = calculate_metrics(
            true,
            pred,
        )

        storm_rows.append(
            {
                "cyclone_id":
                    cyclone_id,

                "samples":
                    len(group),

                "wind_mae_kt":
                    mae,

                "wind_rmse_kt":
                    rmse,

                "wind_r2":
                    r2,
            }
        )

    storm_metrics = pd.DataFrame(
        storm_rows
    )

    if not storm_metrics.empty:

        print(
            storm_metrics.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.2f}",
            )
        )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    print("\n" + "-" * 70)
    print("PREDICTION SANITY CHECK")
    print("-" * 70)

    print(
        f"Predicted wind range     : "
        f"{result['pred_wind_kt'].min():.2f} – "
        f"{result['pred_wind_kt'].max():.2f} kt"
    )

    print(
        f"Predicted pressure range : "
        f"{result['pred_pressure_hpa'].min():.2f} – "
        f"{result['pred_pressure_hpa'].max():.2f} hPa"
    )

    print(
        f"Predicted size range     : "
        f"{result['pred_size_nmi'].min():.2f} – "
        f"{result['pred_size_nmi'].max():.2f} nmi"
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()