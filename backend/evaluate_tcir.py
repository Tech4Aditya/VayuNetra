from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_PATH = Path(
    "backend/data/predictions/tcir_temporal_test_predictions.csv"
)

OUTPUT_DIR = Path(
    "backend/data/evaluation"
)


def calculate_metrics(true, pred):
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)

    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))

    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    return mae, rmse, r2


def evaluate_target(df, true_col, pred_col):
    data = df[[true_col, pred_col]].dropna()

    return calculate_metrics(
        data[true_col].values,
        data[pred_col].values,
    )


def main():

    print("=" * 70)
    print("VAYUNETRA — TCIR MODEL EVALUATION")
    print("=" * 70)

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTION_PATH}"
        )

    df = pd.read_csv(PREDICTION_PATH)

    required = [
        "cyclone_id",
        "timestamp",
        "true_wind_kt",
        "pred_wind_kt",
        "true_pressure_hpa",
        "pred_pressure_hpa",
        "true_size_nmi",
        "pred_size_nmi",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print("\n" + "-" * 70)
    print("TEST DATASET")
    print("-" * 70)

    print(f"Test sequences : {len(df)}")
    print(f"Cyclones       : {df['cyclone_id'].nunique()}")

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    print("\n" + "-" * 70)
    print("OVERALL PERFORMANCE")
    print("-" * 70)

    targets = {
        "Wind": (
            "true_wind_kt",
            "pred_wind_kt",
            "kt",
        ),
        "Pressure": (
            "true_pressure_hpa",
            "pred_pressure_hpa",
            "hPa",
        ),
        "Size": (
            "true_size_nmi",
            "pred_size_nmi",
            "nmi",
        ),
    }

    overall_rows = []

    for name, (
        true_col,
        pred_col,
        unit,
    ) in targets.items():

        mae, rmse, r2 = evaluate_target(
            df,
            true_col,
            pred_col,
        )

        overall_rows.append(
            {
                "target": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "unit": unit,
            }
        )

        print(
            f"{name:10s} "
            f"MAE={mae:.2f} {unit} | "
            f"RMSE={rmse:.2f} {unit} | "
            f"R²={r2:.4f}"
        )

    overall_df = pd.DataFrame(
        overall_rows
    )

    overall_df.to_csv(
        OUTPUT_DIR / "overall_metrics.csv",
        index=False,
    )

    # ========================================================
    # PER-CYCLONE METRICS
    # ========================================================

    print("\n" + "-" * 70)
    print("PER-CYCLONE PERFORMANCE")
    print("-" * 70)

    storm_rows = []

    for cyclone_id, group in df.groupby(
        "cyclone_id"
    ):

        wind_mae, wind_rmse, wind_r2 = (
            evaluate_target(
                group,
                "true_wind_kt",
                "pred_wind_kt",
            )
        )

        pressure_mae, pressure_rmse, pressure_r2 = (
            evaluate_target(
                group,
                "true_pressure_hpa",
                "pred_pressure_hpa",
            )
        )

        size_mae, size_rmse, size_r2 = (
            evaluate_target(
                group,
                "true_size_nmi",
                "pred_size_nmi",
            )
        )

        storm_rows.append(
            {
                "cyclone_id": cyclone_id,
                "samples": len(group),

                "wind_mae_kt": wind_mae,
                "wind_rmse_kt": wind_rmse,
                "wind_r2": wind_r2,

                "pressure_mae_hpa": pressure_mae,
                "pressure_rmse_hpa": pressure_rmse,
                "pressure_r2": pressure_r2,

                "size_mae_nmi": size_mae,
                "size_rmse_nmi": size_rmse,
                "size_r2": size_r2,
            }
        )

    storm_df = pd.DataFrame(
        storm_rows
    )

    print(
        storm_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    storm_df.to_csv(
        OUTPUT_DIR / "per_cyclone_metrics.csv",
        index=False,
    )

    # ========================================================
    # ERROR COLUMNS
    # ========================================================

    result = df.copy()

    result["wind_error_kt"] = (
        result["pred_wind_kt"]
        - result["true_wind_kt"]
    )

    result["wind_abs_error_kt"] = (
        result["wind_error_kt"].abs()
    )

    result["pressure_error_hpa"] = (
        result["pred_pressure_hpa"]
        - result["true_pressure_hpa"]
    )

    result["pressure_abs_error_hpa"] = (
        result["pressure_error_hpa"].abs()
    )

    result["size_error_nmi"] = (
        result["pred_size_nmi"]
        - result["true_size_nmi"]
    )

    result["size_abs_error_nmi"] = (
        result["size_error_nmi"].abs()
    )

    result.to_csv(
        OUTPUT_DIR / "detailed_predictions.csv",
        index=False,
    )

    # ========================================================
    # LARGEST WIND ERRORS
    # ========================================================

    print("\n" + "-" * 70)
    print("LARGEST WIND ERRORS")
    print("-" * 70)

    largest_errors = (
        result.sort_values(
            "wind_abs_error_kt",
            ascending=False,
        )
        .head(10)
    )

    print(
        largest_errors[
            [
                "cyclone_id",
                "timestamp",
                "true_wind_kt",
                "pred_wind_kt",
                "wind_error_kt",
                "wind_abs_error_kt",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    largest_errors[
        [
            "cyclone_id",
            "timestamp",
            "true_wind_kt",
            "pred_wind_kt",
            "wind_error_kt",
            "wind_abs_error_kt",
        ]
    ].to_csv(
        OUTPUT_DIR / "largest_wind_errors.csv",
        index=False,
    )

    # ========================================================
    # BIAS
    # ========================================================

    print("\n" + "-" * 70)
    print("MODEL BIAS")
    print("-" * 70)

    print(
        f"Wind bias     : "
        f"{result['wind_error_kt'].mean():+.2f} kt"
    )

    print(
        f"Pressure bias : "
        f"{result['pressure_error_hpa'].mean():+.2f} hPa"
    )

    print(
        f"Size bias     : "
        f"{result['size_error_nmi'].mean():+.2f} nmi"
    )

    # ========================================================
    # PREDICTION RANGE
    # ========================================================

    print("\n" + "-" * 70)
    print("PREDICTION RANGE")
    print("-" * 70)

    for name, column, unit in [
        ("Wind", "pred_wind_kt", "kt"),
        ("Pressure", "pred_pressure_hpa", "hPa"),
        ("Size", "pred_size_nmi", "nmi"),
    ]:

        print(
            f"{name:10s}: "
            f"{result[column].min():.2f} – "
            f"{result[column].max():.2f} {unit}"
        )

    # ========================================================
    # OUTPUT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    for path in [
        OUTPUT_DIR / "overall_metrics.csv",
        OUTPUT_DIR / "per_cyclone_metrics.csv",
        OUTPUT_DIR / "detailed_predictions.csv",
        OUTPUT_DIR / "largest_wind_errors.csv",
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()