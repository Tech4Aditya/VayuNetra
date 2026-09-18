import pandas as pd
import numpy as np
from pathlib import Path


LABEL_PATH = Path("backend/data/labels/cyclone_labels_v2.csv")

TARGET_STORMS = [
    "FANI",
    "AMPHAN",
    "NIVAR",
    "TAUKTAE",
    "YAAS",
    "BIPARJOY",
    "HAMOON",
    "MICHAUNG",
]


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate():
    print("=" * 70)
    print("VAYUNETRA — INDIAN CYCLONE EVALUATION")
    print("=" * 70)

    df = pd.read_csv(LABEL_PATH)

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    required = [
        "storm_id",
        "storm_name",
        "season",
        "timestamp",
        "lat",
        "lon",
        "wind_kt",
        "pressure_hpa",
        "basin",
        "category",
        "has_intensity",
        "has_category",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df[df["storm_name"].isin(TARGET_STORMS)].copy()

    print(f"\nTotal observations : {len(df)}")
    print(f"Storms             : {df['storm_name'].nunique()}")

    # ---------------------------------------------------------
    # Storm summary
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("STORM DATASET SUMMARY")
    print("-" * 70)

    summary = (
        df.groupby(["storm_name", "season"])
        .agg(
            observations=("timestamp", "count"),
            intensity_available=("has_intensity", "sum"),
            category_available=("has_category", "sum"),
        )
        .reset_index()
    )

    print(summary.to_string(index=False))

    # ---------------------------------------------------------
    # Data quality
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("DATA QUALITY")
    print("-" * 70)

    print(f"Missing wind      : {df['wind_kt'].isna().sum()}")
    print(f"Missing pressure  : {df['pressure_hpa'].isna().sum()}")
    print(f"Missing latitude  : {df['lat'].isna().sum()}")
    print(f"Missing longitude : {df['lon'].isna().sum()}")

    # ---------------------------------------------------------
    # Category distribution
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("CATEGORY DISTRIBUTION")
    print("-" * 70)

    category_counts = df["category"].value_counts(dropna=False)

    print(category_counts.to_string())

    # ---------------------------------------------------------
    # Temporal consistency
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("TEMPORAL CONSISTENCY")
    print("-" * 70)

    gap_counts = []

    for storm_id, group in df.groupby("storm_id"):

        group = group.sort_values("timestamp")

        gaps = (
            group["timestamp"]
            .diff()
            .dropna()
            .dt.total_seconds()
            / 3600
        )

        gap_counts.extend(gaps.tolist())

    gap_counts = pd.Series(gap_counts)

    print("Gap distribution:")
    print(gap_counts.value_counts().sort_index().to_string())

    print(f"\nMean gap : {gap_counts.mean():.2f} hours")

    # ---------------------------------------------------------
    # Intensity statistics
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("INTENSITY STATISTICS")
    print("-" * 70)

    valid = df[df["has_intensity"] == 1].copy()

    print(f"Valid intensity observations : {len(valid)}")

    if len(valid) > 0:

        print(
            f"Wind range : "
            f"{valid['wind_kt'].min():.1f} – "
            f"{valid['wind_kt'].max():.1f} kt"
        )

        print(
            f"Wind mean  : "
            f"{valid['wind_kt'].mean():.2f} kt"
        )

        print(
            f"Pressure range : "
            f"{valid['pressure_hpa'].min():.1f} – "
            f"{valid['pressure_hpa'].max():.1f} hPa"
        )

        print(
            f"Pressure mean  : "
            f"{valid['pressure_hpa'].mean():.2f} hPa"
        )

    # ---------------------------------------------------------
    # Per-storm intensity summary
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("PER-STORM INTENSITY SUMMARY")
    print("-" * 70)

    storm_rows = []

    for storm_name, group in df.groupby("storm_name"):

        valid = group[group["has_intensity"] == 1]

        if len(valid) == 0:
            continue

        storm_rows.append(
            {
                "Storm": storm_name,
                "Season": int(group["season"].iloc[0]),
                "Samples": len(valid),
                "Mean Wind": valid["wind_kt"].mean(),
                "Max Wind": valid["wind_kt"].max(),
                "Min Pressure": valid["pressure_hpa"].min(),
            }
        )

    storm_summary = pd.DataFrame(storm_rows)

    if not storm_summary.empty:
        print(
            storm_summary.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )

    # ---------------------------------------------------------
    # Save cleaned evaluation reference
    # ---------------------------------------------------------

    output_path = Path(
        "backend/data/labels/indian_cyclone_evaluation.csv"
    )

    df.sort_values(
        ["storm_id", "timestamp"]
    ).to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("EVALUATION REFERENCE CREATED")
    print("=" * 70)

    print(output_path)

    print("\nNext stage:")
    print("TCIR predictions → prediction CSV → this reference → metrics")


if __name__ == "__main__":
    evaluate()