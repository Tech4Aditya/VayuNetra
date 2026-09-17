from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "labels" / "cyclone_labels_v2.csv"


def main():

    print("=" * 70)
    print("CYCLONE LABEL DATASET VALIDATION")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    missing = df.isna().sum()

    for col, count in missing.items():

        percentage = (
            count / len(df) * 100
        )

        print(
            f"{col:20s} "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------
    # Storm distribution
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("STORM DISTRIBUTION")
    print("-" * 70)

    print(
        df["storm_name"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # Category distribution
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("CATEGORY DISTRIBUTION")
    print("-" * 70)

    print(
        df["category"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # Wind statistics
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("WIND STATISTICS")
    print("-" * 70)

    print(
        df["wind_kt"]
        .describe()
        .to_string()
    )

    # ---------------------------------------------------------
    # Coordinate sanity
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("COORDINATE CHECK")
    print("-" * 70)

    invalid_lat = (
        (df["lat"] < -90) |
        (df["lat"] > 90)
    ).sum()

    invalid_lon = (
        (df["lon"] < -180) |
        (df["lon"] > 180)
    ).sum()

    print(f"Invalid latitude : {invalid_lat}")
    print(f"Invalid longitude: {invalid_lon}")

    # ---------------------------------------------------------
    # Duplicate check
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("DUPLICATE CHECK")
    print("-" * 70)

    duplicates = df.duplicated(
        subset=[
            "storm_id",
            "timestamp"
        ]
    ).sum()

    print(
        f"Duplicate storm/timestamp rows: {duplicates}"
    )

    # ---------------------------------------------------------
    # Temporal spacing
    # ---------------------------------------------------------

    print("\n" + "-" * 70)
    print("TEMPORAL SPACING")
    print("-" * 70)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    gaps = []

    for storm_id, group in df.groupby("storm_id"):

        group = group.sort_values("timestamp")

        delta = (
            group["timestamp"]
            .diff()
            .dropna()
            .dt.total_seconds()
            / 3600
        )

        gaps.extend(delta.tolist())

    if gaps:

        spacing = pd.Series(gaps)

        print(
            spacing
            .describe()
            .to_string()
        )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()