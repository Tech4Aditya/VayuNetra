from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "labels" / "cyclone_labels.csv"
OUTPUT = ROOT / "data" / "labels" / "cyclone_labels_v2.csv"


def classify_wind(wind_kt):
    """
    North Indian Ocean cyclone classification.

    Wind speed is maximum sustained wind in knots.
    """

    if pd.isna(wind_kt):
        return "Unknown"

    wind = float(wind_kt)

    if wind < 17:
        return "Low Pressure Area"

    if wind <= 27:
        return "Depression"

    if wind <= 33:
        return "Deep Depression"

    if wind <= 47:
        return "Cyclonic Storm"

    if wind <= 63:
        return "Severe Cyclonic Storm"

    if wind <= 90:
        return "Very Severe Cyclonic Storm"

    if wind <= 119:
        return "Extremely Severe Cyclonic Storm"

    return "Super Cyclonic Storm"


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT}"
        )

    df = pd.read_csv(INPUT)

    df["category"] = df["wind_kt"].apply(
        classify_wind
    )

    df["presence"] = 1

    df.to_csv(
        OUTPUT,
        index=False
    )

    print(f"Saved:\n{OUTPUT}")

    print("\nCategory distribution:")
    print(
        df["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nStorm/category distribution:")
    print(
        pd.crosstab(
            df["storm_name"],
            df["category"]
        ).to_string()
    )


if __name__ == "__main__":
    main()