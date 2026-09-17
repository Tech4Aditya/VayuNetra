from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "raw" / "ibtracs" / "ibtracs.NI.list.v04r01.csv"
OUTPUT = ROOT / "data" / "labels" / "cyclone_labels.csv"


# Storms we initially care about for the SIH prototype.
TARGET_STORMS = {
    "FANI",
    "AMPHAN",
    "NIVAR",
    "TAUKTAE",
    "YAAS",
    "BIPARJOY",
    "HAMOON",
    "MICHAUNG",
}


def clean(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in {"", "nan", "NaN", "None", "NOT_NAMED"}:
        return None

    return value


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"IBTrACS file not found:\n{INPUT}"
        )

    print("Reading IBTrACS...")
    print(f"File: {INPUT}")

    # IBTrACS CSV contains metadata/header rows.
    df = pd.read_csv(
        INPUT,
        skiprows=[1]
    )

    print(f"Rows loaded: {len(df):,}")

    print("\nAvailable important columns:")
    wanted = [
        "SID",
        "SEASON",
        "NAME",
        "ISO_TIME",
        "LAT",
        "LON",
        "WMO_WIND",
        "WMO_PRES",
        "BASIN",
    ]

    for column in wanted:
        if column in df.columns:
            print(f"  {column}")

    # Keep North Indian Ocean records.
    if "BASIN" in df.columns:
        df = df[
            df["BASIN"].astype(str).str.upper().eq("NI")
        ]

    # Keep our initial target storms.
    if "NAME" in df.columns:
        names = (
            df["NAME"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        df = df[names.isin(TARGET_STORMS)]

    print(f"\nTarget-storm rows: {len(df):,}")

    if len(df) == 0:
        print("No target storms found.")
        return

    # Convert numeric fields.
    for column in ["SEASON", "LAT", "LON", "WMO_WIND", "WMO_PRES"]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Parse timestamps.
    df["ISO_TIME"] = pd.to_datetime(
        df["ISO_TIME"],
        errors="coerce",
        utc=True
    )

    # Remove invalid observations.
    df = df.dropna(
        subset=["ISO_TIME", "LAT", "LON"]
    )

    # Build our clean label table.
    output = pd.DataFrame({
        "storm_id": df["SID"].astype(str),
        "storm_name": df["NAME"].astype(str).str.strip(),
        "season": df["SEASON"],
        "timestamp": df["ISO_TIME"].dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "lat": df["LAT"],
        "lon": df["LON"],
        "wind_kt": df["WMO_WIND"],
        "pressure_hpa": df["WMO_PRES"],
        "basin": df["BASIN"],
    })

    # Remove duplicate observations.
    output = output.drop_duplicates(
        subset=[
            "storm_id",
            "timestamp"
        ]
    )

    output = output.sort_values(
        [
            "storm_name",
            "timestamp"
        ]
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT,
        index=False
    )

    print("\nSaved:")
    print(OUTPUT)

    print("\nStorm summary:")
    print(
        output.groupby(
            ["season", "storm_name"]
        )
        .agg(
            observations=("timestamp", "count"),
            first_time=("timestamp", "min"),
            last_time=("timestamp", "max"),
            max_wind_kt=("wind_kt", "max"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()