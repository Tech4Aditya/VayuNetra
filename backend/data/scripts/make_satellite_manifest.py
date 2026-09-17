from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "labels" / "cyclone_labels_v2.csv"
OUTPUT = ROOT / "data" / "labels" / "satellite_manifest.csv"


# INSAT-3D L1C SGP is half-hourly.
SATELLITE_INTERVAL_MINUTES = 30


def main():

    df = pd.read_csv(INPUT)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    rows = []

    for _, row in df.iterrows():

        timestamp = row["timestamp"]

        # Round to nearest 30-minute satellite slot.
        satellite_time = timestamp.round(
            f"{SATELLITE_INTERVAL_MINUTES}min"
        )

        rows.append({
            "storm_id": row["storm_id"],
            "storm_name": row["storm_name"],
            "season": row["season"],

            "track_timestamp": timestamp.isoformat(),

            "satellite_timestamp":
                satellite_time.isoformat(),

            "lat": row["lat"],
            "lon": row["lon"],

            "wind_kt": row["wind_kt"],
            "pressure_hpa": row["pressure_hpa"],

            "category": row["category"],
            "presence": row["presence"],

            "dataset_id": "3DIMG_L1C_SGP"
        })

    manifest = pd.DataFrame(rows)

    manifest = manifest.drop_duplicates(
        subset=[
            "storm_id",
            "satellite_timestamp"
        ]
    )

    manifest = manifest.sort_values(
        [
            "storm_name",
            "satellite_timestamp"
        ]
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest.to_csv(
        OUTPUT,
        index=False
    )

    print(f"Saved:\n{OUTPUT}")

    print(
        f"\nManifest rows: {len(manifest):,}"
    )

    print("\nRows per storm:")

    print(
        manifest.groupby(
            "storm_name"
        )
        .size()
        .sort_values(
            ascending=False
        )
        .to_string()
    )


if __name__ == "__main__":
    main()