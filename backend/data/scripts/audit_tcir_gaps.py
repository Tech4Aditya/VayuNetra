import pandas as pd


FILES = {
    "train": "backend/data/labels/tcir_train.csv",
    "val": "backend/data/labels/tcir_val.csv",
    "test": "backend/data/labels/tcir_test.csv",
}


for split, path in FILES.items():

    df = pd.read_csv(path)

    df["timestamp_dt"] = pd.to_datetime(
        df["timestamp"].astype(str),
        format="%Y%m%d%H"
    )

    df = df.sort_values(
        ["cyclone_id", "timestamp_dt"]
    )

    gaps = []

    for cyclone_id, group in df.groupby("cyclone_id"):

        times = group["timestamp_dt"].tolist()

        for i in range(len(times) - 1):

            delta = (
                times[i + 1] - times[i]
            ).total_seconds() / 3600

            gaps.append(delta)

    gap_series = pd.Series(gaps)

    print("\n" + "=" * 60)
    print(split.upper())
    print("=" * 60)

    print(
        gap_series.value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\n3-hour percentage:",
        f"{(gap_series == 3).mean() * 100:.2f}%"
    )