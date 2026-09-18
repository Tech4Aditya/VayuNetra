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

    df = df.sort_values(["cyclone_id", "timestamp_dt"])

    counts = []

    for cyclone_id, group in df.groupby("cyclone_id"):

        times = group["timestamp_dt"].tolist()

        current = 1

        for i in range(len(times) - 1):

            gap = (
                times[i + 1] - times[i]
            ).total_seconds() / 3600

            if gap == 3:
                current += 1
            else:
                counts.append(current)
                current = 1

        counts.append(current)

    s = pd.Series(counts)

    print("\n" + "=" * 60)
    print(split.upper())
    print("=" * 60)

    print("Continuous 3-hour segments:", len(s))
    print()

    print("Segment length distribution:")
    print(s.value_counts().sort_index().to_string())

    print()

    for length in range(2, 11):
        sequences = sum(max(0, x - length + 1) for x in s)

        print(
            f"{length}-frame sequences: {sequences}"
        )