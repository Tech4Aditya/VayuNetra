import os
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = "backend/data/labels/tcir_train.csv"
VAL_CSV = "backend/data/labels/tcir_val.csv"
TEST_CSV = "backend/data/labels/tcir_test.csv"

OUTPUT_DIR = "backend/data/labels"

SEQUENCE_LENGTH = 4

EXPECTED_INTERVAL_HOURS = 3


# ============================================================
# BUILD SEQUENCES
# ============================================================

def build_sequences(csv_path, output_path, split_name):

    print("\n" + "=" * 70)

    print(f"BUILDING {split_name.upper()} SEQUENCES")

    print("=" * 70)


    df = pd.read_csv(csv_path)


    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    df["timestamp_dt"] = pd.to_datetime(
        df["timestamp"].astype(str),
        format="%Y%m%d%H",
        errors="coerce"
    )


    if df["timestamp_dt"].isna().any():

        bad = df["timestamp_dt"].isna().sum()

        raise ValueError(
            f"{bad} invalid timestamps found "
            f"in {split_name}"
        )


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "cyclone_id",
            "timestamp_dt"
        ]
    ).reset_index(drop=True)


    sequences = []


    # --------------------------------------------------------
    # Process each cyclone independently
    # --------------------------------------------------------

    for cyclone_id, group in df.groupby(
        "cyclone_id",
        sort=False
    ):

        group = group.sort_values(
            "timestamp_dt"
        ).reset_index(drop=True)


        # ----------------------------------------------------
        # Look for consecutive 3-hour windows
        # ----------------------------------------------------

        for start in range(

            len(group) - SEQUENCE_LENGTH + 1

        ):

            window = group.iloc[
                start:
                start + SEQUENCE_LENGTH
            ]


            timestamps = (
                window["timestamp_dt"]
                .tolist()
            )


            # ------------------------------------------------
            # Verify temporal continuity
            # ------------------------------------------------

            valid = True


            for i in range(
                len(timestamps) - 1
            ):

                delta_hours = (

                    timestamps[i + 1]
                    - timestamps[i]

                ).total_seconds() / 3600


                if (
                    delta_hours
                    != EXPECTED_INTERVAL_HOURS
                ):

                    valid = False

                    break


            if not valid:

                continue


            # ------------------------------------------------
            # Build sequence record
            # ------------------------------------------------

            record = {

                "cyclone_id":
                    str(cyclone_id),

                "target_timestamp":
                    timestamps[-1].strftime(
                        "%Y%m%d%H"
                    ),

                "target_h5_index":
                    int(
                        window.iloc[-1]["h5_index"]
                    ),

                "target_wind_kt":
                    float(
                        window.iloc[-1]["wind_kt"]
                    ),

                "target_pressure_hpa":
                    float(
                        window.iloc[-1]["pressure_hpa"]
                    ),

                "target_size_nmi":
                    float(
                        window.iloc[-1]["size_nmi"]
                    ),

            }


            # ------------------------------------------------
            # Store frame indices
            # ------------------------------------------------

            for j in range(
                SEQUENCE_LENGTH
            ):

                frame = window.iloc[j]


                record[
                    f"frame_{j}_h5_index"
                ] = int(
                    frame["h5_index"]
                )


                record[
                    f"frame_{j}_timestamp"
                ] = frame[
                    "timestamp"
                ]


            sequences.append(record)


    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    sequence_df = pd.DataFrame(
        sequences
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    sequence_df.to_csv(
        output_path,
        index=False
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        f"\nInput frames: "
        f"{len(df)}"
    )


    print(
        f"Sequences: "
        f"{len(sequence_df)}"
    )


    print(
        f"Cyclones: "
        f"{sequence_df['cyclone_id'].nunique()}"
        if len(sequence_df) > 0
        else "Cyclones: 0"
    )


    print(
        f"Saved to: "
        f"{output_path}"
    )


    return sequence_df


# ============================================================
# MAIN
# ============================================================

def main():

    train_output = os.path.join(
        OUTPUT_DIR,
        "tcir_temporal_train.csv"
    )


    val_output = os.path.join(
        OUTPUT_DIR,
        "tcir_temporal_val.csv"
    )


    test_output = os.path.join(
        OUTPUT_DIR,
        "tcir_temporal_test.csv"
    )


    train_df = build_sequences(

        TRAIN_CSV,

        train_output,

        "train"

    )


    val_df = build_sequences(

        VAL_CSV,

        val_output,

        "val"

    )


    test_df = build_sequences(

        TEST_CSV,

        test_output,

        "test"

    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print("TEMPORAL DATASET SUMMARY")

    print("=" * 70)


    print(
        f"Train sequences: "
        f"{len(train_df)}"
    )


    print(
        f"Val sequences:   "
        f"{len(val_df)}"
    )


    print(
        f"Test sequences:  "
        f"{len(test_df)}"
    )


    total = (

        len(train_df)

        + len(val_df)

        + len(test_df)

    )


    print(
        f"Total sequences: "
        f"{total}"
    )


    print(
        f"\nSequence length: "
        f"{SEQUENCE_LENGTH} frames"
    )


    print(
        f"Interval: "
        f"{EXPECTED_INTERVAL_HOURS} hours"
    )


    print(
        "\nTemporal dataset creation complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()