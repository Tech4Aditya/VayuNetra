import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta


# ============================================================
# PATHS
# ============================================================

API_DIR = Path(__file__).resolve().parent
CONFIG = API_DIR / "config.json"


# ============================================================
# DATASET
# ============================================================

DATASET_ID = "3DIMG_L1C_ASIA_MER"


# ============================================================
# NIO CYCLONES
#
# Date ranges are inclusive.
# One MOSDAC request is made per 24-hour window.
# ============================================================

STORMS = [
    # Storm       Start          End            Bounding Box
    ("AMPHAN",    "2020-05-16", "2020-05-22", "80.0,5.0,95.0,25.0"),
    ("NISARGA",   "2020-06-01", "2020-06-05", "65.0,5.0,80.0,25.0"),
    ("YAAS",      "2021-05-23", "2021-05-28", "80.0,5.0,95.0,25.0"),
    ("TAUKTAE",   "2021-05-14", "2021-05-18", "65.0,5.0,80.0,25.0"),
    ("GULAAB",    "2021-09-24", "2021-09-29", "80.0,5.0,95.0,25.0"),
    ("ASANI",     "2022-05-07", "2022-05-13", "80.0,5.0,95.0,25.0"),
    ("BIPARJOY",  "2023-06-06", "2023-06-20", "60.0,5.0,80.0,30.0"),
    ("MICHAUNG",  "2023-12-01", "2023-12-06", "80.0,5.0,95.0,25.0"),
]


# ============================================================
# DATE WINDOWS
# ============================================================

def dates_between(start, end):
    """
    Generate every date from start to end, inclusive.
    Each date becomes a 24-hour MOSDAC query window.
    """

    current = datetime.strptime(start, "%Y-%m-%d")
    last = datetime.strptime(end, "%Y-%m-%d")

    while current <= last:
        yield current
        current += timedelta(days=1)


# ============================================================
# MAIN
# ============================================================

def main():

    if not CONFIG.exists():
        raise FileNotFoundError(
            f"config.json not found: {CONFIG}"
        )

    with open(CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)

    search = config["search_parameters"]

    # Fixed dataset
    search["datasetId"] = DATASET_ID

    total_windows = 0
    failed_windows = 0

    for storm, start, end, bbox in STORMS:

        print()
        print("#" * 75)
        print(f"# {storm}")
        print("#" * 75)

        for day in dates_between(start, end):

            next_day = day + timedelta(days=1)

            start_time = day.strftime("%Y-%m-%d")
            end_time = next_day.strftime("%Y-%m-%d")

            total_windows += 1

            print()
            print("=" * 70)
            print(f"{storm}: {start_time} -> {end_time}")
            print("=" * 70)

            # Update search parameters
            search["datasetId"] = DATASET_ID
            search["startTime"] = start_time
            search["endTime"] = end_time
            search["count"] = "20"
            search["boundingBox"] = bbox

            # Write temporary search configuration
            with open(CONFIG, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)

            # Run official MOSDAC API client
            result = subprocess.run(
                ["python", str(API_DIR / "mdapi.py")],
                cwd=API_DIR,
                check=False
            )

            if result.returncode != 0:

                failed_windows += 1

                print(
                    f"[WARN] {storm} {start_time}: "
                    f"mdapi exited with code "
                    f"{result.returncode}"
                )

            else:

                print(
                    f"[OK] {storm} "
                    f"{start_time} -> {end_time}"
                )

    print()
    print("#" * 75)
    print("# NIO INSAT ACQUISITION COMPLETE")
    print("#" * 75)

    print(f"Total download windows : {total_windows}")
    print(f"Failed windows         : {failed_windows}")

    if failed_windows == 0:
        print("Status                 : ALL WINDOWS COMPLETED")
    else:
        print(
            "Status                 : "
            "SOME WINDOWS FAILED — CHECK MOSDAC ERROR LOGS"
        )


if __name__ == "__main__":
    main()