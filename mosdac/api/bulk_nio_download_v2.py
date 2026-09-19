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
# TAUKTAE ONLY
#
# Best-track:
# 14 May 2021 -> 18 May 2021
#
# One request per day.
# ============================================================

STORMS = [
    (
        "TAUKTAE",
        "2021-05-14",
        "2021-05-18",
        "65.0,5.0,80.0,25.0"
    ),
]


# ============================================================
# DATE GENERATOR
# ============================================================

def dates_between(start, end):
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    current = start_date

    while current < end_date:
        yield current
        current += timedelta(days=1)


# ============================================================
# MAIN
# ============================================================

with open(CONFIG, "r", encoding="utf-8") as f:
    config = json.load(f)


for storm, start, end, bbox in STORMS:

    print()
    print("#" * 75)
    print(f"# {storm}")
    print("#" * 75)

    for day in dates_between(start, end):

        next_day = day + timedelta(days=1)

        start_time = day.strftime("%Y-%m-%d")
        end_time = next_day.strftime("%Y-%m-%d")

        print()
        print("=" * 70)
        print(f"{storm}: {start_time} -> {end_time}")
        print("=" * 70)

        # ----------------------------------------------------
        # Update MOSDAC search parameters
        # ----------------------------------------------------

        config["search_parameters"]["datasetId"] = (
            "3DIMG_L1C_ASIA_MER"
        )

        config["search_parameters"]["startTime"] = start_time
        config["search_parameters"]["endTime"] = end_time
        config["search_parameters"]["count"] = "20"
        config["search_parameters"]["boundingBox"] = bbox

        # ----------------------------------------------------
        # Save config
        # ----------------------------------------------------

        with open(CONFIG, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        # ----------------------------------------------------
        # Run official MOSDAC downloader
        # ----------------------------------------------------

        result = subprocess.run(
            ["python", "mdapi.py"],
            cwd=API_DIR,
            check=False
        )

        if result.returncode != 0:

            print(
                f"[WARN] Downloader failed for "
                f"{storm} {start_time}"
            )

            print(
                f"       Return code: "
                f"{result.returncode}"
            )

        else:

            print(
                f"[OK] Download completed: "
                f"{start_time} -> {end_time}"
            )


print()
print("#" * 75)
print("# TAUKTAE DOWNLOAD FINISHED")
print("#" * 75)