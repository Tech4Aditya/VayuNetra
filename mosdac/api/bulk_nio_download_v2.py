import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

CONFIG = Path("config.json")

STORMS = [
    ("AMPHAN",   "2020-05-16", "2020-05-22", "80.0,5.0,95.0,25.0"),
    ("NISARGA",  "2020-06-01", "2020-06-05", "65.0,5.0,80.0,25.0"),
    ("YAAS",     "2021-05-23", "2021-05-28", "80.0,5.0,95.0,25.0"),
    ("GULAAB",   "2021-09-24", "2021-09-29", "80.0,5.0,95.0,25.0"),
    ("ASANI",    "2022-05-07", "2022-05-13", "80.0,5.0,95.0,25.0"),
    ("BIPARJOY", "2023-06-06", "2023-06-20", "60.0,5.0,80.0,30.0"),
    ("MICHAUNG", "2023-12-01", "2023-12-06", "80.0,5.0,95.0,25.0"),
]

with open(CONFIG, "r", encoding="utf-8") as f:
    config = json.load(f)

def dates_between(start, end):
    d = datetime.strptime(start, "%Y-%m-%d")
    last = datetime.strptime(end, "%Y-%m-%d")

    while d < last:
        yield d
        d += timedelta(days=1)

for storm, start, end, bbox in STORMS:

    print("\n" + "#" * 75)
    print(f"# {storm}")
    print("#" * 75)

    for day in dates_between(start, end):

        next_day = day + timedelta(days=1)

        s = day.strftime("%Y-%m-%d")
        e = next_day.strftime("%Y-%m-%d")

        print("\n" + "=" * 70)
        print(f"{storm}: {s} -> {e}")
        print("=" * 70)

        config["search_parameters"]["datasetId"] = \
            "3DIMG_L1C_ASIA_MER"

        config["search_parameters"]["startTime"] = s
        config["search_parameters"]["endTime"] = e
        config["search_parameters"]["count"] = "20"
        config["search_parameters"]["boundingBox"] = bbox

        with open(CONFIG, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        result = subprocess.run(
            ["python", "mdapi.py"],
            check=False
        )

        if result.returncode != 0:
            print(f"[WARN] Downloader exited with code {result.returncode}")

print("\n" + "#" * 75)
print("# ALL NIO DAILY WINDOWS FINISHED")
print("#" * 75)