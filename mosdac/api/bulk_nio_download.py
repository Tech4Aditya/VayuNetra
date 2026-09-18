import json
import subprocess
from pathlib import Path

CONFIG = Path("config.json")

STORMS = [
    # name       start         end
    ("AMPHAN",   "2020-05-16", "2020-05-22"),
    ("NISARGA",  "2020-06-01", "2020-06-05"),
    ("YAAS",     "2021-05-23", "2021-05-28"),
    ("GULAAB",   "2021-09-24", "2021-09-29"),
    ("ASANI",    "2022-05-07", "2022-05-13"),
    ("BIPARJOY", "2023-06-06", "2023-06-20"),
    ("MICHAUNG", "2023-12-01", "2023-12-06"),
]

# NIO regions
BBOX = {
    "AMPHAN":   "80.0,5.0,95.0,25.0",
    "NISARGA":  "65.0,5.0,80.0,25.0",
    "YAAS":     "80.0,5.0,95.0,25.0",
    "GULAAB":   "80.0,5.0,95.0,25.0",
    "ASANI":    "80.0,5.0,95.0,25.0",
    "BIPARJOY": "60.0,5.0,80.0,30.0",
    "MICHAUNG": "80.0,5.0,95.0,25.0",
}

with open(CONFIG, "r", encoding="utf-8") as f:
    config = json.load(f)

for storm, start, end in STORMS:

    print("\n" + "=" * 75)
    print(f" {storm}: {start} -> {end}")
    print("=" * 75)

    config["search_parameters"]["datasetId"] = "3DIMG_L1C_ASIA_MER"
    config["search_parameters"]["startTime"] = start
    config["search_parameters"]["endTime"] = end
    config["search_parameters"]["count"] = "20"
    config["search_parameters"]["boundingBox"] = BBOX[storm]

    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    subprocess.run(["python", "mdapi.py"], check=False)

print("\n" + "=" * 75)
print(" ALL NIO DOWNLOAD WINDOWS FINISHED")
print("=" * 75)