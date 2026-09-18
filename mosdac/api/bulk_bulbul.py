import json, subprocess, os

cfg = "config.json"

with open(cfg, "r", encoding="utf-8") as f:
    config = json.load(f)

ranges = [
    ("2019-11-05", "2019-11-06"),
    ("2019-11-06", "2019-11-07"),
    ("2019-11-07", "2019-11-08"),
    ("2019-11-08", "2019-11-09"),
    ("2019-11-09", "2019-11-10"),
]

for start, end in ranges:
    print("\n" + "="*70)
    print(f"BULBUL DOWNLOAD: {start} -> {end}")
    print("="*70)

    config["search_parameters"]["startTime"] = start
    config["search_parameters"]["endTime"] = end
    config["search_parameters"]["count"] = "20"

    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    subprocess.run(["python", "mdapi.py"])

print("\n\nALL BULBUL WINDOWS FINISHED.")
