import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import h5py


BASE = Path(__file__).resolve().parents[2]
API_DIR = Path(__file__).resolve().parent

CONFIG = API_DIR / "config.json"

RAW_ROOT = BASE / "backend" / "data" / "raw" / "insat"

DATASET_ID = "3DIMG_L1C_ASIA_MER"

STORMS = [
    ("AMPHAN",   "2020-05-16", "2020-05-22", "80.0,5.0,95.0,25.0"),
    ("NISARGA",  "2020-06-01", "2020-06-05", "65.0,5.0,80.0,25.0"),
    ("YAAS",     "2021-05-23", "2021-05-28", "80.0,5.0,95.0,25.0"),
    ("GULAAB",   "2021-09-24", "2021-09-29", "80.0,5.0,95.0,25.0"),
    ("ASANI",    "2022-05-07", "2022-05-13", "80.0,5.0,95.0,25.0"),
    ("BIPARJOY", "2023-06-06", "2023-06-20", "60.0,5.0,80.0,30.0"),
    ("MICHAUNG", "2023-12-01", "2023-12-06", "80.0,5.0,95.0,25.0"),
]


REQUIRED_DATASETS = [
    "IMG_TIR1",
    "IMG_WV",
    "X",
    "Y",
    "time",
]


def dates_between(start, end):
    """
    Inclusive date range.
    Generates one 24-hour download window per day.
    """
    d = datetime.strptime(start, "%Y-%m-%d")
    last = datetime.strptime(end, "%Y-%m-%d")

    while d <= last:
        yield d
        d += timedelta(days=1)


def validate_h5(path):
    """
    Validate one INSAT H5 product.
    """

    try:
        with h5py.File(path, "r") as f:

            missing = [
                key for key in REQUIRED_DATASETS
                if key not in f
            ]

            if missing:
                return False, f"missing={missing}"

            tir_shape = f["IMG_TIR1"].shape
            wv_shape = f["IMG_WV"].shape

            if len(tir_shape) != 3:
                return False, f"IMG_TIR1_shape={tir_shape}"

            if len(wv_shape) != 3:
                return False, f"IMG_WV_shape={wv_shape}"

            return True, "OK"

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def audit_downloads():
    """
    Validate every downloaded H5 file.
    """

    print("\n" + "=" * 75)
    print("POST-DOWNLOAD HDF5 AUDIT")
    print("=" * 75)

    files = list(RAW_ROOT.rglob("*.h5"))

    print(f"Found H5 files: {len(files)}")

    valid = 0
    invalid = 0

    for path in sorted(files):

        ok, reason = validate_h5(path)

        if ok:
            valid += 1
        else:
            invalid += 1
            print(f"[BAD] {path}")
            print(f"      {reason}")

    print("\nAUDIT SUMMARY")
    print(f"VALID   : {valid}")
    print(f"INVALID : {invalid}")

    return valid, invalid


def main():

    if not CONFIG.exists():
        raise FileNotFoundError(
            f"Missing config: {CONFIG}"
        )

    with open(CONFIG, "r", encoding="utf-8") as f:
        config = json.load(f)

    search = config["search_parameters"]

    search["datasetId"] = DATASET_ID

    total_windows = 0
    failed_windows = 0

    for storm, start, end, bbox in STORMS:

        print("\n" + "#" * 75)
        print(f"# {storm}")
        print("#" * 75)

        for day in dates_between(start, end):

            next_day = day + timedelta(days=1)

            s = day.strftime("%Y-%m-%d")
            e = next_day.strftime("%Y-%m-%d")

            total_windows += 1

            print("\n" + "=" * 70)
            print(f"{storm}: {s} -> {e}")
            print("=" * 70)

            search["startTime"] = s
            search["endTime"] = e
            search["count"] = "20"
            search["boundingBox"] = bbox

            with open(CONFIG, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)

            result = subprocess.run(
                ["python", str(API_DIR / "mdapi.py")],
                cwd=API_DIR,
                check=False
            )

            if result.returncode != 0:
                failed_windows += 1

                print(
                    f"[WARN] {storm} {s}: "
                    f"mdapi exited with code "
                    f"{result.returncode}"
                )

            else:
                print("[OK] Download window completed")

    print("\n" + "#" * 75)
    print("# ACQUISITION COMPLETE")
    print("#" * 75)

    print(f"Total windows : {total_windows}")
    print(f"Failed        : {failed_windows}")

    audit_downloads()


if __name__ == "__main__":
    main()