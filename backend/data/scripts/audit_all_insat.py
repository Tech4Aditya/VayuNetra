from pathlib import Path
import csv
import h5py


BASE = Path(__file__).resolve().parents[2]
RAW_ROOT = BASE / "data" / "raw" / "insat" / "3DIMG_L1C_ASIA_MER"

REQUIRED = [
    "IMG_TIR1",
    "IMG_WV",
    "X",
    "Y",
    "time",
]


def validate(path):
    try:
        with h5py.File(path, "r") as f:

            missing = [
                key for key in REQUIRED
                if key not in f
            ]

            if missing:
                return False, f"missing={','.join(missing)}"

            tir_shape = f["IMG_TIR1"].shape
            wv_shape = f["IMG_WV"].shape
            x_shape = f["X"].shape
            y_shape = f["Y"].shape

            if len(tir_shape) != 3:
                return False, f"bad_IMG_TIR1_shape={tir_shape}"

            if len(wv_shape) != 3:
                return False, f"bad_IMG_WV_shape={wv_shape}"

            if len(x_shape) == 0:
                return False, f"bad_X_shape={x_shape}"

            if len(y_shape) == 0:
                return False, f"bad_Y_shape={y_shape}"

            return True, "OK"

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():

    if not RAW_ROOT.exists():
        raise FileNotFoundError(RAW_ROOT)

    files = sorted(RAW_ROOT.rglob("*.h5"))

    print("=" * 75)
    print("INSAT-3D FULL DATASET AUDIT")
    print("=" * 75)

    print(f"Root     : {RAW_ROOT}")
    print(f"H5 files : {len(files)}")
    print()

    valid = 0
    invalid = 0

    rows = []

    for i, path in enumerate(files, 1):

        ok, reason = validate(path)

        status = "VALID" if ok else "INVALID"

        if ok:
            valid += 1
        else:
            invalid += 1

        rows.append({
            "file": str(path),
            "status": status,
            "reason": reason,
        })

        if not ok:
            print(f"[BAD] {path}")
            print(f"      {reason}")

        if i % 100 == 0:
            print(f"Checked {i}/{len(files)}")

    report = BASE / "data" / "labels" / "insat_audit_report.csv"

    with open(report, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["file", "status", "reason"]
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 75)
    print("AUDIT SUMMARY")
    print("=" * 75)

    print(f"TOTAL   : {len(files)}")
    print(f"VALID   : {valid}")
    print(f"INVALID : {invalid}")

    print()
    print(f"Report  : {report}")

    if invalid == 0:
        print("\nALL INSAT FILES PASSED BASIC HDF5 VALIDATION.")
    else:
        print("\nINVALID FILES FOUND.")
        print("Do NOT delete them automatically.")
        print("Review the audit report first.")


if __name__ == "__main__":
    main()