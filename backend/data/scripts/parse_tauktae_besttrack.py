import re
import csv
from pathlib import Path
import pymupdf


# ============================================================
# CONFIG
# ============================================================

PDF_DIR = Path("backend/data/raw/imd_besttrack")

OUTPUT = Path(
    "backend/data/labels/tauktae_besttrack.csv"
)

VALID_GRADES = {
    "D",
    "DD",
    "CS",
    "SCS",
    "VSCS",
    "ESCS",
}


# ============================================================
# FIND TAUKTAE PDF
# ============================================================

def find_tauktae_pdf():

    for pdf in PDF_DIR.glob("*.pdf"):

        doc = pymupdf.open(pdf)

        try:

            for page in doc:

                text = page.get_text()

                if "TAUKTAE" in text.upper():
                    return pdf

        finally:
            doc.close()

    raise FileNotFoundError(
        "Could not find TAUKTAE in the IMD best-track PDFs."
    )


# ============================================================
# HELPERS
# ============================================================

def is_date(line):

    return bool(
        re.fullmatch(
            r"\d{2}/\d{2}/\d{4}",
            line
        )
    )


def is_time(line):

    if not re.fullmatch(r"\d{4}", line):
        return False

    hour = int(line[:2])
    minute = int(line[2:])

    return (
        0 <= hour <= 23
        and 0 <= minute <= 59
    )


def parse_optional_float(value):

    if value == "-":
        return None

    return float(value)


# ============================================================
# PARSE TAUKTAE TABLE
# ============================================================

def parse_tauktae_page(text):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    rows = []

    current_date = None

    i = 0

    while i < len(lines):

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        if is_date(lines[i]):

            current_date = lines[i]

            i += 1

            continue


        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        if current_date and is_time(lines[i]):

            time = lines[i]

            values = []

            j = i + 1

            # ------------------------------------------------
            # Collect the 7 table fields
            #
            # latitude
            # longitude
            # CI
            # pressure
            # wind
            # pressure drop
            # grade
            #
            # The PDF may contain narrative text between
            # rows, so we ignore non-table text.
            # ------------------------------------------------

            while j < len(lines) and len(values) < 7:

                line = lines[j]

                # If a new date appears, this row is incomplete.
                if is_date(line):
                    break

                # If another time appears before 7 fields,
                # the current row is malformed.
                if is_time(line):
                    break

                values.append(line)

                j += 1


            # ------------------------------------------------
            # Validate row
            # ------------------------------------------------

            if len(values) == 7:

                try:

                    lat = float(values[0])

                    lon = float(values[1])

                    # CI becomes None when PDF contains "-"
                    ci = parse_optional_float(values[2])

                    pressure = float(values[3])

                    wind = float(values[4])

                    pressure_drop = parse_optional_float(
                        values[5]
                    )

                    grade = values[6]


                    if grade in VALID_GRADES:

                        timestamp = (
                            f"{current_date} "
                            f"{time[:2]}:{time[2:]}"
                        )


                        rows.append({

                            "cyclone": "TAUKTAE",

                            "timestamp": timestamp,

                            "lat": lat,

                            "lon": lon,

                            "ci": ci,

                            "pressure_hpa": pressure,

                            "wind_kt": wind,

                            "pressure_drop_hpa":
                                pressure_drop,

                            "grade": grade,

                        })


                        # Jump directly to the next
                        # unprocessed line.
                        i = j

                        continue


                except ValueError:

                    pass


        i += 1


    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    pdf = find_tauktae_pdf()

    print(f"PDF: {pdf}")

    doc = pymupdf.open(pdf)

    all_rows = []

    try:

        for page_number, page in enumerate(
            doc,
            start=1
        ):

            text = page.get_text()

            if "TAUKTAE" not in text.upper():
                continue

            print(
                f"TAUKTAE found on page {page_number}"
            )

            rows = parse_tauktae_page(text)

            print(
                f"Rows extracted from page: {len(rows)}"
            )

            all_rows.extend(rows)

    finally:

        doc.close()


    # ========================================================
    # REMOVE DUPLICATE TIMESTAMPS
    # ========================================================

    unique_rows = []

    seen = set()

    for row in all_rows:

        key = row["timestamp"]

        if key not in seen:

            seen.add(key)

            unique_rows.append(row)


    all_rows = unique_rows


    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    from datetime import datetime

    all_rows.sort(
        key=lambda row: datetime.strptime(
            row["timestamp"],
            "%d/%m/%Y %H:%M"
        )
    )


    # ========================================================
    # WRITE CSV
    # ========================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    fieldnames = [

        "cyclone",

        "timestamp",

        "lat",

        "lon",

        "ci",

        "pressure_hpa",

        "wind_kt",

        "pressure_drop_hpa",

        "grade",

    ]


    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(all_rows)


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "==================================="
    )

    print(
        "TAUKTAE EXTRACTION COMPLETE"
    )

    print(
        "==================================="
    )

    print(
        f"Observations : {len(all_rows)}"
    )

    print(
        f"Output       : {OUTPUT}"
    )

    print()


    # ========================================================
    # PRINT FIRST 5
    # ========================================================

    print("First 5:")

    for row in all_rows[:5]:

        print(row)


    print()


    # ========================================================
    # PRINT LAST 10
    # ========================================================

    print("Last 10:")

    for row in all_rows[-10:]:

        print(row)


    # ========================================================
    # PRINT TIME RANGE
    # ========================================================

    if all_rows:

        print()

        print(
            f"Start: {all_rows[0]['timestamp']}"
        )

        print(
            f"End  : {all_rows[-1]['timestamp']}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()