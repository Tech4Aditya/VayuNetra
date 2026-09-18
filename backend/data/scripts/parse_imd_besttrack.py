import sys
from pathlib import Path

import pymupdf
import pandas as pd
import re


GRADE_MAP = {
    "D": "Depression",
    "DD": "Deep Depression",
    "CS": "Cyclonic Storm",
    "SCS": "Severe Cyclonic Storm",
    "VSCS": "Very Severe Cyclonic Storm",
    "ESCS": "Extremely Severe Cyclonic Storm",
    "SUCS": "Super Cyclonic Storm",
}


def is_date(s):
    return bool(re.match(r"^\d{2}[./]\d{2}[./]\d{2,4}$", s))


def is_time(s):
    return bool(re.match(r"^\d{4}$", s))


def is_number(s):
    return s == "-" or bool(re.match(r"^-?\d+(?:\.\d+)?$", s))


def normalize_date(s):
    parts = s.replace(".", "/").split("/")

    if len(parts[2]) == 2:
        parts[2] = "20" + parts[2]

    return "/".join(parts)


def parse_page(pdf_path, storm_name, output_path, page_number):

    pdf_path = Path(pdf_path)

    print(f"Reading: {pdf_path}")
    print(f"Target page: {page_number}")

    doc = pymupdf.open(pdf_path)

    if page_number < 1 or page_number > len(doc):
        raise ValueError(
            f"Invalid page {page_number}. PDF has {len(doc)} pages."
        )

    # Human page number -> zero-indexed
    text = doc[page_number - 1].get_text()

    doc.close()

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    rows = []
    current_date = None

    i = 0

    while i < len(lines):

        line = lines[i]

        if is_date(line):
            current_date = normalize_date(line)
            i += 1
            continue

        if current_date is not None and is_time(line):

            values = [line]
            j = i + 1

            while j < len(lines) and len(values) < 8:

                candidate = lines[j]

                if is_number(candidate):
                    values.append(candidate)

                elif re.fullmatch(r"[A-Za-z]+", candidate):
                    values.append(candidate)

                else:
                    break

                j += 1

            if len(values) == 8:

                (
                    time_utc,
                    latitude,
                    longitude,
                    ci,
                    pressure,
                    wind,
                    pressure_drop,
                    grade,
                ) = values

                if (
                    is_number(latitude)
                    and is_number(longitude)
                    and is_number(ci)
                    and is_number(pressure)
                    and is_number(wind)
                    and is_number(pressure_drop)
                    and re.fullmatch(r"[A-Za-z]+", grade)
                ):

                    timestamp = pd.to_datetime(
                        f"{current_date} {time_utc}",
                        format="%d/%m/%Y %H%M",
                        utc=True,
                        errors="coerce",
                    )

                    if not pd.isna(timestamp):

                        rows.append({
                            "storm_id": storm_name.upper(),
                            "timestamp": timestamp.isoformat(),
                            "latitude": float(latitude),
                            "longitude": float(longitude),
                            "wind_kt": float(wind),
                            "pressure_hpa": float(pressure),
                            "category": GRADE_MAP.get(
                                grade.upper(),
                                grade.upper()
                            ),
                        })

                        i = j
                        continue

        i += 1

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            f"No rows found on page {page_number}"
        )

    df = df.drop_duplicates(
        subset=["storm_id", "timestamp"]
    )

    df = df.sort_values("timestamp")

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 60)
    print(f"Storm        : {storm_name.upper()}")
    print(f"Rows parsed  : {len(df)}")
    print(f"Start        : {df['timestamp'].min()}")
    print(f"End          : {df['timestamp'].max()}")
    print(f"Peak wind    : {df['wind_kt'].max():.0f} kt")
    print(f"Min pressure : {df['pressure_hpa'].min():.0f} hPa")
    print(f"Output       : {output_path}")
    print("=" * 60)

    print()
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":

    if len(sys.argv) != 5:

        print(
            "Usage:"
        )

        print(
            "python parse_imd_besttrack.py "
            "<pdf> <storm> <output_csv> <page>"
        )

        sys.exit(1)

    parse_page(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
    )