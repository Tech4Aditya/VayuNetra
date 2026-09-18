import sys
import re
import os

import pandas as pd
import pymupdf


# ============================================================
# CATEGORY MAP
# ============================================================

CATEGORY_MAP = {
    "D": "Depression",
    "DD": "Deep Depression",
    "CS": "Cyclonic Storm",
    "SCS": "Severe Cyclonic Storm",
    "VSCS": "Very Severe Cyclonic Storm",
    "ESCS": "Extremely Severe Cyclonic Storm",
    "SuCS": "Super Cyclonic Storm",
}


# ============================================================
# HELPERS
# ============================================================

def clean_token(value):
    """
    Clean text extracted from PDF.
    """
    return str(value).strip().replace("\xa0", " ")


def is_number(value):
    """
    Check whether a token is numeric.
    """
    try:
        float(clean_token(value))
        return True
    except (ValueError, TypeError):
        return False


def is_time_line(value):
    """
    Detect times such as:

        0000
        0300
        0600
        1200
        1800
        2100
    """

    value = clean_token(value)

    return bool(
        re.fullmatch(r"\d{4}", value)
    )


def is_date_line(value):
    """
    Detect common IMD date formats:

        23.05.21
        23.05.2021
        23/05/2021
        23-05-2021
    """

    value = clean_token(value)

    patterns = [
        r"^\d{2}\.\d{2}\.\d{2}$",
        r"^\d{2}\.\d{2}\.\d{4}$",
        r"^\d{2}/\d{2}/\d{4}$",
        r"^\d{2}-\d{2}-\d{4}$",
    ]

    for pattern in patterns:

        if re.fullmatch(pattern, value):
            return True

    return False


def normalize_date(date_str):
    """
    Convert supported date formats to:

        DD/MM/YYYY
    """

    date_str = clean_token(date_str)

    # --------------------------------------------------------
    # DD/MM/YYYY
    # --------------------------------------------------------

    if re.fullmatch(
        r"\d{2}/\d{2}/\d{4}",
        date_str
    ):

        return date_str

    # --------------------------------------------------------
    # DD-MM-YYYY
    # --------------------------------------------------------

    if re.fullmatch(
        r"\d{2}-\d{2}-\d{4}",
        date_str
    ):

        d, m, y = date_str.split("-")

        return f"{d}/{m}/{y}"

    # --------------------------------------------------------
    # DD.MM.YY / DD.MM.YYYY
    # --------------------------------------------------------

    if "." in date_str:

        parts = date_str.split(".")

        if len(parts) == 3:

            d, m, y = parts

            if len(y) == 2:
                y = "20" + y

            return f"{d}/{m}/{y}"

    return None


# ============================================================
# PDF READER
# ============================================================

def parse_pages(pdf_path, pages):
    """
    Read selected 1-based PDF pages.

    Example:

        [4]

    or:

        [2, 3]

    or:

        [5, 6]

    The parser allows a target table to continue onto the
    following page.

    If another table begins later, parsing stops before that
    unrelated table.
    """

    doc = pymupdf.open(pdf_path)

    try:

        page_count = len(doc)

        # ----------------------------------------------------
        # Validate requested pages.
        # ----------------------------------------------------

        for page in pages:

            if page < 1 or page > page_count:

                raise ValueError(
                    f"Page {page} is outside PDF range "
                    f"1-{page_count}"
                )

        text_parts = []

        # ----------------------------------------------------
        # Track whether we have encountered the target table.
        # ----------------------------------------------------

        target_table_started = False

        for page_number in pages:

            text = doc[
                page_number - 1
            ].get_text()

            # ------------------------------------------------
            # Find table headings.
            #
            # Examples:
            #
            # Table 1:
            # Table 2:
            # Table 3:
            # ------------------------------------------------

            table_matches = list(
                re.finditer(
                    r"\bTable\s+\d+\s*:",
                    text,
                    flags=re.IGNORECASE
                )
            )

            # ------------------------------------------------
            # If this page contains a table heading.
            # ------------------------------------------------

            if table_matches:

                # --------------------------------------------
                # First table heading encountered.
                # --------------------------------------------

                if not target_table_started:

                    target_table_started = True

                    # Keep the page unless there is another
                    # table heading later on the same page.
                    #
                    # Example:
                    #
                    # Table 1: AMPHAN
                    # ...
                    # Table 2: Oman depression
                    #
                    # We only want Table 1.
                    # ----------------------------------------

                    if len(table_matches) > 1:

                        text = text[
                            :table_matches[1].start()
                        ]

                # --------------------------------------------
                # A table had already started and another
                # table heading appears.
                #
                # This is the beginning of another storm/table.
                # --------------------------------------------

                else:

                    text = text[
                        :table_matches[0].start()
                    ]

                    text_parts.append(text)

                    break

            # ------------------------------------------------
            # Store page text.
            # ------------------------------------------------

            text_parts.append(text)

        return "\n".join(text_parts)

    finally:

        doc.close()


# ============================================================
# BEST-TRACK PARSER
# ============================================================

def parse_numeric_rows(text):
    """
    Parse IMD best-track rows from PDF-extracted text.

    Expected logical structure:

        DATE
        TIME
        LATITUDE
        LONGITUDE
        CI
        PRESSURE
        WIND
        PRESSURE DROP
        CATEGORY

    PDF extraction can disturb the column order, so CATEGORY
    is used as an anchor.

    Dates are carried forward for continuation rows.
    """

    # --------------------------------------------------------
    # Clean PDF lines.
    # --------------------------------------------------------

    lines = [
        clean_token(line)
        for line in text.splitlines()
        if clean_token(line)
    ]

    rows = []

    current_date = None

    i = 0

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while i < len(lines):

        line = lines[i]

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        if is_date_line(line):

            normalized = normalize_date(line)

            if normalized is not None:

                current_date = normalized

            i += 1

            continue

        # ----------------------------------------------------
        # We cannot parse a time without a known date.
        # ----------------------------------------------------

        if current_date is None:

            i += 1

            continue

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        if not is_time_line(line):

            i += 1

            continue

        time = line

        # ----------------------------------------------------
        # Search for category.
        #
        # Normally the category appears within ~6-8 tokens.
        # Allow a slightly larger window because PDF extraction
        # can insert extra tokens.
        # ----------------------------------------------------

        category_idx = None
        category_code = None

        search_end = min(
            i + 12,
            len(lines)
        )

        for j in range(
            i + 1,
            search_end
        ):

            token = clean_token(lines[j])

            if token in CATEGORY_MAP:

                category_idx = j
                category_code = token

                break

        # ----------------------------------------------------
        # No category found.
        # ----------------------------------------------------

        if category_idx is None:

            i += 1

            continue

        # ----------------------------------------------------
        # Extract values between TIME and CATEGORY.
        # ----------------------------------------------------

        values = lines[
            i + 1:category_idx
        ]

        # ----------------------------------------------------
        # Collect numeric tokens.
        #
        # CI may be numeric or '-'.
        # Pressure-drop may be numeric or '-'.
        # ----------------------------------------------------

        numeric_values = []

        for value in values:

            if is_number(value):

                numeric_values.append(
                    float(value)
                )

        # Need at minimum:
        #
        # latitude
        # longitude
        # pressure
        # wind
        # ----------------------------------------------------

        if len(numeric_values) < 4:

            i += 1

            continue

        # ====================================================
        # LATITUDE / LONGITUDE
        # ====================================================

        latitude = numeric_values[0]
        longitude = numeric_values[1]

        # ----------------------------------------------------
        # Geographic sanity checks.
        #
        # These bounds cover the Indian Ocean / India region
        # used by the IMD best-track reports here.
        # ----------------------------------------------------

        if not (
            -30 <= latitude <= 40
        ):

            i += 1

            continue

        if not (
            30 <= longitude <= 110
        ):

            i += 1

            continue

        # ====================================================
        # PRESSURE
        # ====================================================

        pressure_candidates = [
            value
            for value in numeric_values[2:]
            if 850 <= value <= 1050
        ]

        if not pressure_candidates:

            i += 1

            continue

        pressure = pressure_candidates[0]

        # ----------------------------------------------------
        # Find the position of pressure in numeric_values.
        # ----------------------------------------------------

        pressure_position = None

        for k, value in enumerate(
            numeric_values
        ):

            if (
                k >= 2
                and value == pressure
            ):

                pressure_position = k

                break

        if pressure_position is None:

            i += 1

            continue

        # ====================================================
        # WIND
        # ====================================================

        after_pressure = numeric_values[
            pressure_position + 1:
        ]

        wind_candidates = [
            value
            for value in after_pressure
            if 0 <= value <= 180
        ]

        if not wind_candidates:

            i += 1

            continue

        wind = wind_candidates[0]

        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp = pd.to_datetime(
            f"{current_date} {time}",
            format="%d/%m/%Y %H%M",
            utc=True,
            errors="coerce",
        )

        if pd.isna(timestamp):

            i += 1

            continue

        # ====================================================
        # CONTAMINATION PROTECTION
        # ====================================================
        #
        # Best-track tables are normally separated by a few
        # hours. If we suddenly jump multiple days, we have
        # almost certainly entered another storm's table.
        #
        # Example:
        #
        # AMPHAN:
        # 21 May 12:00
        #
        # then accidentally:
        # 29 May 09:00
        #
        # That is a different table.
        # ====================================================

        if rows:

            previous_timestamp = rows[-1][
                "timestamp"
            ]

            gap_hours = (
                timestamp -
                previous_timestamp
            ).total_seconds() / 3600.0

            if gap_hours > 48:

                print(
                    "\nStopping parser: "
                    f"timestamp gap of "
                    f"{gap_hours:.1f} hours detected."
                )

                print(
                    f"Previous: {previous_timestamp}"
                )

                print(
                    f"Current : {timestamp}"
                )

                print(
                    "This likely indicates the beginning "
                    "of another storm/table."
                )

                break

        # ====================================================
        # APPEND ROW
        # ====================================================

        rows.append(
            {
                "timestamp": timestamp,
                "latitude": latitude,
                "longitude": longitude,
                "wind_kt": wind,
                "pressure_hpa": pressure,
                "category": CATEGORY_MAP[
                    category_code
                ],
            }
        )

        # ----------------------------------------------------
        # Move after category.
        # ----------------------------------------------------

        i = category_idx + 1

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CLI validation
    # --------------------------------------------------------

    if len(sys.argv) != 5:

        print(
            "\nUsage:\n"
            "python parse_imd_besttrack.py "
            '"<pdf>" "<storm>" "<output_csv>" "<pages>"\n\n'

            "Examples:\n"

            'python parse_imd_besttrack.py '
            '"file.pdf" '
            '"YAAS" '
            '"yaas.csv" '
            '"4"\n'

            'python parse_imd_besttrack.py '
            '"file.pdf" '
            '"AMPHAN" '
            '"amphan.csv" '
            '"2,3"\n'

            'python parse_imd_besttrack.py '
            '"file.pdf" '
            '"ASANI" '
            '"asani.csv" '
            '"5,6"\n'
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    pdf_path = sys.argv[1]
    storm = sys.argv[2]
    output_csv = sys.argv[3]
    page_string = sys.argv[4]

    # --------------------------------------------------------
    # Parse page numbers.
    # --------------------------------------------------------

    try:

        pages = [
            int(page.strip())
            for page in page_string.split(",")
            if page.strip()
        ]

    except ValueError:

        print(
            "\nERROR: Invalid page list."
        )

        print(
            'Use something like "4" or "2,3" or "5,6".'
        )

        sys.exit(1)

    if not pages:

        print(
            "\nERROR: No pages specified."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Print input.
    # --------------------------------------------------------

    print(
        f"Reading: {pdf_path}"
    )

    print(
        f"Target pages: {pages}"
    )

    # --------------------------------------------------------
    # Check PDF.
    # --------------------------------------------------------

    if not os.path.exists(pdf_path):

        print(
            f"\nERROR: PDF not found:\n{pdf_path}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Read PDF.
    # --------------------------------------------------------

    try:

        text = parse_pages(
            pdf_path,
            pages
        )

    except Exception as e:

        print(
            f"\nERROR reading PDF: {e}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Parse.
    # --------------------------------------------------------

    rows = parse_numeric_rows(
        text
    )

    if not rows:

        print(
            "\nERROR: No best-track rows parsed."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # DataFrame.
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Add storm ID.
    # --------------------------------------------------------

    df.insert(
        0,
        "storm_id",
        storm
    )

    # --------------------------------------------------------
    # Remove duplicate timestamps.
    # --------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=["timestamp"],
            keep="first"
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    # --------------------------------------------------------
    # Check chronological ordering.
    # --------------------------------------------------------

    if not df[
        "timestamp"
    ].is_monotonic_increasing:

        print(
            "\nWARNING: timestamps are not "
            "strictly increasing."
        )

    # --------------------------------------------------------
    # Check coordinate ranges.
    # --------------------------------------------------------

    invalid_coordinates = df[
        (~df["latitude"].between(-30, 40))
        |
        (~df["longitude"].between(30, 110))
    ]

    if len(invalid_coordinates) > 0:

        print(
            "\nWARNING:"
            f" {len(invalid_coordinates)} rows "
            "have unusual coordinates."
        )

    # --------------------------------------------------------
    # Check pressure.
    # --------------------------------------------------------

    invalid_pressure = df[
        (~df["pressure_hpa"].between(850, 1050))
    ]

    if len(invalid_pressure) > 0:

        print(
            "\nWARNING:"
            f" {len(invalid_pressure)} rows "
            "have unusual pressure."
        )

    # --------------------------------------------------------
    # Check wind.
    # --------------------------------------------------------

    invalid_wind = df[
        (~df["wind_kt"].between(0, 180))
    ]

    if len(invalid_wind) > 0:

        print(
            "\nWARNING:"
            f" {len(invalid_wind)} rows "
            "have unusual wind."
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = os.path.dirname(
        output_csv
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    df.to_csv(
        output_csv,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        f"Storm        : {storm}"
    )

    print(
        f"Rows parsed  : {len(df)}"
    )

    print(
        f"Start        : {df['timestamp'].min()}"
    )

    print(
        f"End          : {df['timestamp'].max()}"
    )

    print(
        f"Peak wind    : "
        f"{df['wind_kt'].max():.0f} kt"
    )

    print(
        f"Min pressure : "
        f"{df['pressure_hpa'].min():.0f} hPa"
    )

    print(
        f"Output       : {output_csv}"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # CATEGORY COUNTS
    # ========================================================

    print(
        "\nCategory counts:"
    )

    print(
        df["category"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # FULL TABLE
    # ========================================================

    print(
        "\nParsed rows:\n"
    )

    print(
        df.to_string(
            index=False
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()