import fitz

path = r"data\raw\imd_besttrack\imd_annual_report_2021.pdf"

doc = fitz.open(path)

keywords = [
    "SEPTEMBER",
    "23.09.2021",
    "24.09.2021",
    "25.09.2021",
    "26.09.2021",
    "27.09.2021",
    "28.09.2021",
    "29.09.2021",
    "30.09.2021",
]

for i, page in enumerate(doc):

    text = page.get_text()

    matches = [
        k for k in keywords
        if k in text.upper()
    ]

    if matches:

        print()
        print("=" * 70)
        print("PAGE", i + 1, "|", matches)
        print("=" * 70)

        print(text[:6000])