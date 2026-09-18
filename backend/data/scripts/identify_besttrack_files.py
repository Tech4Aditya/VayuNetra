import fitz
import glob
import os

names = [
    "BULBUL",
    "YAAS",
    "GULAAB",
    "ASANI",
    "BIPARJOY",
    "MICHAUNG",
]

files = glob.glob(
    r"backend\data\raw\imd_besttrack\*.pdf"
)

for f in files:

    doc = fitz.open(f)
    text = "\n".join(
        page.get_text()
        for page in doc
    ).upper()

    print()
    print("=" * 70)
    print(os.path.basename(f))
    print("=" * 70)

    found = False

    for name in names:
        if name in text:
            print("FOUND:", name)
            found = True

    if not found:
        print("No target cyclone found")