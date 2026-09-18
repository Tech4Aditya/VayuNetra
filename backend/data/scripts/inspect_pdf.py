import fitz
import os

path = r"backend\data\raw\imd_besttrack\33_6a2fe6738ccb6.pdf"

doc = fitz.open(path)

print("Pages:", len(doc))

for i, page in enumerate(doc):
    print()
    print("=" * 70)
    print("PAGE", i + 1)
    print("=" * 70)

    text = page.get_text()

    print(text[:5000])