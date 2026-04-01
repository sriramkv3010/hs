"""
Inspect NFHS Household Recode Files
Run: python3 inspect_household.py
"""

import os
import pyreadstat

NFHS4_HH = os.path.expanduser(
    "~/Desktop/HS/Dataset/1.NFHS/NFHS 4 (2015-16)/IAHR74DT/IAHR74FL.DTA"
)
NFHS5_HH = os.path.expanduser(
    "~/Desktop/HS/Dataset/1.NFHS/NHFS 5 (2019-20)/IAHR7EDT/IAHR7EFL.DTA"
)

for label, path in [("NFHS-4 Household", NFHS4_HH), ("NFHS-5 Household", NFHS5_HH)]:
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        continue

    size_mb = os.path.getsize(path) / 1e6
    print(f"Size: {size_mb:.1f} MB")

    _, meta = pyreadstat.read_dta(path, metadataonly=True)
    print(f"Total variables: {len(meta.column_names)}")
    print(f"\nAll variables and labels:")
    for var in meta.column_names:
        lbl = meta.column_names_to_labels.get(var, "")
        print(f"  {var:15s}: {lbl}")
