"""
Inspect NFHS Individual Recode Files
Run: python3 inspect_individual.py
"""

import os
import pyreadstat

NFHS4_IR = os.path.expanduser(
    "~/Desktop/HS/Dataset/1.NFHS/NFHS 4 (2015-16)/IAIR74DT/IAIR74FL.DTA"
)
NFHS5_IR = os.path.expanduser(
    "~/Desktop/HS/Dataset/1.NFHS/NHFS 5 (2019-20)/IAIR7EDT/IAIR7EFL.DTA"
)

for label, path in [("NFHS-4 Individual", NFHS4_IR), ("NFHS-5 Individual", NFHS5_IR)]:
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
    print(f"\nFirst 80 variables:")
    for var in meta.column_names[:80]:
        lbl = meta.column_names_to_labels.get(var, "")
        print(f"  {var:15s}: {lbl}")

    # Search for key variables
    print(f"\nVariables related to nutrition/anaemia/BMI:")
    for var in meta.column_names:
        lbl = meta.column_names_to_labels.get(var, "").lower()
        if any(
            k in lbl
            for k in [
                "anaem",
                "anemi",
                "bmi",
                "height",
                "weight",
                "hemo",
                "haemo",
                "nutrition",
            ]
        ):
            print(f"  {var:15s}: {meta.column_names_to_labels.get(var,'')}")

    print(f"\nVariables related to district/geography:")
    for var in meta.column_names:
        lbl = meta.column_names_to_labels.get(var, "").lower()
        if any(k in lbl for k in ["district", "state", "region"]):
            print(f"  {var:15s}: {meta.column_names_to_labels.get(var,'')}")
        if var in ["sdistri", "shdist", "v024", "v025", "v101", "v102"]:
            print(f"  {var:15s}: {meta.column_names_to_labels.get(var,'')}")
