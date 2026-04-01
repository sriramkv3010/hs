"""
Quick merge key diagnostic
python3 diagnose_merges.py
"""

import pandas as pd, numpy as np

PROCESSED = "Dataset/processed"

print("=" * 60)
print("MERGE KEY DIAGNOSTIC")
print("=" * 60)

# Load a small sample of births
births_sample = pd.read_parquet(f"{PROCESSED}/nfhs_births_combined.parquet").head(5)
print("\n--- BIRTHS keys ---")
print(f"v001: {births_sample['v001'].tolist()}")
print(f"v002: {births_sample['v002'].tolist()}")
print(f"v001 dtype: {births_sample['v001'].dtype}")
print(f"v002 dtype: {births_sample['v002'].dtype}")
print(f"wave: {births_sample['wave'].tolist()}")

# Household
hh = pd.read_parquet(f"{PROCESSED}/nfhs_household_combined.parquet").head(5)
print("\n--- HOUSEHOLD keys ---")
print(f"hh_merge_key sample: {hh['hh_merge_key'].head(5).tolist()}")
print(f"hv001: {hh['hv001'].head(5).tolist()}")
print(f"hv002: {hh['hv002'].head(5).tolist()}")
print(f"wave: {hh['wave'].head(5).tolist()}")

# Individual
ir = pd.read_parquet(f"{PROCESSED}/nfhs_individual_combined.parquet").head(5)
print("\n--- INDIVIDUAL keys ---")
print(f"ir_merge_key sample: {ir['ir_merge_key'].head(5).tolist()}")
print(
    f"Cols with 'key': {[c for c in ir.columns if 'key' in c.lower() or 'merge' in c.lower()]}"
)
print(f"v001: {ir['v001'].head(5).tolist()}")
print(f"v002: {ir['v002'].head(5).tolist()}")
print(f"v003: {ir['v003'].head(5).tolist()}")
print(f"wave: {ir['wave'].head(5).tolist()}")

# NHS
import csv

nhs = pd.read_csv(f"{PROCESSED}/nhs_combined.csv")
print("\n--- NHS keys ---")
print(f"state col: {nhs['state'].head(5).tolist()}")
print(f"year col: {nhs['year'].head(5).tolist()}")
print(f"wave col: {nhs['wave'].head(5).tolist()}")
