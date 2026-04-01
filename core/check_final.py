"""
Quick dataset check before ML
Run: python3 check_final.py
"""

import pandas as pd
import numpy as np

df = pd.read_parquet("Dataset/processed/final_analytical_dataset.parquet")

print("Shape:", df.shape)
print("\nAll columns:")
for c in df.columns:
    miss = df[c].isna().mean() * 100
    print(f"  {c:45s} dtype={str(df[c].dtype):12s} missing={miss:.1f}%")

print("\nOutcome rates:")
for col in ["lbw", "preterm", "neonatal_death", "infant_death"]:
    if col in df.columns:
        print(f"  {col}: {df[col].mean():.4f}")

print("\nClimate variables present:")
clim = [
    c
    for c in df.columns
    if "tmax" in c
    or "anomaly" in c
    or "hot_days" in c
    or "rainfall" in c
    or "drought" in c
]
for c in clim:
    print(f"  {c}: missing={df[c].isna().mean():.1%}  mean={df[c].mean():.3f}")

print("\nAnalysis sample sizes:")
for col in ["analysis_full", "analysis_trimester", "analysis_ready"]:
    if col in df.columns:
        print(f"  {col}: {df[col].sum():,}")
