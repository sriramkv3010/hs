"""
IGP FIX PATCH
=============
Run this ONCE to fix igp_state in the analytical dataset without rerunning
the full 30-minute climate merge.

IGP states (Indo-Gangetic Plain — correct definition):
  3=Punjab, 6=Haryana, 7=Delhi, 8=Rajasthan, 9=UP, 10=Bihar,
  19=West Bengal, 20=Jharkhand, 23=MP

Run: python3 igp_fix_patch.py
Takes ~2 minutes
"""

import pandas as pd, numpy as np, shutil, os

PROCESSED = "Dataset/processed"
IGP_CODES = [3, 6, 7, 8, 9, 10, 19, 20, 23]  # Correct full IGP definition

print("Loading dataset...")
df = pd.read_parquet(f"{PROCESSED}/final_analytical_dataset.parquet")
print(f"  Shape: {df.shape}")

# Backup
shutil.copy(
    f"{PROCESSED}/final_analytical_dataset.parquet",
    f"{PROCESSED}/final_analytical_dataset_pre_igp_fix.parquet",
)
print("  Backup saved")

# Fix igp_state
old_igp = df["igp_state"].sum()
df["igp_state"] = (
    pd.to_numeric(df["state"], errors="coerce").isin(IGP_CODES).astype("Int64")
)
new_igp = df["igp_state"].sum()

print(f"\n  IGP births before fix: {old_igp:,}")
print(f"  IGP births after fix:  {new_igp:,}")
print(f"  IGP fraction: {df['igp_state'].mean():.1%}")

# Verify LBW split
igp_lbw = df[df["igp_state"] == 1]["lbw"].mean()
nigp_lbw = df[df["igp_state"] == 0]["lbw"].mean()
print(f"  LBW IGP: {igp_lbw:.4f}   Non-IGP: {nigp_lbw:.4f}")

# Save
df.to_parquet(
    f"{PROCESSED}/final_analytical_dataset.parquet", index=False, compression="snappy"
)
df.to_csv(f"{PROCESSED}/final_analytical_dataset.csv", index=False)
print(f"\n  Saved. Now rerun: python3 final_analysis.py")
