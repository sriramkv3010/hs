import os
import numpy as np

# Check what IMD files exist
print("=== IMD FILES ===")
search_dirs = [
    "Dataset",
    "Dataset/raw",
    "Dataset/processed",
    "data",
    "Data",
    ".",
    "raw",
]
for d in search_dirs:
    if os.path.exists(d):
        for f in os.listdir(d):
            if any(k in f.lower() for k in ["imd", "tmax", "grd", "nc", "grid"]):
                fp = os.path.join(d, f)
                print(f"  {fp}  ({os.path.getsize(fp)/1e6:.1f} MB)")

# Check parquet for N5 lat/lon and conception variables
import pandas as pd

raw = pd.read_parquet("Dataset/processed/final_analytical_dataset.parquet")
n5 = raw[raw.wave == 5].copy()

print("\n=== N5 LAT/LON COVERAGE ===")
print(f"  tmax_lat  notna: {n5['tmax_lat'].notna().sum():,}")
print(f"  tmax_lon  notna: {n5['tmax_lon'].notna().sum():,}")
print(f"  tmax_lat  range: {n5['tmax_lat'].min():.2f} to {n5['tmax_lat'].max():.2f}")
print(f"  tmax_lon  range: {n5['tmax_lon'].min():.2f} to {n5['tmax_lon'].max():.2f}")
print(
    f"  Unique lat/lon pairs: "
    f"{n5[['tmax_lat','tmax_lon']].dropna().drop_duplicates().shape[0]:,}"
)

print("\n=== N5 BIRTH/CONCEPTION DATE INPUTS ===")
for col in ["b3", "b3_num", "m18", "birth_year", "birth_month"]:
    if col in n5.columns:
        print(
            f"  {col}: notna={n5[col].notna().sum():,}  "
            f"sample={n5[col].dropna().head(3).tolist()}"
        )

print("\n=== HOW birth_month_tmax_anomaly WAS BUILT FOR N5 ===")
# Check if there's a lookup table or if it came from the grd directly
bma = n5[n5["birth_month_tmax_anomaly"].notna()][
    ["tmax_lat", "tmax_lon", "birth_year", "birth_month", "birth_month_tmax_anomaly"]
].head(5)
print(bma.to_string())
