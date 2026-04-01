"""
Verify district code alignment and build crosswalk
Run: python3 verify_districts.py
"""

import pandas as pd
import geopandas as gpd
import numpy as np

# Load
df = pd.read_parquet("Dataset/processed/nfhs_merged.parquet")
shp = gpd.read_file(
    "Dataset/6.Census district boundaries/District Boundary Shapefile/India-Districts-2011Census.shp"
)

print("=== NFHS district codes ===")
nfhs_dists = sorted(df["district"].dropna().unique().astype(int).tolist())
print(f"Count: {len(nfhs_dists)}")
print(f"Min: {min(nfhs_dists)}  Max: {max(nfhs_dists)}")
print(f"First 20: {nfhs_dists[:20]}")
print(f"Last 10:  {nfhs_dists[-10:]}")

print("\n=== Shapefile census codes ===")
print(f"Count: {len(shp)}")
print(f"censuscode range: {shp['censuscode'].min()} – {shp['censuscode'].max()}")
print(f"censuscode dtype: {shp['censuscode'].dtype}")
print(f"\nFirst 20 rows (DISTRICT, censuscode, ST_CEN_CD, DT_CEN_CD):")
print(
    shp[["DISTRICT", "ST_NM", "censuscode", "ST_CEN_CD", "DT_CEN_CD"]]
    .head(20)
    .to_string()
)

print("\n=== Check if NFHS codes match censuscode ===")
shp_codes = set(shp["censuscode"].astype(int).tolist())
nfhs_set = set(nfhs_dists)
overlap = nfhs_set & shp_codes
print(f"NFHS codes that match censuscode: {len(overlap)} / {len(nfhs_set)}")
print(f"NFHS codes NOT in censuscode: {sorted(nfhs_set - shp_codes)[:20]}")

print("\n=== Check if NFHS codes are sequential row indices ===")
# If NFHS district 1 = shapefile row 0, district 2 = row 1, etc.
shp_sorted = shp.sort_values("censuscode").reset_index(drop=True)
print("Shapefile sorted by censuscode (first 10):")
print(shp_sorted[["DISTRICT", "ST_NM", "censuscode"]].head(10).to_string())

print("\n=== NFHS state codes for cross-reference ===")
print("NFHS state (v024) vs shapefile ST_CEN_CD:")
state_dist = df.groupby(["v024", "district"])["lbw"].count().reset_index()
state_dist.columns = ["state_code", "district_code", "n_births"]
print(
    state_dist.groupby("state_code")["district_code"]
    .agg(["min", "max", "count"])
    .head(20)
    .to_string()
)

print("\n=== Shapefile state codes ===")
print(shp.groupby("ST_NM")["ST_CEN_CD"].first().sort_values().to_string())
