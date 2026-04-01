"""
Verify NFHS-5 birth weight before running final analysis
python3 verify_nfhs5.py
"""

import pyreadstat, pandas as pd, numpy as np, os

path = "Dataset/1.NFHS/NHFS 5 (2019-20)/IABR7EDT/IABR7EFL.DTA"

if not os.path.exists(path):
    print(f"NOT FOUND: {path}")
    exit()

print(f"File size: {os.path.getsize(path)/1e6:.0f} MB")
print("Loading metadata only...")
_, meta = pyreadstat.read_dta(path, metadataonly=True)
print(f"Total rows in NFHS-5: checking...")

# Load key variables only — no row limit this time
print("Loading m19, caseid, v024, b3, wave indicators (no row limit)...")
print("This will take 3-5 minutes...")

df, _ = pyreadstat.read_dta(
    path,
    usecols=["caseid", "m19", "v024", "b3", "v001", "v002", "v003"],
    apply_value_formats=False,
)

print(f"\nTotal NFHS-5 rows: {len(df):,}")
print(f"Unique states (v024): {df['v024'].nunique()}")
print(
    f"State distribution: {df['v024'].value_counts().sort_index().head(10).to_dict()}"
)

m19 = pd.to_numeric(df["m19"], errors="coerce")
print(f"\nm19 stats:")
print(f"  Non-null:      {m19.notna().sum():,} ({m19.notna().mean():.1%})")
print(f"  Valid (500-9000g): {((m19>=500)&(m19<=9000)).sum():,}")
print(f"  Missing (>9000):   {(m19>9000).sum():,}")
print(f"  LBW (<2500g):      {(m19<2500).sum():,}")
print(f"  LBW rate:          {(m19[m19<=9000]<2500).mean():.4f}")

# Birth years
b3 = pd.to_numeric(df["b3"], errors="coerce")
birth_yr = (1900 + (b3 - 1) // 12).astype("Int64")
print(f"\nBirth year distribution:")
print(birth_yr.value_counts().sort_index().to_string())

# caseid sample
print(f"\ncaseid sample: {df['caseid'].head(5).tolist()}")
print(f"caseid dtype: {df['caseid'].dtype}")

# Check merge key
df["caseid_str"] = df["caseid"].astype(str).str.strip()
print(f"\nMerge key sample (caseid_str): {df['caseid_str'].head(3).tolist()}")

# Save a mapping file for the merge
bw_map = df[["caseid_str", "m19"]].copy()
bw_map["m19"] = pd.to_numeric(bw_map["m19"], errors="coerce")
bw_map["bw_valid"] = (bw_map["m19"] >= 500) & (bw_map["m19"] <= 9000)
bw_map["birthweight_g"] = bw_map["m19"].where(bw_map["bw_valid"], np.nan)
bw_map["lbw"] = (bw_map["birthweight_g"] < 2500).astype("Int64")
bw_map.loc[bw_map["birthweight_g"].isna(), "lbw"] = pd.NA

out = "Dataset/processed/nfhs5_bw_map.parquet"
bw_map[["caseid_str", "birthweight_g", "lbw"]].to_parquet(out, index=False)
print(f"\nSaved NFHS-5 BW mapping: {out}")
print(f"  Rows with valid BW: {bw_map['bw_valid'].sum():,}")
print(f"  LBW count: {bw_map['lbw'].sum():,}")
print(f"  LBW rate: {bw_map['lbw'].mean():.4f}")
