"""
Rebuild NFHS-5 birth weight map with correct row-level identifier
The problem: caseid identifies women not births — multiple births per woman
Solution: use row position (same order as births recode) OR caseid+b3 (birth CMC)

Run: python3 rebuild_nfhs5_map.py
"""

import pyreadstat, pandas as pd, numpy as np, os

PROCESSED = "Dataset/processed"
path5 = "Dataset/1.NFHS/NHFS 5 (2019-20)/IABR7EDT/IABR7EFL.DTA"

print("Loading NFHS-5 births recode (full file, key vars only)...")
print("This takes 3-5 minutes...")

df5, meta = pyreadstat.read_dta(
    path5,
    usecols=["caseid", "v001", "v002", "v003", "b3", "m19"],
    apply_value_formats=False,
)
print(f"Total rows: {len(df5):,}")
print(f"caseid sample: {df5['caseid'].head(3).tolist()}")
print(f"caseid dtype: {df5['caseid'].dtype}")

# Normalise caseid
df5["caseid_str"] = df5["caseid"].astype(str).str.strip()

# m19 is birth weight in grams
m19 = pd.to_numeric(df5["m19"], errors="coerce")
bw_g = m19.where((m19 >= 500) & (m19 <= 9000), np.nan)
df5["birthweight_g"] = bw_g
df5["lbw"] = (bw_g < 2500).astype("Int64")
df5.loc[bw_g.isna(), "lbw"] = pd.NA

print(f"\nValid birth weights: {bw_g.notna().sum():,}")
print(f"LBW count: {df5['lbw'].sum():,}")
print(f"LBW rate: {df5['lbw'].mean():.4f}")

# Build merge key using caseid + b3 (birth CMC = unique per birth per woman)
df5["b3_num"] = pd.to_numeric(df5["b3"], errors="coerce")
df5["merge_key"] = (
    df5["caseid_str"] + "_" + df5["b3_num"].fillna(-1).astype(int).astype(str)
)

print(f"\nUnique merge keys: {df5['merge_key'].nunique():,}")
print(f"Total rows: {len(df5):,}")
print(f"Duplicate keys: {df5.duplicated('merge_key').sum():,}")
print(f"\nSample merge keys:")
print(
    df5[["caseid_str", "b3_num", "merge_key", "birthweight_g", "lbw"]]
    .head(5)
    .to_string()
)

# Save the corrected map
out_path = f"{PROCESSED}/nfhs5_bw_map_v2.parquet"
df5[["caseid_str", "b3_num", "merge_key", "birthweight_g", "lbw"]].to_parquet(
    out_path, index=False
)
print(f"\nSaved: {out_path}")

# Also save row-order map (for index-based merge)
df5_clean = df5[["birthweight_g", "lbw"]].copy()
df5_clean["row_idx"] = np.arange(len(df5))
df5_clean.to_parquet(f"{PROCESSED}/nfhs5_bw_map_roworder.parquet", index=False)
print(f"Saved row-order map: {PROCESSED}/nfhs5_bw_map_roworder.parquet")

# Now check what the final_analytical_dataset has for wave==5
print("\n=== Checking merge compatibility ===")
raw = pd.read_parquet(f"{PROCESSED}/final_analytical_dataset.parquet")
raw5 = raw[raw["wave"] == 5].copy()
raw5["caseid_str"] = raw5["caseid"].astype(str).str.strip()
raw5["b3_num"] = pd.to_numeric(raw5["b3"], errors="coerce")
raw5["merge_key"] = (
    raw5["caseid_str"] + "_" + raw5["b3_num"].fillna(-1).astype(int).astype(str)
)

print(f"raw wave==5: {len(raw5):,}")
print(f"raw5 caseid sample: {raw5['caseid'].head(3).tolist()}")
print(f"raw5 merge_key sample: {raw5['merge_key'].head(3).tolist()}")
print(f"df5 merge_key sample:  {df5['merge_key'].head(3).tolist()}")

# Check overlap
overlap = set(raw5["merge_key"]) & set(df5["merge_key"])
print(f"\nMerge key overlap: {len(overlap):,} / {len(raw5):,} raw5 rows")

if len(overlap) > 0:
    print("✅ Merge will work!")
else:
    print("❌ Still no overlap — checking why...")
    print(f"\nraw5 caseid format:  '{raw5['caseid_str'].iloc[0]}'")
    print(f"df5  caseid format:  '{df5['caseid_str'].iloc[0]}'")
    print(f"raw5 b3_num sample: {raw5['b3_num'].head(5).tolist()}")
    print(f"df5  b3_num sample:  {df5['b3_num'].head(5).tolist()}")

    # Try row-order check
    print(
        f"\nRow count match: raw5={len(raw5):,}  df5={len(df5):,}  match={len(raw5)==len(df5)}"
    )
