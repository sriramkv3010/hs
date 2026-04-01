"""
Pre-flight check — run before final analysis
python3 preflight.py
"""

import os, pandas as pd, numpy as np

PROCESSED = "Dataset/processed"

print("=" * 60)
print("PRE-FLIGHT CHECK")
print("=" * 60)

# 1. Final dataset
df = pd.read_parquet(f"{PROCESSED}/final_analytical_dataset.parquet")
print(f"\n[1] Final dataset: {df.shape}")
print(f"    Columns with 'era5': {[c for c in df.columns if 'era5' in c.lower()]}")
print(f"    v024 (state) present: {'v024' in df.columns}")
print(
    f"    v024 unique vals: {df['v024'].nunique() if 'v024' in df.columns else 'N/A'}"
)
print(
    f"    v024 sample: {df['v024'].dropna().head(5).tolist() if 'v024' in df.columns else 'N/A'}"
)
print(f"    district sample: {df['district'].dropna().head(5).tolist()}")
print(f"    district dtype: {df['district'].dtype}")

# 2. ERA5 files
print(f"\n[2] ERA5 files:")
for fn in os.listdir(PROCESSED):
    if "era5" in fn.lower():
        path = f"{PROCESSED}/{fn}"
        size = os.path.getsize(path) / 1e6
        print(f"    {fn}  ({size:.1f} MB)")
        if fn.endswith(".parquet"):
            try:
                e = pd.read_parquet(path)
                print(f"      Shape: {e.shape}")
                print(f"      Cols: {e.columns.tolist()}")
                print(
                    f"      Years: {e['year'].min() if 'year' in e.columns else 'N/A'} - {e['year'].max() if 'year' in e.columns else 'N/A'}"
                )
                print(f"      Sample:\n{e.head(2)}")
            except Exception as ex:
                print(f"      Load error: {ex}")
        elif fn.endswith(".csv"):
            try:
                e = pd.read_csv(path, nrows=3)
                print(f"      Cols: {e.columns.tolist()}")
                print(f"      Sample:\n{e.head(2)}")
            except Exception as ex:
                print(f"      Load error: {ex}")

# 3. NFHS-5 birth weight
print(f"\n[3] NFHS-5 birth weight check:")
nfhs5_path = "Dataset/1.NFHS/NHFS 5 (2019-20)/IABR7EDT/IABR7EFL.DTA"
if os.path.exists(nfhs5_path):
    try:
        import pyreadstat

        df5, meta5 = pyreadstat.read_dta(
            nfhs5_path,
            usecols=["m19", "caseid", "b3", "v024"],
            row_limit=30000,
            apply_value_formats=False,
        )
        m = pd.to_numeric(df5["m19"], errors="coerce")
        print(f"    m19 stats: min={m.min():.0f} max={m.max():.0f} mean={m.mean():.0f}")
        print(f"    m19 non-null: {m.notna().sum():,}")
        print(f"    m19 500-9000 (valid g): {((m>=500)&(m<=9000)).sum():,}")
        print(f"    m19 <2500 (LBW): {(m<2500).sum():,}")
        print(
            f"    m19 value dist (top 15): {m.value_counts().sort_index().head(15).to_dict()}"
        )
        # Check b3 (birth year)
        b3 = pd.to_numeric(df5["b3"], errors="coerce")
        birth_yr = 1900 + (b3 - 1) // 12
        print(
            f"    Birth years in sample: {birth_yr.value_counts().sort_index().tail(10).to_dict()}"
        )
        print(f"    v024 (state) in NFHS-5: {df5['v024'].nunique()} unique")
    except Exception as ex:
        print(f"    Error: {ex}")
else:
    print(f"    NOT FOUND: {nfhs5_path}")
    # Search for it
    for root, dirs, files in os.walk("Dataset/1.NFHS"):
        for f in files:
            if "IABR7E" in f:
                print(f"    Found at: {os.path.join(root, f)}")

# 4. State variable check
print(f"\n[4] State variable check:")
df2 = pd.read_parquet(f"{PROCESSED}/final_analytical_dataset.parquet")
for c in ["v024", "v101", "state", "sdistri", "district"]:
    if c in df2.columns:
        vals = pd.to_numeric(df2[c], errors="coerce")
        print(
            f"    {c}: unique={vals.nunique()} min={vals.min()} max={vals.max()} dtype={df2[c].dtype}"
        )

# 5. NHS infrastructure
print(f"\n[5] NHS/infrastructure file:")
nhs_path = f"{PROCESSED}/nhs_combined.csv"
if os.path.exists(nhs_path):
    nhs = pd.read_csv(nhs_path)
    print(f"    Shape: {nhs.shape}")
    print(f"    Cols: {nhs.columns.tolist()[:15]}")
    print(f"    Sample:\n{nhs.head(2)}")

# 6. What FE is feasible
print(f"\n[6] FE feasibility:")
df3 = df2.copy()
df3["district"] = pd.to_numeric(df3["district"], errors="coerce")
df3["v024"] = pd.to_numeric(df3["v024"], errors="coerce")
print(f"    Districts: {df3['district'].nunique()}")
print(f"    States (v024): {df3['v024'].nunique()}")
print(f"    Grid cells (tmax): {df3.groupby(['tmax_lat','tmax_lon']).ngroups}")
print(
    f"    Districts per grid cell: {df3['district'].nunique() / df3.groupby(['tmax_lat','tmax_lon']).ngroups:.1f}"
)
print(f"    → State FE feasible: {df3['v024'].nunique() < 40}")
print(
    f"    → District FE valid: {df3['district'].nunique() <= df3.groupby(['tmax_lat','tmax_lon']).ngroups}"
)
