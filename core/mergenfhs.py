"""
Master Merge: Births + Individual + Household
==============================================
Combines three NFHS parquet files into one analytical dataset.

Merge logic:
  births + individual  →  on v001+v002+v003+wave  (mother-level)
  result  + household  →  on v001+v002+wave        (household-level)

Output:
  nfhs_merged.parquet  (~490k births with all variables)
  nfhs_merged.csv

Run: python3 merge_nfhs.py
"""

import os
import pandas as pd
import numpy as np

# ── PATHS ──────────────────────────────────────────────────────────────────────
PROCESSED = os.path.expanduser("~/Desktop/HS/Dataset/processed")

BIRTHS_PATH = os.path.join(PROCESSED, "nfhs_births_combined.parquet")
INDIV_PATH = os.path.join(PROCESSED, "nfhs_individual_combined.parquet")
HH_PATH = os.path.join(PROCESSED, "nfhs_household_combined.parquet")
OUTPUT_DIR = PROCESSED

# ── LOAD ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("NFHS MASTER MERGE: BIRTHS + INDIVIDUAL + HOUSEHOLD")
print("=" * 60)

print("\nLoading births...")
births = pd.read_parquet(BIRTHS_PATH)
print(f"  Shape: {births.shape}")

print("Loading individual...")
indiv = pd.read_parquet(INDIV_PATH)
print(f"  Shape: {indiv.shape}")

print("Loading household...")
hh = pd.read_parquet(HH_PATH)
print(f"  Shape: {hh.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# MERGE 1: BIRTHS + INDIVIDUAL
# Key: v001 (cluster) + v002 (household) + v003 (respondent line) + wave
# One woman can have multiple births → left join keeps all births
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("MERGE 1: births + individual recode")
print("─" * 60)

# Build merge key in births
births["ir_merge_key"] = (
    births["v001"].astype(str).str.strip()
    + "_"
    + births["v002"].astype(str).str.strip()
    + "_"
    + births["v003"].astype(str).str.strip()
    + "_"
    + births["wave"].astype(str)
)

# Columns to bring from individual recode
# EXCLUDE columns already in births to avoid duplicates
births_cols = set(births.columns)
indiv_keep = [
    "ir_merge_key",
    # Anaemia — KEY variables not in births
    "haemoglobin_gdl",
    "haemoglobin_unadj_gdl",
    "anaemic",
    "anaemia_level",
    "severe_moderate_anaemia",
    # Nutritional status — not in births
    "weight_kg",
    "height_cm",
    "bmi",
    "underweight",
    # Adaptive capacity — not in births
    "has_health_insurance",
    # District from individual recode (cross-check)
    "district",
]
# Only keep columns that exist and are not already in births
indiv_keep = [
    c
    for c in indiv_keep
    if c in indiv.columns and (c == "ir_merge_key" or c not in births_cols)
]

# Add district as district_ir for cross-check
if "district" in indiv.columns:
    indiv = indiv.rename(columns={"district": "district_ir"})
    if "district_ir" not in indiv_keep:
        indiv_keep = [c if c != "district" else "district_ir" for c in indiv_keep]
        if "district_ir" not in indiv_keep:
            indiv_keep.append("district_ir")

indiv_subset = indiv[indiv_keep].copy()

print(f"Individual columns to merge: {[c for c in indiv_keep if c != 'ir_merge_key']}")

# Check for duplicate keys in individual recode
n_dupes = indiv_subset.duplicated("ir_merge_key").sum()
print(f"Duplicate ir_merge_keys in individual: {n_dupes:,}")
if n_dupes > 0:
    # Keep first occurrence (most recent birth info)
    indiv_subset = indiv_subset.drop_duplicates("ir_merge_key", keep="first")
    print(f"  Deduplicated to: {len(indiv_subset):,} rows")

# Left join — keep all births, add individual data where available
n_before = len(births)
df = births.merge(indiv_subset, on="ir_merge_key", how="left")
n_after = len(df)

print(f"\nBirths before merge: {n_before:,}")
print(f"Births after merge:  {n_after:,}")
assert n_before == n_after, "ROW COUNT CHANGED — merge logic error"
print("Row count preserved ✓")

# Check merge rate
merge_rate = df["anaemic"].notna().mean()
print(f"Anaemia merge rate:  {merge_rate:.1%}")
print(f"  (Expected ~40-60% — anaemia only measured for subsample)")

# ══════════════════════════════════════════════════════════════════════════════
# MERGE 2: RESULT + HOUSEHOLD
# Key: v001 (cluster) + v002 (household) + wave
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("MERGE 2: result + household recode")
print("─" * 60)

# Build household merge key in main dataframe
df["hh_merge_key"] = (
    df["v001"].astype(str).str.strip()
    + "_"
    + df["v002"].astype(str).str.strip()
    + "_"
    + df["wave"].astype(str)
)

# Columns to bring from household recode
current_cols = set(df.columns)
hh_keep = [
    "hh_merge_key",
    # Housing quality — heat adaptation proxies
    "housing_quality_score",
    "good_housing",
    "floor_quality",
    "wall_quality",
    "roof_quality",
    # Utilities
    "has_electricity",
    "clean_fuel",
    # Household composition
    "hv009",  # number of household members
    "hv014",  # number of children under 5
    # Raw material codes (for robustness)
    "hv213",
    "hv214",
    "hv215",
    "hv206",
    "hv226",
]
# Only keep columns that exist and aren't already in main df
hh_keep = [
    c
    for c in hh_keep
    if c in hh.columns and (c == "hh_merge_key" or c not in current_cols)
]

hh_subset = hh[hh_keep].copy()
print(f"Household columns to merge: {[c for c in hh_keep if c != 'hh_merge_key']}")

# Deduplicate household key
n_dupes_hh = hh_subset.duplicated("hh_merge_key").sum()
print(f"Duplicate hh_merge_keys: {n_dupes_hh:,}")
if n_dupes_hh > 0:
    hh_subset = hh_subset.drop_duplicates("hh_merge_key", keep="first")

# Left join
n_before = len(df)
df = df.merge(hh_subset, on="hh_merge_key", how="left")
n_after = len(df)

print(f"\nRows before merge: {n_before:,}")
print(f"Rows after merge:  {n_after:,}")
assert n_before == n_after, "ROW COUNT CHANGED — merge logic error"
print("Row count preserved ✓")

hh_rate = (
    df["housing_quality_score"].notna().mean()
    if "housing_quality_score" in df.columns
    else 0
)
print(f"Housing quality merge rate: {hh_rate:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# POST-MERGE CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("POST-MERGE CLEANING")
print("─" * 60)

# Resolve district — use births district as primary, individual as backup
if "district_ir" in df.columns:
    df["district"] = df["district"].fillna(df["district_ir"])
    missing_before = df["district"].isna().sum()
    print(
        f"District filled from individual recode: "
        f"{(df['district_ir'].notna() & df['district'].notna()).sum():,}"
    )
    df = df.drop(columns=["district_ir"])

# Flag births with complete data for analysis
df["has_outcome"] = df["lbw"].notna().astype("Int64")
df["has_district"] = df["district"].notna().astype("Int64")
df["has_climate_window"] = (
    df["t3_start_cmc"].notna() & df["t3_end_cmc"].notna()
).astype("Int64")

# Analysis-ready flag — all essential variables present
df["analysis_ready"] = (
    df["has_outcome"].eq(1)
    & df["has_district"].eq(1)
    & df["birth_year"].between(2009, 2021)
).astype("Int64")

print(f"\nData completeness summary:")
print(f"  Total births:           {len(df):,}")
print(f"  Has LBW outcome:        {df['has_outcome'].mean():.1%}")
print(f"  Has district:           {df['has_district'].mean():.1%}")
print(f"  Has trimester window:   {df['has_climate_window'].mean():.1%}")
print(f"  Analysis ready:         {df['analysis_ready'].mean():.1%}")
print(f"  Analysis ready (count): {df['analysis_ready'].sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL COLUMN ORDER
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("ORGANISING COLUMNS")
print("─" * 60)

# Logical grouping for readability
COL_ORDER = [
    # ── Identifiers ──────────────────────────────────────────────────────
    "wave",
    "survey_label",
    "caseid",
    "mother_id",
    "ir_merge_key",
    "hh_merge_key",
    "v001",
    "v002",
    "v003",
    "v021",
    "v022",
    "v023",
    "v024",
    "v025",
    "sdistri",
    "district",
    "weight",
    # ── Analysis flags ────────────────────────────────────────────────────
    "analysis_ready",
    "has_outcome",
    "has_district",
    "has_climate_window",
    # ── PRIMARY OUTCOMES ─────────────────────────────────────────────────
    "lbw",
    "vlbw",
    "preterm",
    "neonatal_death",
    "postneonatal_death",
    "infant_death",
    "birthweight_kg",
    "birthweight_g",
    "size_perceived",
    "has_bw",
    # ── BIRTH DATE + TRIMESTER WINDOWS ───────────────────────────────────
    "b3",
    "b3_num",
    "birth_year",
    "birth_month",
    "preg_months",
    "preg_valid",
    "conception_cmc",
    "conception_year",
    "conception_month",
    "t1_start_cmc",
    "t1_end_cmc",
    "t2_start_cmc",
    "t2_end_cmc",
    "t3_start_cmc",
    "t3_end_cmc",
    "t1_start_year",
    "t1_start_month",
    "t2_start_year",
    "t2_start_month",
    "t3_start_year",
    "t3_start_month",
    "t3_end_year",
    "t3_end_month",
    "t1_start_cmc_wide",
    "t1_end_cmc_wide",
    "t3_start_cmc_wide",
    "t3_end_cmc_wide",
    # ── MATERNAL CHARACTERISTICS (controls) ──────────────────────────────
    "maternal_age",
    "education",
    "wealth_q",
    "wealth_score",
    "rural",
    "birth_order",
    "anc_visits",
    "institutional",
    "caesarean",
    "religion",
    "caste",
    "multi_birth_mother",
    "long_resident",
    # ── MATERNAL HEALTH (from individual recode) ──────────────────────────
    "haemoglobin_gdl",
    "haemoglobin_unadj_gdl",
    "anaemic",
    "anaemia_level",
    "severe_moderate_anaemia",
    "weight_kg",
    "height_cm",
    "bmi",
    "underweight",
    "has_health_insurance",
    # ── HOUSING / ADAPTIVE CAPACITY (from household recode) ───────────────
    "housing_quality_score",
    "good_housing",
    "floor_quality",
    "wall_quality",
    "roof_quality",
    "has_electricity",
    "clean_fuel",
    "hv009",
    "hv014",
    "hv213",
    "hv214",
    "hv215",
    "hv206",
    "hv226",
    # ── RAW DHS VARIABLES (keep for reference) ────────────────────────────
    "b0",
    "b4",
    "b5",
    "b6",
    "b7",
    "b8",
    "b11",
    "b12",
    "bord",
    "m14",
    "m15",
    "m17",
    "m18",
    "m19",
    "s220a",
    "v104",
    "v105",
    "v106",
    "v130",
    "v131",
    "v190",
    "v191",
    "v201",
    "b1",
    "b2",
]

# Keep only columns that exist
final_cols = [c for c in COL_ORDER if c in df.columns]
# Add any remaining columns not in our order
remaining = [c for c in df.columns if c not in final_cols]
if remaining:
    print(f"Additional columns added at end: {remaining}")
final_cols = final_cols + remaining

df = df[final_cols]
print(f"Final column count: {len(df.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("QUALITY CHECKS")
print("─" * 60)

print(f"\nFinal shape: {df.shape}")
print(f"\nOutcome rates:")
print(f"  LBW:              {df['lbw'].mean():.3f}")
print(f"  Preterm:          {df['preterm'].mean():.3f}")
print(f"  Neonatal mort:    {df['neonatal_death'].mean():.4f}")
print(f"  Infant mort:      {df['infant_death'].mean():.4f}")

print(f"\nKey variable coverage:")
key_vars = [
    "lbw",
    "district",
    "wealth_q",
    "maternal_age",
    "anaemic",
    "bmi",
    "housing_quality_score",
    "has_electricity",
    "t3_start_cmc",
    "preg_months",
]
for col in key_vars:
    if col in df.columns:
        pct_miss = df[col].isna().mean() * 100
        pct_have = 100 - pct_miss
        bar = "█" * int(pct_have / 5)
        print(f"  {col:28s}: {pct_have:5.1f}% complete  {bar}")

print(f"\nBy wave:")
for w, lbl in [(4, "NFHS-4"), (5, "NFHS-5")]:
    sub = df[df.wave == w]
    print(
        f"  {lbl}: {len(sub):,} births  "
        f"LBW={sub['lbw'].mean():.3f}  "
        f"District={sub['district'].notna().mean():.1%}"
    )

print(f"\nDistricts: {df['district'].nunique()}")
print(f"Birth years: {df['birth_year'].min()} – {df['birth_year'].max()}")
print(f"\nWealth quintile distribution:")
print(
    df["wealth_q"]
    .value_counts()
    .sort_index()
    .rename({1: "Q1 poorest", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 richest"})
)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("SAVING")
print("─" * 60)

parquet_path = os.path.join(OUTPUT_DIR, "nfhs_merged.parquet")
df.to_parquet(parquet_path, index=False, compression="snappy")
print(f"Parquet: {parquet_path}")
print(f"  Size:  {os.path.getsize(parquet_path)/1e6:.1f} MB")

csv_path = os.path.join(OUTPUT_DIR, "nfhs_merged.csv")
df.to_csv(csv_path, index=False)
print(f"CSV:     {csv_path}")
print(f"  Size:  {os.path.getsize(csv_path)/1e6:.1f} MB")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DONE — NFHS MERGED DATASET")
print("=" * 60)
print(f"Total births:    {len(df):,}")
print(f"Total columns:   {len(df.columns)}")
print(f"Analysis ready:  {df['analysis_ready'].sum():,} births")
print(
    f"""
WHAT WAS MERGED:
  births     → outcomes, trimester windows, maternal controls
  individual → anaemia, BMI, haemoglobin
  household  → housing quality, electricity, cooking fuel

NEXT STEP — Climate Merge (Notebook 06):
  Load:   df = pd.read_parquet('nfhs_merged.parquet')
  Add:    T1/T2/T3 temperature anomaly per birth
          from imd_tmax_combined.parquet
  Method: district → nearest IMD grid cell → average over trimester months

COLUMN GROUPS:
  Cols 1-17:   Identifiers + geography
  Cols 18-21:  Analysis flags
  Cols 22-31:  Primary outcomes (LBW, preterm, mortality)
  Cols 32-51:  Birth dates + trimester CMC windows
  Cols 52-66:  Maternal controls
  Cols 67-75:  Maternal health (anaemia, BMI)
  Cols 76-86:  Housing quality + adaptive capacity
  Cols 87+:    Raw DHS variables
"""
)
