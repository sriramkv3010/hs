"""
NFHS Household Recode Combiner — NFHS-4 + NFHS-5
==================================================
Combines IAHR74FL.DTA (NFHS-4) and IAHR7EFL.DTA (NFHS-5)
Extracts only variables relevant to your paper.

Key variables extracted:
  hv213  — floor material  (housing quality / heat adaptation)
  hv214  — wall material   (housing quality / heat adaptation)
  hv215  — roof material   (housing quality / heat adaptation)
  hv206  — has electricity (cooling access)
  hv226  — cooking fuel    (indoor heat + pollution)
  hv270  — wealth index    (cross-check with births recode)
  hv271  — wealth score
  + geography merge keys

Run: python3 combine_nfhs_household.py
"""

import os
import numpy as np
import pandas as pd
import pyreadstat

# ── PATHS ──────────────────────────────────────────────────────────────────────
NFHS4_HH = os.path.expanduser(
    "~/Desktop/HS/Dataset/1.NFHS/NFHS 4 (2015-16)/IAHR74DT/IAHR74FL.DTA"
)
NFHS5_HH = os.path.expanduser(
    "~/Desktop/HS/Dataset/1.NFHS/NHFS 5 (2019-20)/IAHR7EDT/IAHR7EFL.DTA"
)
OUTPUT_DIR = os.path.expanduser("~/Desktop/HS/Dataset/processed/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── VARIABLES TO EXTRACT ───────────────────────────────────────────────────────
# Only what your paper needs — household recode has 5,265 variables
# We keep ~20 relevant ones
KEEP_VARS = [
    # ── Merge keys (match to births recode) ────────────────────────────────
    "hv001",  # cluster number  → matches v001 in births recode
    "hv002",  # household number → matches v002 in births recode
    # ── Geography ───────────────────────────────────────────────────────────
    "hv005",  # household sample weight
    "hv006",  # month of interview
    "hv007",  # year of interview
    "hv021",  # primary sampling unit
    "hv022",  # sample strata
    "hv024",  # state
    "hv025",  # urban/rural
    # ── Housing quality — heat adaptation proxies ────────────────────────
    # These are critical for your paper:
    # Poor housing (mud walls, thatched roof) = higher heat exposure
    # Good housing (brick walls, concrete roof) = natural cooling buffer
    "hv213",  # main floor material
    "hv214",  # main wall material
    "hv215",  # main roof material
    "hv216",  # number of rooms for sleeping (overcrowding)
    # ── Utilities / cooling access ───────────────────────────────────────
    "hv206",  # has electricity (needed for fans/AC)
    "hv207",  # has radio
    "hv208",  # has television
    "hv209",  # has refrigerator
    "hv210",  # has bicycle
    "hv211",  # has motorcycle/scooter
    "hv212",  # has car/truck
    "hv221",  # has telephone (landline)
    "hv226",  # type of cooking fuel (biomass = indoor heat stress)
    # ── Water and sanitation ─────────────────────────────────────────────
    "hv201",  # source of drinking water
    "hv204",  # time to water source (minutes)
    "hv205",  # type of toilet facility
    # ── Household composition ─────────────────────────────────────────────
    "hv009",  # number of household members
    "hv014",  # number of children under 5
    # ── Wealth index (cross-check with births recode v190) ───────────────
    "hv270",  # wealth index quintile (household recode version)
    "hv271",  # wealth index score (continuous)
]


# ── LOAD FUNCTION ─────────────────────────────────────────────────────────────
def load_household(path, wave, label):
    print(f"\n{'─'*60}")
    print(f"Loading {label}  (wave={wave})")
    print(f"Path: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"NOT FOUND: {path}")

    size_mb = os.path.getsize(path) / 1e6
    print(f"Size: {size_mb:.1f} MB")

    # Check which variables exist in this file
    _, meta = pyreadstat.read_dta(path, metadataonly=True)
    available = set(meta.column_names)
    load_cols = [v for v in KEEP_VARS if v in available]
    missing = [v for v in KEEP_VARS if v not in available]

    print(f"Variables requested : {len(KEEP_VARS)}")
    print(f"Found in file       : {len(load_cols)}")
    if missing:
        print(f"Not in this file    : {missing}")

    print("Loading... (may take 1-2 minutes)")
    df, meta = pyreadstat.read_dta(
        path,
        usecols=load_cols,
        apply_value_formats=False,
    )
    df["wave"] = wave
    df["survey_label"] = label
    print(f"Loaded: {len(df):,} households × {df.shape[1]} variables")
    return df, meta


# ── LOAD BOTH ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("NFHS HOUSEHOLD RECODE COMBINER — NFHS-4 + NFHS-5")
print("=" * 60)

df4, meta4 = load_household(NFHS4_HH, wave=4, label="NFHS-4 (2015-16)")
df5, meta5 = load_household(NFHS5_HH, wave=5, label="NFHS-5 (2019-20)")

# ── HARMONISE COLUMNS ─────────────────────────────────────────────────────────
print("\n--- Harmonising columns ---")
only_in_4 = set(df4.columns) - set(df5.columns)
only_in_5 = set(df5.columns) - set(df4.columns)
in_both = set(df4.columns) & set(df5.columns)
print(f"In both waves : {len(in_both)}")
print(f"Only NFHS-4   : {sorted(only_in_4)}")
print(f"Only NFHS-5   : {sorted(only_in_5)}")

for col in only_in_5:
    df4[col] = np.nan
for col in only_in_4:
    df5[col] = np.nan

df4 = df4[sorted(df4.columns)]
df5 = df5[sorted(df5.columns)]

# ── STACK ─────────────────────────────────────────────────────────────────────
print("\n--- Stacking ---")
df = pd.concat([df4, df5], axis=0, ignore_index=True)
print(
    f"Combined: {len(df):,} households  "
    f"(NFHS-4: {(df.wave==4).sum():,}  "
    f"NFHS-5: {(df.wave==5).sum():,})"
)

# ── BUILD MERGE KEY ────────────────────────────────────────────────────────────
# This key matches to births recode: v001 + v002 + wave
df["hh_merge_key"] = (
    df["hv001"].astype(str).str.strip()
    + "_"
    + df["hv002"].astype(str).str.strip()
    + "_"
    + df["wave"].astype(str)
)
print(f"Merge key built: hv001_hv002_wave")
print(f"Unique households: {df['hh_merge_key'].nunique():,}")

# ── CONSTRUCT DERIVED VARIABLES ───────────────────────────────────────────────
print("\n--- Constructing derived variables ---")

# Housing quality index — for heat adaptation analysis
# Better materials = more protection from external heat
# hv213 floor: 10-19=earth/sand(worst), 20-29=rudimentary, 30-39=finished(best)
# hv214 wall:  10-19=no wall/natural, 20-29=rudimentary, 30-39=finished(best)
# hv215 roof:  10-19=no roof/natural, 20-29=rudimentary, 30-39=finished(best)


def housing_quality(series):
    """Convert DHS material code to quality tier: 1=poor, 2=medium, 3=good"""
    s = pd.to_numeric(series, errors="coerce")
    return pd.cut(s, bins=[0, 19, 29, 100], labels=[1, 2, 3], right=True).astype(
        "Int64"
    )


if "hv213" in df.columns:
    df["floor_quality"] = housing_quality(df["hv213"])
if "hv214" in df.columns:
    df["wall_quality"] = housing_quality(df["hv214"])
if "hv215" in df.columns:
    df["roof_quality"] = housing_quality(df["hv215"])

# Overall housing quality score (1-9, higher = better)
quality_cols = [
    c for c in ["floor_quality", "wall_quality", "roof_quality"] if c in df.columns
]
if quality_cols:
    df["housing_quality_score"] = df[quality_cols].sum(axis=1, skipna=False)
    df["good_housing"] = (df["housing_quality_score"] >= 7).astype("Int64")
    print(f"Good housing (score>=7): {df['good_housing'].mean():.1%}")

# Electricity access
if "hv206" in df.columns:
    df["has_electricity"] = (pd.to_numeric(df["hv206"], errors="coerce") == 1).astype(
        "Int64"
    )
    print(f"Has electricity: {df['has_electricity'].mean():.1%}")

# Clean cooking fuel (1=electricity/gas, 0=biomass/coal)
# hv226: 1=electricity, 2=LPG, 3=natural gas → clean
#        6=charcoal, 7=wood, 8=straw, 9=animal dung → dirty
if "hv226" in df.columns:
    fuel = pd.to_numeric(df["hv226"], errors="coerce")
    df["clean_fuel"] = fuel.isin([1, 2, 3]).astype("Int64")
    print(f"Clean cooking fuel: {df['clean_fuel'].mean():.1%}")

# Household wealth (from household recode — cross check)
if "hv270" in df.columns:
    df["hh_wealth_q"] = pd.to_numeric(df["hv270"], errors="coerce")

# ── QUALITY CHECKS ────────────────────────────────────────────────────────────
print("\n--- Quality Checks ---")
print(f"Shape: {df.shape}")
print(
    f"Years: {sorted(df['hv007'].unique().tolist()) if 'hv007' in df.columns else 'N/A'}"
)

if "hv025" in df.columns:
    rural = (pd.to_numeric(df["hv025"], errors="coerce") == 2).mean()
    print(f"Rural households: {rural:.1%}")

if "hv213" in df.columns:
    print(f"\nFloor material distribution (hv213):")
    print(
        pd.to_numeric(df["hv213"], errors="coerce").value_counts().sort_index().head(10)
    )

if "housing_quality_score" in df.columns:
    print(f"\nHousing quality score distribution:")
    print(df["housing_quality_score"].value_counts().sort_index())

print(f"\nMissing data in key variables:")
check = [
    "hv001",
    "hv002",
    "hv213",
    "hv214",
    "hv215",
    "hv206",
    "hv226",
    "has_electricity",
    "clean_fuel",
    "housing_quality_score",
]
for col in check:
    if col in df.columns:
        pct = df[col].isna().mean() * 100
        print(f"  {col:30s}: {pct:.1f}%")

# ── FINAL COLUMN ORDER ────────────────────────────────────────────────────────
FINAL_COLS = [
    # Identifiers and merge keys
    "wave",
    "survey_label",
    "hh_merge_key",
    "hv001",
    "hv002",
    "hv005",
    "hv006",
    "hv007",
    "hv021",
    "hv022",
    "hv024",
    "hv025",
    # Housing quality — derived
    "floor_quality",
    "wall_quality",
    "roof_quality",
    "housing_quality_score",
    "good_housing",
    # Utilities
    "has_electricity",
    "clean_fuel",
    # Raw housing variables
    "hv213",
    "hv214",
    "hv215",
    "hv216",
    "hv206",
    "hv226",
    # Water/sanitation
    "hv201",
    "hv204",
    "hv205",
    # Assets
    "hv207",
    "hv208",
    "hv209",
    "hv210",
    "hv211",
    "hv212",
    "hv221",
    # Household composition
    "hv009",
    "hv014",
    # Wealth
    "hh_wealth_q",
    "hv271",
]

final_cols = [c for c in FINAL_COLS if c in df.columns]
df_out = df[final_cols].reset_index(drop=True)

# ── SAVE ──────────────────────────────────────────────────────────────────────
print("\n--- Saving ---")

csv_path = os.path.join(OUTPUT_DIR, "nfhs_household_combined.csv")
df_out.to_csv(csv_path, index=False)
print(f"CSV:     {csv_path}")
print(f"  Rows:  {len(df_out):,}")
print(f"  Size:  {os.path.getsize(csv_path)/1e6:.1f} MB")

parquet_path = os.path.join(OUTPUT_DIR, "nfhs_household_combined.parquet")
df_out.to_parquet(parquet_path, index=False)
print(f"Parquet: {parquet_path}")
print(f"  Size:  {os.path.getsize(parquet_path)/1e6:.1f} MB")

# Save variable labels
labels = {}
for col in df_out.columns:
    l4 = meta4.column_names_to_labels.get(col, "")
    l5 = meta5.column_names_to_labels.get(col, "")
    labels[col] = l4 or l5 or col

pd.DataFrame(list(labels.items()), columns=["variable", "label"]).to_csv(
    os.path.join(OUTPUT_DIR, "nfhs_household_variable_labels.csv"), index=False
)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Total households: {len(df_out):,}")
print(f"  NFHS-4: {(df_out.wave==4).sum():,}")
print(f"  NFHS-5: {(df_out.wave==5).sum():,}")
print(f"Columns: {len(df_out.columns)}")
print(
    f"""
HOW TO MERGE WITH BIRTHS RECODE:
  births = pd.read_parquet('nfhs_births_combined.parquet')
  hh     = pd.read_parquet('nfhs_household_combined.parquet')

  # Build merge key in births file
  births['hh_merge_key'] = (
      births['v001'].astype(str) + '_' +
      births['v002'].astype(str) + '_' +
      births['wave'].astype(str)
  )

  # Merge
  merged = births.merge(
      hh[['hh_merge_key','housing_quality_score',
          'good_housing','has_electricity','clean_fuel',
          'hv213','hv214','hv215','hv206','hv226']],
      on='hh_merge_key',
      how='left'
  )

KEY VARIABLES FOR YOUR PAPER:
  housing_quality_score  — 3-9, higher = better heat protection
  good_housing           — binary, score >= 7
  has_electricity        — binary, proxy for fan/AC access
  clean_fuel             — binary, proxy for indoor thermal burden
  hv213/214/215          — raw floor/wall/roof codes
"""
)
