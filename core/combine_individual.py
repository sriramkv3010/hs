"""
NFHS Individual Recode Combiner — NFHS-4 + NFHS-5
===================================================
Combines IAIR74FL.DTA (NFHS-4) and IAIR7EFL.DTA (NFHS-5)
Extracts only variables relevant to your paper.

KEY VARIABLES EXTRACTED:
  v453  — maternal haemoglobin level (g/dl)
  v456  — haemoglobin adjusted for altitude (g/dl)
  v457  — anaemia level (1=severe,2=moderate,3=mild,4=not anaemic)
  v437  — maternal weight (kg)
  v438  — maternal height (cm)
  v190  — wealth index quintile
  sdistri/sdist — district code

NOTE: District variable has DIFFERENT NAMES across waves:
  NFHS-4: sdistri
  NFHS-5: sdist
  This script handles it automatically.

Run: python3 combine_nfhs_individual.py
"""

import os
import numpy as np
import pandas as pd
import pyreadstat

# ── PATHS ──────────────────────────────────────────────────────────────────────
NFHS4_IR = os.path.expanduser(
    "~/Desktop/HS/Dataset/1.NFHS/NFHS 4 (2015-16)/IAIR74DT/IAIR74FL.DTA"
)
NFHS5_IR = os.path.expanduser(
    "~/Desktop/HS/Dataset/1.NFHS/NHFS 5 (2019-20)/IAIR7EDT/IAIR7EFL.DTA"
)
OUTPUT_DIR = os.path.expanduser("~/Desktop/HS/Dataset/processed/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── VARIABLES TO EXTRACT ───────────────────────────────────────────────────────
# Individual recode has 4,796–5,972 variables
# We extract only ~30 that your paper needs
# Everything else already exists in the births recode

KEEP_VARS = [
    # ── Merge keys ──────────────────────────────────────────────────────────
    "v001",  # cluster number  → matches births recode
    "v002",  # household number → matches births recode
    "v003",  # respondent line number → matches births recode
    # ── Geography ────────────────────────────────────────────────────────────
    "v005",  # sample weight
    "v006",  # interview month
    "v007",  # interview year
    "v021",  # primary sampling unit
    "v022",  # sample strata
    "v024",  # state
    "v025",  # urban/rural
    "sdistri",  # district — NFHS-4 name
    "sdist",  # district — NFHS-5 name
    # ── MATERNAL ANAEMIA — KEY SENSITIVITY VARIABLE ──────────────────────────
    # Anaemia amplifies heat sensitivity during pregnancy
    # Women with Hb < 11 g/dl are clinically anaemic
    "v453",  # haemoglobin level g/dl (1 decimal, divide by 10)
    "v455",  # result of haemoglobin measurement (1=measured)
    "v456",  # haemoglobin adjusted for altitude (preferred measure)
    "v457",  # anaemia level:
    #   1 = severe anaemia (Hb < 7.0)
    #   2 = moderate anaemia (Hb 7.0-9.9)
    #   3 = mild anaemia (Hb 10.0-10.9 non-pregnant / 10.0-11.9 pregnant)
    #   4 = not anaemic
    # ── MATERNAL NUTRITIONAL STATUS ───────────────────────────────────────────
    "v437",  # maternal weight in kg (1 decimal, divide by 10)
    "v438",  # maternal height in cm (1 decimal, divide by 10)
    "v447",  # result of height/weight measurement (1=measured)
    # ── MATERNAL CHARACTERISTICS ──────────────────────────────────────────────
    "v012",  # current age
    "v106",  # education level
    "v190",  # wealth index quintile
    "v191",  # wealth index score
    "v130",  # religion
    "v131",  # caste/tribe
    "v104",  # years lived at current place (migration filter)
    # ── HEALTH INSURANCE ──────────────────────────────────────────────────────
    # Proxy for adaptive capacity — insured women can access care
    "v481a",  # ESIS health insurance
    "v481c",  # state health insurance scheme
    # ── HOUSING (from individual recode — same as household recode) ──────────
    "v127",  # floor material
    "v128",  # wall material
    "v129",  # roof material
    "v119",  # has electricity
    # ── TOTAL BIRTHS (for mother FE) ─────────────────────────────────────────
    "v201",  # total children ever born
]


# ── LOAD FUNCTION ─────────────────────────────────────────────────────────────
def load_individual(path, wave, label):
    print(f"\n{'─'*60}")
    print(f"Loading {label}  (wave={wave})")
    print(f"Path: {path}")
    print(f"NOTE: Files are 4-5GB — this may take 3-5 minutes")

    if not os.path.exists(path):
        raise FileNotFoundError(f"NOT FOUND: {path}")

    size_mb = os.path.getsize(path) / 1e6
    print(f"File size: {size_mb:.1f} MB")

    # Check which variables exist
    _, meta = pyreadstat.read_dta(path, metadataonly=True)
    available = set(meta.column_names)

    load_cols = [v for v in KEEP_VARS if v in available]
    missing = [v for v in KEEP_VARS if v not in available]

    print(f"Variables requested : {len(KEEP_VARS)}")
    print(f"Found in file       : {len(load_cols)}")
    if missing:
        print(f"Not in this file    : {missing}")

    print("Loading data...")
    df, meta = pyreadstat.read_dta(
        path,
        usecols=load_cols,
        apply_value_formats=False,
    )
    df["wave"] = wave
    df["survey_label"] = label
    print(f"Loaded: {len(df):,} women × {df.shape[1]} variables")
    return df, meta


# ── LOAD BOTH WAVES ───────────────────────────────────────────────────────────
print("=" * 60)
print("NFHS INDIVIDUAL RECODE COMBINER — NFHS-4 + NFHS-5")
print("=" * 60)

df4, meta4 = load_individual(NFHS4_IR, wave=4, label="NFHS-4 (2015-16)")
df5, meta5 = load_individual(NFHS5_IR, wave=5, label="NFHS-5 (2019-20)")

# ── STANDARDISE DISTRICT VARIABLE ─────────────────────────────────────────────
# NFHS-4 uses 'sdistri', NFHS-5 uses 'sdist'
# Rename both to 'district' for consistency

print("\n--- Standardising district variable ---")
if "sdistri" in df4.columns:
    df4 = df4.rename(columns={"sdistri": "district"})
    print("  NFHS-4: sdistri → district")
elif "sdist" in df4.columns:
    df4 = df4.rename(columns={"sdist": "district"})
    print("  NFHS-4: sdist → district")

if "sdist" in df5.columns:
    df5 = df5.rename(columns={"sdist": "district"})
    print("  NFHS-5: sdist → district")
elif "sdistri" in df5.columns:
    df5 = df5.rename(columns={"sdistri": "district"})
    print("  NFHS-5: sdistri → district")

# Remove the other district variable if it exists as NaN column
for df in [df4, df5]:
    for col in ["sdistri", "sdist"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

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
print("\n--- Stacking both waves ---")
df = pd.concat([df4, df5], axis=0, ignore_index=True)
print(
    f"Combined: {len(df):,} women  "
    f"(NFHS-4: {(df.wave==4).sum():,}  "
    f"NFHS-5: {(df.wave==5).sum():,})"
)

# ── BUILD MERGE KEY ────────────────────────────────────────────────────────────
# Matches to births recode on v001 + v002 + v003 + wave
df["ir_merge_key"] = (
    df["v001"].astype(str).str.strip()
    + "_"
    + df["v002"].astype(str).str.strip()
    + "_"
    + df["v003"].astype(str).str.strip()
    + "_"
    + df["wave"].astype(str)
)
print(f"Unique women (merge keys): {df['ir_merge_key'].nunique():,}")

# ── CONSTRUCT DERIVED VARIABLES ───────────────────────────────────────────────
print("\n--- Constructing derived variables ---")

# ── Haemoglobin (stored as integer × 10, e.g. 112 = 11.2 g/dl) ───────────────
hb_raw = pd.to_numeric(df["v456"], errors="coerce")  # altitude-adjusted preferred
hb_raw_unadj = pd.to_numeric(df["v453"], errors="coerce")

# DHS codes: 94=not measured, 95=not present, 96=not applicable, 97-99=missing
df["haemoglobin_gdl"] = (hb_raw / 10).where(hb_raw < 900, np.nan)
df["haemoglobin_unadj_gdl"] = (hb_raw_unadj / 10).where(hb_raw_unadj < 900, np.nan)

# Anaemia binary (WHO threshold for pregnant women: < 11.0 g/dl)
df["anaemic"] = (df["haemoglobin_gdl"] < 11.0).astype("Int64")
df.loc[df["haemoglobin_gdl"].isna(), "anaemic"] = pd.NA

# Anaemia severity from v457 (1=severe, 2=moderate, 3=mild, 4=not anaemic)
df["anaemia_level"] = pd.to_numeric(df["v457"], errors="coerce")

# Severe + moderate anaemia combined
df["severe_moderate_anaemia"] = (df["anaemia_level"].isin([1, 2])).astype("Int64")
df.loc[df["anaemia_level"].isna(), "severe_moderate_anaemia"] = pd.NA

print(f"Haemoglobin measured: {df['haemoglobin_gdl'].notna().mean():.1%}")
print(f"Anaemic (<11 g/dl):   {df['anaemic'].mean():.1%}")

# ── Maternal BMI ──────────────────────────────────────────────────────────────
# v437 = weight in kg × 10, v438 = height in cm × 10
# Missing codes: 9996-9999
weight_raw = pd.to_numeric(df["v437"], errors="coerce")
height_raw = pd.to_numeric(df["v438"], errors="coerce")

df["weight_kg"] = (weight_raw / 10).where(weight_raw < 9000, np.nan)
df["height_cm"] = (height_raw / 10).where(height_raw < 9000, np.nan)

# BMI = weight(kg) / (height(m))²
height_m = df["height_cm"] / 100
df["bmi"] = (df["weight_kg"] / (height_m**2)).round(1)
df.loc[df["bmi"] > 60, "bmi"] = np.nan  # implausible values
df.loc[df["bmi"] < 10, "bmi"] = np.nan

# Underweight (BMI < 18.5) — increases heat sensitivity
df["underweight"] = (df["bmi"] < 18.5).astype("Int64")
df.loc[df["bmi"].isna(), "underweight"] = pd.NA

print(f"BMI measured:       {df['bmi'].notna().mean():.1%}")
print(f"BMI mean:           {df['bmi'].mean():.1f}")
print(f"Underweight (<18.5):{df['underweight'].mean():.1%}")

# ── Health insurance ──────────────────────────────────────────────────────────
ins_a = pd.to_numeric(df.get("v481a", pd.Series(dtype=float)), errors="coerce")
ins_c = pd.to_numeric(df.get("v481c", pd.Series(dtype=float)), errors="coerce")
df["has_health_insurance"] = (ins_a.isin([1]) | ins_c.isin([1])).astype("Int64")

# ── Maternal characteristics ──────────────────────────────────────────────────
df["maternal_age"] = pd.to_numeric(df["v012"], errors="coerce")
df["education"] = pd.to_numeric(df["v106"], errors="coerce")
df["wealth_q"] = pd.to_numeric(df["v190"], errors="coerce")
df["wealth_score"] = pd.to_numeric(df["v191"], errors="coerce")
df["rural"] = (pd.to_numeric(df["v025"], errors="coerce") == 2).astype("Int64")
df["religion"] = pd.to_numeric(df["v130"], errors="coerce")
df["caste"] = pd.to_numeric(df["v131"], errors="coerce")
df["long_resident"] = (pd.to_numeric(df["v104"], errors="coerce") >= 5).astype("Int64")
df["total_births"] = pd.to_numeric(df["v201"], errors="coerce")
df["weight"] = pd.to_numeric(df["v005"], errors="coerce") / 1_000_000

# ── QUALITY CHECKS ────────────────────────────────────────────────────────────
print("\n--- Quality Checks ---")
print(f"Shape: {df.shape}")
print(f"Years: {sorted(df['v007'].unique().tolist())}")
print(f"Districts: {df['district'].nunique()}")
print(f"\nMissing data in key variables:")
check_cols = [
    "district",
    "haemoglobin_gdl",
    "anaemic",
    "anaemia_level",
    "bmi",
    "underweight",
    "maternal_age",
    "wealth_q",
    "rural",
]
for col in check_cols:
    if col in df.columns:
        pct = df[col].isna().mean() * 100
        print(f"  {col:30s}: {pct:.1f}%")

print(f"\nAnaemia distribution (v457):")
print(
    df["anaemia_level"]
    .value_counts()
    .sort_index()
    .rename({1: "1=severe", 2: "2=moderate", 3: "3=mild", 4: "4=not anaemic"})
)

print(f"\nBMI distribution:")
print(df["bmi"].describe().round(1))

print(f"\nWave distribution:")
print(df["wave"].value_counts())

# ── FINAL COLUMN ORDER ────────────────────────────────────────────────────────
FINAL_COLS = [
    # Identifiers
    "wave",
    "survey_label",
    "ir_merge_key",
    "v001",
    "v002",
    "v003",
    "v005",
    "v006",
    "v007",
    "v021",
    "v022",
    "v024",
    "v025",
    "district",
    "weight",
    # Anaemia — KEY FOR YOUR PAPER
    "haemoglobin_gdl",
    "haemoglobin_unadj_gdl",
    "anaemic",
    "anaemia_level",
    "severe_moderate_anaemia",
    # Nutritional status
    "weight_kg",
    "height_cm",
    "bmi",
    "underweight",
    # Health insurance
    "has_health_insurance",
    # Maternal characteristics
    "maternal_age",
    "education",
    "wealth_q",
    "wealth_score",
    "rural",
    "religion",
    "caste",
    "long_resident",
    "total_births",
    # Housing (from individual recode)
    "v127",
    "v128",
    "v129",
    "v119",
    # Raw DHS variables
    "v012",
    "v106",
    "v190",
    "v191",
    "v104",
    "v130",
    "v131",
    "v201",
    "v437",
    "v438",
    "v447",
    "v453",
    "v455",
    "v456",
    "v457",
]

final_cols = [c for c in FINAL_COLS if c in df.columns]
df_out = df[final_cols].reset_index(drop=True)

# ── SAVE ──────────────────────────────────────────────────────────────────────
print("\n--- Saving ---")

csv_path = os.path.join(OUTPUT_DIR, "nfhs_individual_combined.csv")
df_out.to_csv(csv_path, index=False)
print(f"CSV:     {csv_path}")
print(f"  Rows:  {len(df_out):,}")
print(f"  Size:  {os.path.getsize(csv_path)/1e6:.1f} MB")

parquet_path = os.path.join(OUTPUT_DIR, "nfhs_individual_combined.parquet")
df_out.to_parquet(parquet_path, index=False)
print(f"Parquet: {parquet_path}")
print(f"  Size:  {os.path.getsize(parquet_path)/1e6:.1f} MB")

# Variable labels reference
labels = {}
for col in df_out.columns:
    l4 = meta4.column_names_to_labels.get(col, "")
    l5 = meta5.column_names_to_labels.get(col, "")
    labels[col] = l4 or l5 or col
pd.DataFrame(list(labels.items()), columns=["variable", "label"]).to_csv(
    os.path.join(OUTPUT_DIR, "nfhs_individual_variable_labels.csv"), index=False
)

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Total women:  {len(df_out):,}")
print(f"  NFHS-4:     {(df_out.wave==4).sum():,}")
print(f"  NFHS-5:     {(df_out.wave==5).sum():,}")
print(f"Columns:      {len(df_out.columns)}")
print(f"Districts:    {df_out['district'].nunique()}")
print(
    f"""
HOW TO MERGE WITH BIRTHS RECODE:

  births = pd.read_parquet('nfhs_births_combined.parquet')
  indiv  = pd.read_parquet('nfhs_individual_combined.parquet')

  # Build merge key in births file (v001 + v002 + v003 + wave)
  births['ir_merge_key'] = (
      births['v001'].astype(str) + '_' +
      births['v002'].astype(str) + '_' +
      births['v003'].astype(str) + '_' +
      births['wave'].astype(str)
  )

  # Merge — each birth gets its mother's anaemia and BMI
  merged = births.merge(
      indiv[['ir_merge_key', 'haemoglobin_gdl', 'anaemic',
             'anaemia_level', 'bmi', 'underweight',
             'has_health_insurance']],
      on='ir_merge_key',
      how='left'
  )

  print(f"Merge rate: {{merged['anaemic'].notna().mean():.1%}}")

KEY VARIABLES FOR YOUR PAPER:
  anaemic              — binary, Hb < 11 g/dl
  anaemia_level        — 1=severe, 2=moderate, 3=mild, 4=not
  haemoglobin_gdl      — continuous Hb level (g/dl)
  bmi                  — maternal BMI
  underweight          — binary, BMI < 18.5
  has_health_insurance — binary, adaptive capacity proxy

WHY ANAEMIA MATTERS FOR YOUR PAPER:
  Anaemic women have reduced oxygen-carrying capacity.
  Under heat stress, this amplifies physiological burden —
  the body cannot compensate for heat-induced cardiovascular
  strain as effectively. Anaemia is therefore a direct
  measure of 'sensitivity' in your vulnerability framework
  (Sensitivity = how much a system is affected by a stressor).

  In your heterogeneity analysis (Section 6.2), you can show:
  Heat effect on LBW is larger for anaemic mothers than
  non-anaemic mothers — confirming the vulnerability framework.
"""
)
