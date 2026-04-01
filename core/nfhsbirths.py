"""
NFHS Births Combiner — NFHS-4 + NFHS-5 (CORRECTED)
=====================================================
Combines IABR74FL.DTA (NFHS-4) and IABR7EFL.DTA (NFHS-5)

Run: python3 combine_nfhs_births.py

Requirements: pip install pyreadstat pandas numpy
"""

import os
import pandas as pd
import numpy as np
import pyreadstat

# ── PATHS ──────────────────────────────────────────────────────────────────────
NFHS4 = os.path.expanduser(
    "~/Desktop/HS/Dataset/1.NFHS/NFHS 4 (2015-16)/IABR74DT/IABR74FL.DTA"
)
NFHS5 = os.path.expanduser(
    "~/Desktop/HS/Dataset/1.NFHS/NHFS 5 (2019-20)/IABR7EDT/IABR7EFL.DTA"
)
OUTPUT_DIR = os.path.expanduser("~/Desktop/HS/Dataset/processed/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── VARIABLES TO LOAD ──────────────────────────────────────────────────────────
KEEP_VARS = [
    # Identifiers / geography
    "caseid",
    "v001",
    "v002",
    "v003",
    "v005",
    "v006",
    "v007",
    "v021",
    "v022",
    "v023",
    "v024",
    "v025",
    "v101",
    "v102",
    "sdistri",
    # Birth history
    "b0",
    "b1",
    "b2",
    "b3",
    "b4",
    "b5",
    "b6",
    "b7",
    "b8",
    "b11",
    "b12",
    "b16",
    "bord",
    # Maternity — outcomes and controls
    "m14",
    "m15",
    "m17",
    "m18",
    "m19",
    # Pregnancy duration — KEY for trimester construction
    "s220a",
    # Maternal characteristics
    "v012",
    "v106",
    "v130",
    "v131",
    "v190",
    "v191",
    # Residence
    "v104",
    "v105",
    # Fertility
    "v201",
    "v202",
    "v203",
]


# ── HELPER: CMC TO YEAR + MONTH ────────────────────────────────────────────────
def cmc_to_ym(cmc_series):
    """Century Month Code → (year, month)"""
    cmc = pd.to_numeric(cmc_series, errors="coerce")
    year = (1900 + ((cmc - 1) // 12)).astype("Int64")
    mon = (cmc - (year - 1900) * 12).astype("Int64")
    return year, mon


# ── HELPER: LOAD ONE DTA FILE ──────────────────────────────────────────────────
def load_births(path, wave, survey_label):
    print(f"\n{'─'*60}")
    print(f"Loading {survey_label}  (wave={wave})")
    print(f"Path: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")

    size_mb = os.path.getsize(path) / 1e6
    print(f"File size: {size_mb:.1f} MB")

    # Check which variables actually exist
    _, meta = pyreadstat.read_dta(path, metadataonly=True)
    available = set(meta.column_names)
    load_cols = [v for v in KEEP_VARS if v in available]
    missing = [v for v in KEEP_VARS if v not in available]

    print(f"Variables requested : {len(KEEP_VARS)}")
    print(f"Found in file       : {len(load_cols)}")
    if missing:
        print(f"Not found (skip)    : {missing}")

    # Load
    print("Loading data (may take 1-3 minutes)...")
    df, meta = pyreadstat.read_dta(
        path,
        usecols=load_cols,
        apply_value_formats=False,  # numeric codes only
    )
    df["wave"] = wave
    df["survey_label"] = survey_label
    print(f"Loaded: {len(df):,} births × {df.shape[1]} columns")
    return df, meta


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("NFHS BIRTHS COMBINER — NFHS-4 + NFHS-5")
print("=" * 60)

df4, meta4 = load_births(NFHS4, wave=4, survey_label="NFHS-4 (2015-16)")
df5, meta5 = load_births(NFHS5, wave=5, survey_label="NFHS-5 (2019-20)")

# ── HARMONISE COLUMNS ─────────────────────────────────────────────────────────
print("\n--- Harmonising columns ---")
all_cols = sorted(set(df4.columns) | set(df5.columns))
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
    f"Combined: {len(df):,} births  "
    f"(NFHS-4: {(df.wave==4).sum():,}  "
    f"NFHS-5: {(df.wave==5).sum():,})"
)

# ── BIRTH DATE ────────────────────────────────────────────────────────────────
print("\n--- Decoding birth dates ---")
df["birth_year"], df["birth_month"] = cmc_to_ym(df["b3"])
print(f"Birth year range: " f"{df['birth_year'].min()} – {df['birth_year'].max()}")
print("NFHS-4 births by year:")
print(df[df.wave == 4]["birth_year"].value_counts().sort_index().to_string())
print("NFHS-5 births by year:")
print(df[df.wave == 5]["birth_year"].value_counts().sort_index().to_string())

# ── PREGNANCY DURATION → TRIMESTER WINDOWS ───────────────────────────────────
print("\n--- Constructing trimester windows ---")

# s220a = pregnancy duration in months (4–10)
# 9 months is full term, 7-8 = typical, <7 = preterm
df["preg_months"] = pd.to_numeric(df["s220a"], errors="coerce")

# Valid range: 4–10 months
df["preg_valid"] = df["preg_months"].between(4, 10)
print(f"s220a valid (4-10 months): {df['preg_valid'].mean():.1%} of births")
print(f"s220a distribution:")
print(df["preg_months"].value_counts().sort_index())

# NOTE: s220a is only recorded for the MOST RECENT birth per woman
# For other births it will be missing — this is expected
# We work with what we have and note it as a limitation

# Conception CMC = birth CMC − pregnancy duration in months
df["b3_num"] = pd.to_numeric(df["b3"], errors="coerce")
df["conception_cmc"] = df["b3_num"] - df["preg_months"]
df["conception_year"], df["conception_month"] = cmc_to_ym(df["conception_cmc"])

# ── Trimester windows as CMC ranges ──────────────────────────────────────────
# T1: conception month through month 3 of pregnancy
# T2: month 4 through month 6
# T3: month 7 through birth month
df["t1_start_cmc"] = df["conception_cmc"]
df["t1_end_cmc"] = df["conception_cmc"] + 2
df["t2_start_cmc"] = df["conception_cmc"] + 3
df["t2_end_cmc"] = df["conception_cmc"] + 5
df["t3_start_cmc"] = df["conception_cmc"] + 6
df["t3_end_cmc"] = df["b3_num"] - 1

# Convert boundaries to year-month
for tri in ["t1_start", "t1_end", "t2_start", "t2_end", "t3_start", "t3_end"]:
    df[f"{tri}_year"], df[f"{tri}_month"] = cmc_to_ym(df[f"{tri}_cmc"])

# ── Robustness: ±2 month window expansion for misclassification check ─────────
df["t1_start_cmc_wide"] = df["conception_cmc"] - 1
df["t1_end_cmc_wide"] = df["conception_cmc"] + 3
df["t3_start_cmc_wide"] = df["conception_cmc"] + 5
df["t3_end_cmc_wide"] = df["b3_num"]

print(
    f"Conception year range: "
    f"{df['conception_year'].min()} – {df['conception_year'].max()}"
)

# ── OUTCOMES ──────────────────────────────────────────────────────────────────
print("\n--- Constructing outcome variables ---")

# ── Birth weight ──────────────────────────────────────────────────────────────
# m19 = birth weight in kg (3 decimals), stored as float
# Missing/don't know codes: 9.996, 9.997, 9.998, 9.999
# Valid range: 0.5 kg – 9.0 kg
bw_raw = pd.to_numeric(df["m19"], errors="coerce")
df["birthweight_kg"] = bw_raw.where(bw_raw <= 9.0, np.nan)

# Also convert to grams for easier interpretation
df["birthweight_g"] = df["birthweight_kg"] * 1000

# PRIMARY OUTCOME: Low birth weight (<2.5 kg)
df["lbw"] = (df["birthweight_kg"] < 2.5).astype("Int64")
df.loc[df["birthweight_kg"].isna(), "lbw"] = pd.NA

# Very low birth weight (<1.5 kg) — additional outcome
df["vlbw"] = (df["birthweight_kg"] < 1.5).astype("Int64")
df.loc[df["birthweight_kg"].isna(), "vlbw"] = pd.NA

# ── Preterm ───────────────────────────────────────────────────────────────────
# <7 months gestation (approximately <30 weeks)
df["preterm"] = (df["preg_months"] < 7).astype("Int64")
df.loc[~df["preg_valid"], "preterm"] = pd.NA

# ── Mortality ─────────────────────────────────────────────────────────────────
b5_num = pd.to_numeric(df["b5"], errors="coerce")
b7_num = pd.to_numeric(df["b7"], errors="coerce")

# Neonatal mortality: died within 1 month
df["neonatal_death"] = ((b5_num == 0) & (b7_num <= 1)).astype("Int64")

# Post-neonatal mortality: died between 1-11 months
df["postneonatal_death"] = ((b5_num == 0) & (b7_num > 1) & (b7_num <= 11)).astype(
    "Int64"
)

# Infant mortality: died within 12 months
df["infant_death"] = ((b5_num == 0) & (b7_num <= 11)).astype("Int64")

# ── Birth size perceived (m18) ────────────────────────────────────────────────
# 1=very large, 2=larger than average, 3=average, 4=smaller than average, 5=very small
df["size_perceived"] = pd.to_numeric(df["m18"], errors="coerce")

# Outcome summary
print(f"Outcome variable rates:")
print(f"  LBW (<2.5kg):          " f"{df['lbw'].mean():.3f}  " f"(expect 0.15–0.30)")
print(f"  VLBW (<1.5kg):         {df['vlbw'].mean():.3f}")
print(f"  Preterm (<7mo):        {df['preterm'].mean():.3f}")
print(f"  Neonatal mortality:    {df['neonatal_death'].mean():.4f}")
print(f"  Infant mortality:      {df['infant_death'].mean():.4f}")
print(f"  Missing birth weight:  {df['birthweight_kg'].isna().mean():.1%}")

# Digit preference diagnostic
print(f"\nDigit preference in birth weight (grams):")
for w in [1000, 1500, 2000, 2500, 3000, 3500, 4000]:
    n = (df["birthweight_g"] == w).sum()
    if n > 0:
        print(f"  Exactly {w}g: {n:,}")

# ── CONTROLS ──────────────────────────────────────────────────────────────────
print("\n--- Constructing control variables ---")

df["maternal_age"] = pd.to_numeric(df["v012"], errors="coerce")
df["education"] = pd.to_numeric(df["v106"], errors="coerce")
df["wealth_q"] = pd.to_numeric(df["v190"], errors="coerce")
df["wealth_score"] = pd.to_numeric(df["v191"], errors="coerce")
df["rural"] = (pd.to_numeric(df["v025"], errors="coerce") == 2).astype("Int64")
df["birth_order"] = pd.to_numeric(df["bord"], errors="coerce")
df["anc_visits"] = pd.to_numeric(df["m14"], errors="coerce").clip(0, 20)
df["religion"] = pd.to_numeric(df["v130"], errors="coerce")
df["caste"] = pd.to_numeric(df["v131"], errors="coerce")

# Institutional delivery
# m15 codes vary by wave — facility codes are typically 20–99
m15_num = pd.to_numeric(df["m15"], errors="coerce")
df["institutional"] = (m15_num >= 20).astype("Int64")
df.loc[m15_num.isna(), "institutional"] = pd.NA

# Caesarean
df["caesarean"] = pd.to_numeric(df["m17"], errors="coerce")

# Survey weight
df["weight"] = pd.to_numeric(df["v005"], errors="coerce") / 1_000_000

# ── IDENTIFIERS ───────────────────────────────────────────────────────────────
print("\n--- Constructing identifiers ---")

# Unique mother ID across both waves
df["mother_id"] = (
    df["v001"].astype(str).str.strip()
    + "_"
    + df["v002"].astype(str).str.strip()
    + "_"
    + df["v003"].astype(str).str.strip()
    + "_"
    + df["wave"].astype(str)
)

# Flag mothers with ≥2 births (for mother fixed effects)
birth_counts = df.groupby("mother_id")["b3"].count()
multi_mothers = birth_counts[birth_counts >= 2].index
df["multi_birth_mother"] = df["mother_id"].isin(multi_mothers).astype("Int64")
print(f"Mothers with ≥2 births: {df['multi_birth_mother'].mean():.1%}")

# Migration robustness filter
df["long_resident"] = (pd.to_numeric(df["v104"], errors="coerce") >= 5).astype("Int64")

# District variable
df["district"] = pd.to_numeric(df["sdistri"], errors="coerce")

# ── SAMPLE RESTRICTIONS ───────────────────────────────────────────────────────
print("\n--- Sample restrictions ---")
n0 = len(df)

# 1. Singleton births only
b0_num = pd.to_numeric(df["b0"], errors="coerce")
df = df[b0_num.isin([0]) | b0_num.isna()]
print(f"After singleton filter:        {len(df):,}  (dropped {n0-len(df):,})")

# 2. Valid birth date
n1 = len(df)
df = df[df["birth_year"].between(2005, 2022)]
print(f"After valid birth year:        {len(df):,}  (dropped {n1-len(df):,})")

# 3. Non-missing district
n2 = len(df)
df = df[df["district"].notna()]
print(f"After non-missing district:    {len(df):,}  (dropped {n2-len(df):,})")

# 4. Non-missing LBW (for analysis sample — keep missing for sensitivity)
# NOTE: We do NOT drop missing LBW here
# We keep the full sample and flag which births have outcome data
df["has_bw"] = df["birthweight_kg"].notna().astype("Int64")
print(f"Births with birth weight data: {df['has_bw'].mean():.1%}")
print(f"Births with s220a duration:    {df['preg_valid'].mean():.1%}")

print(f"\nFinal combined sample: {len(df):,} births")
print(f"  NFHS-4: {(df.wave==4).sum():,}")
print(f"  NFHS-5: {(df.wave==5).sum():,}")

# ── FINAL COLUMN ORDER ────────────────────────────────────────────────────────
FINAL_COLS = [
    # Core identifiers
    "wave",
    "survey_label",
    "caseid",
    "mother_id",
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
    # Outcomes
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
    # Birth date
    "b3",
    "b3_num",
    "birth_year",
    "birth_month",
    "b1",
    "b2",
    # Pregnancy duration
    "preg_months",
    "preg_valid",
    # Conception + trimester windows
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
    # Wide windows for robustness
    "t1_start_cmc_wide",
    "t1_end_cmc_wide",
    "t3_start_cmc_wide",
    "t3_end_cmc_wide",
    # Controls
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
    # Raw DHS vars (keep for reference)
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
]

final_cols = [c for c in FINAL_COLS if c in df.columns]
df_out = df[final_cols].reset_index(drop=True)

# ── QUALITY CHECKS ────────────────────────────────────────────────────────────
print("\n--- Quality checks ---")
print(f"Final shape: {df_out.shape}")

print(f"\nMissing data in key variables:")
for col in [
    "lbw",
    "preterm",
    "neonatal_death",
    "birthweight_kg",
    "district",
    "wealth_q",
    "maternal_age",
    "preg_months",
    "conception_cmc",
    "t3_start_cmc",
]:
    if col in df_out.columns:
        pct = df_out[col].isna().mean() * 100
        print(f"  {col:30s}: {pct:.1f}%")

print(f"\nOutcome rates by wave:")
for wave, lbl in [(4, "NFHS-4"), (5, "NFHS-5")]:
    sub = df_out[df_out.wave == wave]
    print(f"  {lbl}:")
    print(f"    LBW rate:         {sub['lbw'].mean():.3f}")
    print(f"    Neonatal mort:    {sub['neonatal_death'].mean():.4f}")
    print(f"    With birth wt:    {sub['has_bw'].mean():.1%}")
    print(f"    With preg dur:    {sub['preg_valid'].mean():.1%}")

print(f"\nDistricts: {df_out['district'].nunique()}")
print(f"Birth years: {df_out['birth_year'].min()} – " f"{df_out['birth_year'].max()}")
print(f"Wealth quintile distribution:")
print(df_out["wealth_q"].value_counts().sort_index())

# ── SAVE ──────────────────────────────────────────────────────────────────────
print("\n--- Saving outputs ---")

csv_path = os.path.join(OUTPUT_DIR, "nfhs_births_combined.csv")
df_out.to_csv(csv_path, index=False)
print(f"CSV:     {csv_path}  ({os.path.getsize(csv_path)/1e6:.1f} MB)")

parquet_path = os.path.join(OUTPUT_DIR, "nfhs_births_combined.parquet")
df_out.to_parquet(parquet_path, index=False)
print(f"Parquet: {parquet_path}  ({os.path.getsize(parquet_path)/1e6:.1f} MB)")

# Variable labels reference
labels = {}
for col in df_out.columns:
    l4 = meta4.column_names_to_labels.get(col, "")
    l5 = meta5.column_names_to_labels.get(col, "")
    labels[col] = l4 or l5 or col

pd.DataFrame(list(labels.items()), columns=["variable", "label"]).to_csv(
    os.path.join(OUTPUT_DIR, "nfhs_births_variable_labels.csv"), index=False
)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Total births:  {len(df_out):,}")
print(f"  NFHS-4:      {(df_out.wave==4).sum():,}")
print(f"  NFHS-5:      {(df_out.wave==5).sum():,}")
print(f"Columns:       {len(df_out.columns)}")
print(f"Districts:     {df_out['district'].nunique()}")
print(f"\nSaved to: {OUTPUT_DIR}")
print(f"  nfhs_births_combined.csv")
print(f"  nfhs_births_combined.parquet  ← use in notebooks")
print(f"  nfhs_births_variable_labels.csv")
print(f"\nIMPORTANT NOTES:")
print(f"  - preg_valid = True for {df_out['preg_valid'].mean():.1%} of births")
print(f"    (s220a only recorded for most recent birth per woman)")
print(f"  - Trimester windows only meaningful where preg_valid=True")
print(f"  - has_bw = True for {df_out['has_bw'].mean():.1%} of births")
print(f"    (birth weight only recorded at health facilities)")
print(f"\nNext: run Notebook 01 (IMD Tmax processing)")
