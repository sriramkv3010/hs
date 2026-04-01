"""
COMPLETE FINAL ANALYTICAL DATASET — v2 CORRECTED
═══════════════════════════════════════════════════
ROOT CAUSE OF NFHS-5 TRIMESTER FAILURE:
  Original script hardcoded b5["preg_valid"] = False and
  b5["t3_start_cmc"] = np.nan for ALL NFHS-5 rows, then
  gated the entire T1/T2/T3 climate loop on preg_valid==1.
  Result: zero NFHS-5 rows ever entered the climate loop.

  The comment "s220a not available in NFHS-5" was correct
  but the conclusion was wrong. s220a gives exact gestational
  duration for NFHS-4. For NFHS-5 we use the standard
  9-month approximation from b3 (birth CMC), which is what
  every comparable published study uses.

WHAT CHANGED:
  FIX-1  Step 2: Compute t1/t2/t3 CMC windows from b3.
                 Set preg_valid=1 for N5 rows with valid b3.
  FIX-2  Step 8: valid_t3 now passes for N5 automatically.
  FIX-3  Step 10: SDs computed on combined N4+N5 data.
                  t1_tmax_anomaly and t2_tmax_anomaly added.

Run: python3 climate_merge_final_v2.py  (~20-30 min)
"""

import os, warnings, shutil
import numpy as np, pandas as pd, pyreadstat

warnings.filterwarnings("ignore")

PROCESSED = "Dataset/processed"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


print("=" * 65)
print("BUILDING COMPLETE ANALYTICAL DATASET v2")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# 1. NFHS-4 — unchanged
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] NFHS-4 births...")
b4 = pd.read_parquet(f"{PROCESSED}/nfhs_births_combined.parquet")
b4 = b4[b4["wave"] == 4].copy().reset_index(drop=True)
for c in ["v001", "v002", "v003"]:
    b4[c] = to_num(b4[c]).astype("Int64")
m19_4 = to_num(b4["m19"])
bw4 = m19_4.where((m19_4 >= 500) & (m19_4 <= 9000), np.nan)
b4["birthweight_g"] = bw4
b4["birthweight_kg"] = bw4 / 1000
b4["lbw"] = (bw4 < 2500).astype("Int64")
b4.loc[bw4.isna(), "lbw"] = pd.NA
b4["vlbw"] = (bw4 < 1500).astype("Int64")
b4.loc[bw4.isna(), "vlbw"] = pd.NA
b4["has_bw"] = bw4.notna().astype("Int64")
b4["district"] = to_num(b4["sdistri"]).astype(float)
b4["state"] = to_num(b4["v024"]).astype(float)
b4["igp_state"] = b4["state"].isin([9, 10, 20, 8, 23]).astype("Int64")
b4["trimester_method"] = "exact"  # s220a-based precise windows
print(
    f"    N4:{len(b4):,}  LBW={b4['lbw'].mean():.4f}  "
    f"has_bw={b4['has_bw'].mean():.1%}"
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. NFHS-5 — FIX-1: build trimester CMC windows from b3
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] NFHS-5 births (~5 min)...")
PATH5 = "Dataset/1.NFHS/NHFS 5 (2019-20)/IABR7EDT/IABR7EFL.DTA"
VARS5 = [
    "caseid",
    "v001",
    "v002",
    "v003",
    "v012",
    "v021",
    "v022",
    "v023",
    "v024",
    "v025",
    "v104",
    "v105",
    "v106",
    "v130",
    "v131",
    "v190",
    "v191",
    "v201",
    "sdist",
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
    "bord",
    "m14",
    "m15",
    "m17",
    "m18",
    "m19",
]
_, meta5 = pyreadstat.read_dta(PATH5, metadataonly=True)
load5 = [v for v in VARS5 if v in set(meta5.column_names)]
df5, _ = pyreadstat.read_dta(PATH5, usecols=load5, apply_value_formats=False)

b5 = pd.DataFrame()
b5["caseid"] = df5["caseid"].astype(str).str.strip()
b5["wave"] = 5
b5["survey_label"] = "NFHS-5 (2019-21)"
for c in ["v001", "v002", "v003"]:
    b5[c] = to_num(df5[c]).astype("Int64")
for c in [
    "v021",
    "v022",
    "v023",
    "v024",
    "v025",
    "v104",
    "v106",
    "v130",
    "v131",
    "v190",
    "v191",
]:
    b5[c] = to_num(df5[c]).astype(float)
for c in ["v012", "v105", "v201"]:
    b5[c] = to_num(df5[c]).astype(float) if c in df5.columns else np.nan
b5["v104"] = to_num(df5["v104"]).astype(float)
b5["district"] = to_num(df5["sdist"]).astype(float)
b5["sdistri"] = b5["district"]
b5["state"] = b5["v024"]
b5["igp_state"] = b5["v024"].isin([9, 10, 20, 8, 23]).astype("Int64")
b5["weight"] = 1.0

b3_5 = to_num(df5["b3"])
b5["b3"] = b3_5.astype(float)
b5["b3_num"] = b3_5.astype(float)
b5["birth_year"] = (1900 + (b3_5 - 1) // 12).astype("Int64")
b5["birth_month"] = (b3_5 - (b5["birth_year"] - 1900) * 12).astype("Int64")
for c in ["b0", "b1", "b2", "b4", "b5", "b6", "b7", "b8", "b11", "b12"]:
    b5[c] = to_num(df5[c]).astype(float) if c in df5.columns else np.nan

m19_5 = to_num(df5["m19"])
bw5 = m19_5.where((m19_5 >= 500) & (m19_5 <= 9000), np.nan)
b5["m19"] = m19_5.astype(float)
b5["birthweight_g"] = bw5
b5["birthweight_kg"] = bw5 / 1000
b5["lbw"] = (bw5 < 2500).astype("Int64")
b5.loc[bw5.isna(), "lbw"] = pd.NA
b5["vlbw"] = (bw5 < 1500).astype("Int64")
b5.loc[bw5.isna(), "vlbw"] = pd.NA
b5["has_bw"] = bw5.notna().astype("Int64")

b5_a = to_num(df5["b5"])
b7_a = to_num(df5["b7"])
b5["neonatal_death"] = ((b5_a == 0) & (b7_a <= 1)).astype("Int64")
b5["postneonatal_death"] = ((b5_a == 0) & (b7_a > 1) & (b7_a <= 11)).astype("Int64")
b5["infant_death"] = ((b5_a == 0) & (b7_a <= 11)).astype("Int64")
b5["rural"] = (to_num(df5["v025"]) == 2).astype("Int64")
b5["birth_order"] = to_num(df5["bord"]).astype(float)
b5["bord"] = b5["birth_order"]
b5["maternal_age"] = b5["v012"]
b5["education"] = b5["v106"]
b5["wealth_q"] = b5["v190"]
b5["wealth_score"] = b5["v191"]
b5["long_resident"] = (b5["v104"] >= 5).astype("Int64")
b5["religion"] = b5["v130"]
b5["caste"] = b5["v131"]
b5["anc_visits"] = to_num(df5["m14"]).clip(0, 20).astype(float)
m15_5 = to_num(df5["m15"])
b5["institutional"] = (m15_5 >= 20).astype("Int64")
b5.loc[m15_5.isna(), "institutional"] = pd.NA
b5["m14"] = b5["anc_visits"]
b5["m15"] = m15_5.astype(float)
b5["m17"] = to_num(df5["m17"]).astype(float) if "m17" in df5.columns else np.nan
b5["m18"] = to_num(df5["m18"]).astype(float) if "m18" in df5.columns else np.nan
b5["caesarean"] = b5["m17"]
b5["size_perceived"] = b5["m18"]  # m18=perceived size, NOT gestational age
b5["preg_months"] = np.nan
b5["s220a"] = np.nan
b5["preterm"] = np.nan  # requires exact gestation — N4 only

# ── FIX-1: BUILD TRIMESTER CMC WINDOWS FROM b3 ───────────────────────────────
# NFHS-4 used s220a (exact gestational duration in months) to place T1/T2/T3.
# NFHS-5 does NOT have s220a. We use the 9-month standard approximation:
#   T3 = last 3 months  → CMC window: (b3 - 2) to b3
#   T2 = middle 3 months → CMC window: (b3 - 5) to (b3 - 3)
#   T1 = first 3 months → CMC window: (b3 - 8) to (b3 - 6)
# This introduces ±2-week boundary noise addressed in robustness checks.
b3_valid = b3_5.notna() & (b3_5 > 0)

# Core trimester windows
b5["t3_end_cmc"] = b3_5.where(b3_valid, np.nan)
b5["t3_start_cmc"] = (b3_5 - 2).where(b3_valid, np.nan)
b5["t2_end_cmc"] = (b3_5 - 3).where(b3_valid, np.nan)
b5["t2_start_cmc"] = (b3_5 - 5).where(b3_valid, np.nan)
b5["t1_end_cmc"] = (b3_5 - 6).where(b3_valid, np.nan)
b5["t1_start_cmc"] = (b3_5 - 8).where(b3_valid, np.nan)

# Wide robustness windows (±1 month at boundaries)
b5["t1_start_cmc_wide"] = (b3_5 - 9).where(b3_valid, np.nan)
b5["t1_end_cmc_wide"] = (b3_5 - 5).where(b3_valid, np.nan)
b5["t3_start_cmc_wide"] = (b3_5 - 3).where(b3_valid, np.nan)
b5["t3_end_cmc_wide"] = b3_5.where(b3_valid, np.nan)

# Trimester start year/month helpers
for tri_label, cmc_offset in [
    ("t1_start", -8),
    ("t2_start", -5),
    ("t3_start", -2),
    ("t3_end", 0),
]:
    cmc_v = (b3_5 + cmc_offset).where(b3_valid, np.nan)
    yr_v = (1900 + (cmc_v - 1) // 12).where(b3_valid, np.nan)
    mo_v = (cmc_v - (yr_v - 1900) * 12).where(b3_valid, np.nan)
    b5[f"{tri_label}_month"] = mo_v
    b5[f"{tri_label}_year"] = yr_v

# Conception date (9 months before birth)
b5["conception_cmc"] = (b3_5 - 9).where(b3_valid, np.nan)
b5["conception_month"] = ((b5["conception_cmc"] - 1) % 12 + 1).where(b3_valid, np.nan)
b5["conception_year"] = (1900 + (b5["conception_cmc"] - 1) // 12).where(
    b3_valid, np.nan
)

# FIX-1 KEY LINE: mark N5 rows as preg_valid=1 so climate loop includes them
b5["preg_valid"] = b3_valid.astype(int)  # was hardcoded False — THIS WAS THE BUG

# Tag method for transparency in analysis
b5["trimester_method"] = "approx_b3"

b5["mother_id"] = (
    b5["v001"].astype(str)
    + "_"
    + b5["v002"].astype(str)
    + "_"
    + b5["v003"].astype(str)
    + "_5"
)
bc5 = b5.groupby("mother_id")["b3"].count()
b5["multi_birth_mother"] = b5["mother_id"].isin(bc5[bc5 >= 2].index).astype("Int64")
b5 = b5[b5["b0"].isin([0]) | b5["b0"].isna()]
b5 = b5[b5["birth_year"].between(2005, 2022)]
b5 = b5[b5["district"].notna()]
b5 = b5.reset_index(drop=True)

n5_pv = b5["preg_valid"].eq(1).sum()
print(
    f"    N5:{len(b5):,}  LBW={b5['lbw'].mean():.4f}  "
    f"has_bw={b5['has_bw'].mean():.1%}"
)
print(f"    FIX-1: preg_valid=1 for {n5_pv:,} N5 rows ({n5_pv/len(b5):.1%})")
print(f"    t3_start_cmc notna: {b5['t3_start_cmc'].notna().sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMBINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Combining...")
all_cols = sorted(set(b4.columns) | set(b5.columns))
for col in all_cols:
    if col not in b4.columns:
        b4[col] = np.nan
    if col not in b5.columns:
        b5[col] = np.nan
births = pd.concat([b4[all_cols], b5[all_cols]], axis=0, ignore_index=True)
for c in ["v001", "v002", "v003"]:
    births[c] = to_num(births[c]).astype("Int64")
pv4 = to_num(births[births.wave == 4]["preg_valid"]).eq(1).sum()
pv5 = to_num(births[births.wave == 5]["preg_valid"]).eq(1).sum()
print(
    f"    Combined:{len(births):,}  "
    f"N4={(births.wave==4).sum():,}  N5={(births.wave==5).sum():,}"
)
print(f"    preg_valid=1: N4={pv4:,}  N5={pv5:,}  ← N5 should now be large")

# ══════════════════════════════════════════════════════════════════════════════
# 4. HOUSEHOLD — unchanged
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Household merge...")
hh = pd.read_parquet(f"{PROCESSED}/nfhs_household_combined.parquet")
births["hh_key"] = (
    births["v001"].astype(str)
    + "_"
    + births["v002"].astype(str)
    + "_"
    + births["wave"].astype(str)
)
hh_keep = [
    "hh_merge_key",
    "housing_quality_score",
    "good_housing",
    "floor_quality",
    "wall_quality",
    "roof_quality",
    "has_electricity",
    "clean_fuel",
    "hv009",
    "hv014",
    "hv216",
    "hv206",
    "hv226",
]
hh_keep = [c for c in hh_keep if c in hh.columns]
hh_sub = hh[hh_keep].drop_duplicates("hh_merge_key")
n_b = len(births)
births = births.merge(hh_sub, left_on="hh_key", right_on="hh_merge_key", how="left")
assert len(births) == n_b
births = births.drop(
    columns=[c for c in ["hh_key", "hh_merge_key"] if c in births.columns]
)
print(
    f"    Housing N4:{births[births.wave==4]['housing_quality_score'].notna().mean():.1%}  "
    f"N5:{births[births.wave==5]['housing_quality_score'].notna().mean():.1%}"
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. INDIVIDUAL — unchanged
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Individual merge (anaemia, BMI)...")
ir = pd.read_parquet(f"{PROCESSED}/nfhs_individual_combined.parquet")
births["ir_key"] = (
    births["v001"].astype(str)
    + "_"
    + births["v002"].astype(str)
    + "_"
    + births["v003"].astype(str)
    + "_"
    + births["wave"].astype(str)
)
ir_keep = [
    "ir_merge_key",
    "haemoglobin_gdl",
    "haemoglobin_unadj_gdl",
    "anaemic",
    "anaemia_level",
    "bmi",
    "underweight",
    "weight_kg",
    "height_cm",
    "has_health_insurance",
]
ir_keep = [c for c in ir_keep if c in ir.columns]
ir_sub = ir[ir_keep].drop_duplicates("ir_merge_key")
n_b = len(births)
births = births.merge(ir_sub, left_on="ir_key", right_on="ir_merge_key", how="left")
assert len(births) == n_b
births = births.drop(
    columns=[c for c in ["ir_key", "ir_merge_key"] if c in births.columns]
)
print(
    f"    Anaemia N4:{births[births.wave==4]['anaemic'].notna().mean():.1%}  "
    f"N5:{births[births.wave==5]['anaemic'].notna().mean():.1%}"
)

# ══════════════════════════════════════════════════════════════════════════════
# 6. NHS — unchanged
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] NHS infrastructure...")
nhs = pd.read_csv(f"{PROCESSED}/nhs_combined.csv")
wave_map = {"NHS-2015-16": 4, "2015-16": 4, "NHS-2019-20": 5, "2019-20": 5}
nhs["nhs_wave_num"] = nhs["wave"].map(wave_map)
if nhs["nhs_wave_num"].isna().all():
    nhs["nhs_wave_num"] = nhs["year"].map(wave_map)
STATE_MAP = {
    "Jammu & Kashmir": 1,
    "Jammu and Kashmir": 1,
    "Himachal Pradesh": 2,
    "Punjab": 3,
    "Chandigarh": 4,
    "Uttarakhand": 5,
    "Uttaranchal": 5,
    "Haryana": 6,
    "NCT of Delhi": 7,
    "Delhi": 7,
    "Rajasthan": 8,
    "Uttar Pradesh": 9,
    "Bihar": 10,
    "Sikkim": 11,
    "Arunachal Pradesh": 12,
    "Arunanchal Pradesh": 12,
    "Nagaland": 13,
    "Manipur": 14,
    "Mizoram": 15,
    "Tripura": 16,
    "Meghalaya": 17,
    "Assam": 18,
    "West Bengal": 19,
    "Jharkhand": 20,
    "Odisha": 21,
    "Orissa": 21,
    "Chhattisgarh": 22,
    "Madhya Pradesh": 23,
    "Gujarat": 24,
    "Daman & Diu": 25,
    "Dadra & Nagar Havelli": 26,
    "Dadara & Nagar Havelli": 26,
    "Maharashtra": 27,
    "Andhra Pradesh": 28,
    "Telangana": 29,
    "Karnataka": 30,
    "Goa": 31,
    "Lakshadweep": 32,
    "Kerala": 33,
    "Tamil Nadu": 33,
    "Puducherry": 34,
    "Pondicherry": 34,
    "Andaman & Nicobar Islands": 35,
    "A & N Islands": 35,
    "Andaman & Nicobar Island": 35,
}
nhs["state_code_num"] = nhs["state"].map(STATE_MAP)
nhs_c = nhs[nhs["state_code_num"].notna() & nhs["nhs_wave_num"].notna()].copy()
NHS_COLS_KEEP = [
    "state_code_num",
    "nhs_wave_num",
    "estimated_pregnancies",
    "institutional_deliveries",
    "pct_institutional_deliveries",
    "newborns_weighed",
    "pct_newborns_weighed",
    "newborns_lbw_count",
    "pct_lbw",
    "anaemia_cases_hb_lt11",
    "infant_deaths",
    "pct_infant_deaths_lbw",
    "stillbirths",
]
NHS_COLS_KEEP = [c for c in NHS_COLS_KEEP if c in nhs_c.columns]
NHS_COLS_KEEP = list(dict.fromkeys(["state_code_num", "nhs_wave_num"] + NHS_COLS_KEEP))
nhs_sub = nhs_c[NHS_COLS_KEEP].copy()
ren = {
    c: f"nhs_{c}" for c in NHS_COLS_KEEP if c not in ("state_code_num", "nhs_wave_num")
}
nhs_sub = nhs_sub.rename(columns=ren)
nhs_sub = nhs_sub.drop_duplicates(["state_code_num", "nhs_wave_num"])
births["state_f"] = to_num(births["state"]).astype(float)
births["wave_f"] = to_num(births["wave"]).astype(float)
n_b = len(births)
births = births.merge(
    nhs_sub,
    left_on=["state_f", "wave_f"],
    right_on=["state_code_num", "nhs_wave_num"],
    how="left",
)
if len(births) != n_b:
    births = births.drop_duplicates(subset=["caseid", "b3", "wave_f"]).reset_index(
        drop=True
    )
births = births.drop(
    columns=[
        c
        for c in ["state_f", "wave_f", "state_code_num", "nhs_wave_num"]
        if c in births.columns
    ]
)
nhs_added = [c for c in births.columns if c.startswith("nhs_")]
print(f"    NHS cols added: {len(nhs_added)}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. TRIMESTER CMC year/month helpers — unchanged (now works for N5 too)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Trimester year-month...")


def cmc_ym(s):
    c = to_num(s)
    yr = (1900 + (c - 1) // 12).astype("Int64")
    mo = (c - (yr - 1900) * 12).astype("Int64")
    return yr, mo


for tri in ["t1_start", "t1_end", "t2_start", "t2_end", "t3_start", "t3_end"]:
    col = f"{tri}_cmc"
    if col in births.columns:
        births[f"{tri}_yr"], births[f"{tri}_mo"] = cmc_ym(births[col])

pv4 = to_num(births[births.wave == 4]["preg_valid"]).eq(1).sum()
pv5 = to_num(births[births.wave == 5]["preg_valid"]).eq(1).sum()
print(f"    preg_valid=1: N4={pv4:,}  N5={pv5:,}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. CLIMATE MERGE — FIX-2: valid_t3 now includes N5 automatically
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Climate merge...")
tmax = pd.read_parquet(f"{PROCESSED}/imd_tmax_combined.parquet")
rain = pd.read_parquet(f"{PROCESSED}/imd_rainfall_combined.parquet")
tmax_idx = tmax.set_index(["lat", "lon", "year", "month"])
rain_idx = rain.set_index(["lat", "lon", "year", "month"])
tmax_set = set(tmax_idx.index)
rain_set = set(rain_idx.index)
tmax_years = sorted(tmax["year"].unique())
print(f"    Tmax:{tmax.shape}  Rain:{rain.shape}")
print(f"    IMD Tmax year range: {tmax_years[0]}–{tmax_years[-1]}")

cw = pd.read_csv(f"{PROCESSED}/district_climate_crosswalk.csv")
births["district_int"] = to_num(births["district"]).astype(float)
births = births.merge(
    cw[["district_code", "tmax_lat", "tmax_lon", "rain_lat", "rain_lon"]],
    left_on="district_int",
    right_on="district_code",
    how="left",
)
births = births.drop(
    columns=[c for c in ["district_int", "district_code"] if c in births.columns]
)
print(f"    Grid coords: {births['tmax_lat'].notna().mean():.1%}")

valid_bm = births["tmax_lat"].notna() & to_num(births["birth_year"]).between(2009, 2021)
print(
    f"    In climate range: {valid_bm.sum():,}  "
    f"N4={(valid_bm & (births.wave==4)).sum():,}  "
    f"N5={(valid_bm & (births.wave==5)).sum():,}"
)

tlat = births["tmax_lat"].values
tlon = births["tmax_lon"].values
by = to_num(births["birth_year"]).values
bm_v = to_num(births["birth_month"]).values

# Birth-month Tmax anomaly — unchanged, works for both waves
print("    Birth-month Tmax anomaly...")
bm_anom = np.full(len(births), np.nan)
for i in range(len(births)):
    if not valid_bm.iloc[i]:
        continue
    tk = (tlat[i], tlon[i], int(by[i]), int(bm_v[i]))
    if tk in tmax_set:
        bm_anom[i] = float(tmax_idx.loc[tk]["tmax_anomaly"])
births["birth_month_tmax_anomaly"] = bm_anom
n4_bm = (~np.isnan(bm_anom) & (births.wave == 4).values).sum()
n5_bm = (~np.isnan(bm_anom) & (births.wave == 5).values).sum()
print(f"    Birth-month done: N4={n4_bm:,}  N5={n5_bm:,}")

# FIX-2: T3 gate — now passes for N5 because FIX-1 set preg_valid=1
# and t3_start_cmc is populated. No other change needed here.
valid_t3 = (
    valid_bm & to_num(births["preg_valid"]).eq(1) & births["t3_start_cmc"].notna()
)
print(
    f"    T3 valid: {valid_t3.sum():,}  "
    f"N4={(valid_t3 & (births.wave==4)).sum():,}  "
    f"N5={(valid_t3 & (births.wave==5)).sum():,}  "
    f"← N5 should now be large"
)

t3h35 = np.full(len(births), np.nan)
t3h33 = np.full(len(births), np.nan)
t3an = np.full(len(births), np.nan)
t3rf = np.full(len(births), np.nan)
t3dr = np.full(len(births), np.nan)
t1h35 = np.full(len(births), np.nan)
t1an = np.full(len(births), np.nan)  # FIX-3 NEW: T1 anomaly
t2h35 = np.full(len(births), np.nan)
t2an = np.full(len(births), np.nan)  # FIX-3 NEW: T2 anomaly

t3s = births["t3_start_cmc"].values
t3e = births["t3_end_cmc"].values
t1s = births["t1_start_cmc"].values
t1e = births["t1_end_cmc"].values
t2s = births["t2_start_cmc"].values
t2e = births["t2_end_cmc"].values
rlat = births["rain_lat"].values
rlon_v = births["rain_lon"].values

total = int(valid_t3.sum())
done = 0
rep = max(1, total // 10)
print(f"    Processing T1/T2/T3 ({total:,} rows, ~20-30 min)...")

for i in range(len(births)):
    if not valid_t3.iloc[i]:
        continue
    la, lo = tlat[i], tlon[i]
    rla, rlo = rlat[i], rlon_v[i]

    # T3
    h35 = h33 = an = rf = dr = n = 0
    for cmc in range(int(t3s[i]), int(t3e[i]) + 1):
        yr = 1900 + (cmc - 1) // 12
        mo = cmc - (yr - 1900) * 12
        if 2009 <= yr <= 2021:
            tk = (la, lo, yr, mo)
            if tk in tmax_set:
                row = tmax_idx.loc[tk]
                h35 += float(row["hot_days_35"])
                h33 += float(row["hot_days_33"])
                an += float(row["tmax_anomaly"])
                n += 1
            if not pd.isna(rla):
                rk = (rla, rlo, yr, mo)
                if rk in rain_set:
                    rr = rain_idx.loc[rk]
                    rf += float(rr["total_rainfall_mm"])
                    dr = max(dr, int(rr["drought_flag"]))
    if n > 0:
        t3h35[i] = h35
        t3h33[i] = h33
        t3an[i] = an / n
        t3rf[i] = rf
        t3dr[i] = dr

    # T1
    if not (pd.isna(t1s[i]) or pd.isna(t1e[i])):
        h = an1 = nn = 0
        for cmc in range(int(t1s[i]), int(t1e[i]) + 1):
            yr = 1900 + (cmc - 1) // 12
            mo = cmc - (yr - 1900) * 12
            if 2009 <= yr <= 2021:
                tk = (la, lo, yr, mo)
                if tk in tmax_set:
                    h += float(tmax_idx.loc[tk]["hot_days_35"])
                    an1 += float(tmax_idx.loc[tk]["tmax_anomaly"])
                    nn += 1
        if nn > 0:
            t1h35[i] = h
            t1an[i] = an1 / nn  # NEW

    # T2
    if not (pd.isna(t2s[i]) or pd.isna(t2e[i])):
        h = an2 = nn = 0
        for cmc in range(int(t2s[i]), int(t2e[i]) + 1):
            yr = 1900 + (cmc - 1) // 12
            mo = cmc - (yr - 1900) * 12
            if 2009 <= yr <= 2021:
                tk = (la, lo, yr, mo)
                if tk in tmax_set:
                    h += float(tmax_idx.loc[tk]["hot_days_35"])
                    an2 += float(tmax_idx.loc[tk]["tmax_anomaly"])
                    nn += 1
        if nn > 0:
            t2h35[i] = h
            t2an[i] = an2 / nn  # NEW

    done += 1
    if done % rep == 0:
        n5_so_far = sum(
            1
            for j in range(i + 1)
            if valid_t3.iloc[j]
            and births["wave"].iloc[j] == 5
            and not np.isnan(t3an[j])
        )
        print(
            f"      {done:,}/{total:,} ({done/total*100:.0f}%)  "
            f"N5 T3 matched so far: {n5_so_far:,}"
        )

births["t3_hot_days_35"] = t3h35
births["t3_hot_days_33"] = t3h33
births["t3_tmax_anomaly"] = t3an
births["t3_rainfall_mm"] = t3rf
births["t3_drought_flag"] = t3dr
births["t1_hot_days_35"] = t1h35
births["t1_tmax_anomaly"] = t1an  # NEW
births["t2_hot_days_35"] = t2h35
births["t2_tmax_anomaly"] = t2an  # NEW

n4_t3 = np.sum(~np.isnan(t3h35) & (births.wave == 4).values)
n5_t3 = np.sum(~np.isnan(t3h35) & (births.wave == 5).values)
print(f"    T3 matched: N4={n4_t3:,}  N5={n5_t3:,}")
if n5_t3 == 0:
    print("    *** STILL ZERO: IMD may not cover N5 birth years ***")
    print(f"    IMD covers: {tmax_years[0]}–{tmax_years[-1]}")
    print("    N5 T3 needs IMD from ~2013 onwards (T3 of 2015 births)")

# ══════════════════════════════════════════════════════════════════════════════
# 9. ERA5 — unchanged
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] ERA5...")
e5p = f"{PROCESSED}/era5_merged_monthly_1950_2023.parquet"
if os.path.exists(e5p):
    e5 = pd.read_parquet(e5p)
    clim = (
        e5[e5["year"].between(1950, 2000)].groupby("month")["tasmax"].mean().to_dict()
    )
    e5["anom"] = e5["tasmax"] - e5["month"].map(clim)
    e5map = e5.set_index(["year", "month"])["anom"].to_dict()
    births["era5_anom"] = [
        e5map.get((int(y), int(m)), np.nan) if pd.notna(y) and pd.notna(m) else np.nan
        for y, m in zip(births["birth_year"], births["birth_month"])
    ]
    print(f"    ERA5: {births['era5_anom'].notna().mean():.1%}")
else:
    births["era5_anom"] = np.nan

# ══════════════════════════════════════════════════════════════════════════════
# 10. FINAL VARIABLES + FIX-3: standardise on combined N4+N5
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Final variables and standardisation...")
births["igp_state"] = to_num(births["state"]).isin([9, 10, 20, 8, 23]).astype("Int64")

# FIX-3: now that N5 has trimester data, SDs incorporate both waves
STD_COLS = [
    "t3_hot_days_35",
    "t3_hot_days_33",
    "t3_tmax_anomaly",
    "t1_tmax_anomaly",
    "t2_tmax_anomaly",  # NEW cols
    "birth_month_tmax_anomaly",
    "t1_hot_days_35",
    "t2_hot_days_35",
    "t3_rainfall_mm",
    "era5_anom",
]
print("    Standardising (global SD, N4+N5 combined):")
for col in STD_COLS:
    if col in births.columns and births[col].notna().sum() > 100:
        sd = births[col].std()
        if sd > 0:
            births[f"{col}_std"] = (births[col] / sd).astype(float)
            n4_n = births[births.wave == 4][col].notna().sum()
            n5_n = births[births.wave == 5][col].notna().sum()
            print(f"    {col}_std: N4={n4_n:,}  N5={n5_n:,}  SD={sd:.4f}")

for c in ["mother_id"]:
    if c in births.columns:
        births.drop(columns=[c], inplace=True)

# ══════════════════════════════════════════════════════════════════════════════
# 11. FIX DUPLICATE COLUMNS — unchanged
# ══════════════════════════════════════════════════════════════════════════════
print("\n[11] Checking for duplicate columns...")
cols = list(births.columns)
seen = {}
new_cols = []
for c in cols:
    if c in seen:
        seen[c] += 1
        new_cols.append(f"_DROP_{c}_{seen[c]}")
    else:
        seen[c] = 0
        new_cols.append(c)
dup_count = sum(1 for c in new_cols if c.startswith("_DROP_"))
if dup_count > 0:
    births.columns = new_cols
    drop_these = [c for c in births.columns if c.startswith("_DROP_")]
    births = births.drop(columns=drop_these)
    print(f"    Removed {dup_count} duplicate columns")
else:
    print(f"    No duplicates found")
print(f"    Final columns: {len(births.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# 12. QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"QUALITY REPORT   Shape:{births.shape}")
print("=" * 65)
n4_t3f = births[births.wave == 4]["t3_tmax_anomaly"].notna().sum()
n5_t3f = births[births.wave == 5]["t3_tmax_anomaly"].notna().sum()
print(f"  N4:{(births.wave==4).sum():,}  N5:{(births.wave==5).sum():,}")
if n5_t3f > 0:
    print(f"  *** SUCCESS: T3 anomaly N4={n4_t3f:,}  N5={n5_t3f:,} ***")
else:
    print(f"  *** WARNING: T3 anomaly N4={n4_t3f:,}  N5={n5_t3f:,} ***")
    print(f"  *** Check IMD year coverage: {tmax_years[0]}–{tmax_years[-1]} ***")

print(f"\n  Coverage (N4 / N5):")
for col, lbl in [
    ("lbw", "LBW"),
    ("birthweight_g", "Birth weight"),
    ("neonatal_death", "Neonatal death"),
    ("birth_month_tmax_anomaly", "Birth-month Tmax"),
    ("t3_tmax_anomaly", "T3 Tmax anomaly"),
    ("t1_tmax_anomaly", "T1 Tmax anomaly"),
    ("t2_tmax_anomaly", "T2 Tmax anomaly"),
    ("t3_hot_days_35", "T3 hot days >35C"),
    ("anaemic", "Anaemia"),
    ("bmi", "BMI"),
    ("housing_quality_score", "Housing"),
    ("era5_anom", "ERA5"),
    ("trimester_method", "Trimester method"),
]:
    if col in births.columns:
        r4 = births[births.wave == 4][col].notna().mean()
        r5 = births[births.wave == 5][col].notna().mean()
        print(f"    {lbl:28s}: N4={r4:.0%}  N5={r5:.0%}")

print(f"\n  LBW by wealth:")
for q in [1, 2, 3, 4, 5]:
    s = births[births["wealth_q"] == q]
    if len(s) > 0:
        print(f"    Q{q}: {s['lbw'].mean():.4f}  n={len(s):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 13. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[13] Saving...")
old = f"{PROCESSED}/final_analytical_dataset.parquet"
if os.path.exists(old):
    shutil.copy(old, f"{PROCESSED}/final_analytical_dataset_v1_backup.parquet")
    print("    Backup saved → final_analytical_dataset_v1_backup.parquet")
births.to_parquet(old, index=False, compression="snappy")
births.to_csv(f"{PROCESSED}/final_analytical_dataset.csv", index=False)
print(
    f"    Parquet: {os.path.getsize(old)/1e6:.0f} MB  " f"Cols: {len(births.columns)}"
)

print("\n" + "=" * 65)
print("COMPLETE")
print("=" * 65)
print(
    f"""
  Total: {len(births):,}
  N4:    {(births.wave==4).sum():,}  LBW={births[births.wave==4]['lbw'].mean():.4f}
  N5:    {(births.wave==5).sum():,}  LBW={births[births.wave==5]['lbw'].mean():.4f}

  T3 Tmax anomaly:
    N4: {births[births.wave==4]['t3_tmax_anomaly'].notna().sum():,}
    N5: {births[births.wave==5]['t3_tmax_anomaly'].notna().sum():,}  ← was 0

  WHAT CHANGED vs original:
    FIX-1  N5 now has t1/t2/t3 CMC windows from b3 (9-month approx).
           preg_valid=1 for all N5 rows with valid b3.
           trimester_method='approx_b3' flags N5 rows.
    FIX-2  T3 climate loop processes N5 rows automatically.
    FIX-3  SDs on combined N4+N5. t1/t2_tmax_anomaly added.

  UNCHANGED:
    NFHS-4 uses exact s220a-based trimester windows.
    LBW still N4-heavy (DHS design for m19).
    Mortality analysis uses both waves fully.

  Next: python3 fullanalysis_v5.py
"""
)
