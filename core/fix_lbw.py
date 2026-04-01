"""
Fix LBW — m19 is in GRAMS (500-9998), not kg
Missing codes: > 9000 (9996=not weighed, 9997=inconsistent, 9998=DK, 9999=missing)
LBW threshold: < 2500 grams
Run: python3 fix_lbw2.py
"""

import pandas as pd
import numpy as np

PROCESSED = "Dataset/processed"

print("Loading...")
df = pd.read_parquet(f"{PROCESSED}/final_analytical_dataset.parquet")
print(f"Shape: {df.shape}")

# ── DIAGNOSE m19 ───────────────────────────────────────────────────────────────
m19 = pd.to_numeric(df["m19"], errors="coerce")
print(f"\nm19 raw stats:")
print(f"  min={m19.min():.0f}  max={m19.max():.0f}  mean={m19.mean():.0f}")
print(f"  non-null: {m19.notna().sum():,}")
print(f"  > 9000 (missing codes): {(m19 > 9000).sum():,}")
print(f"  500-9000 (valid grams): {((m19 >= 500) & (m19 <= 9000)).sum():,}")
print(f"  < 2500 (LBW in grams): {(m19 < 2500).sum():,}")

# ── RECONSTRUCT WITH CORRECT UNIT ─────────────────────────────────────────────
print("\n=== FIXING LBW (grams) ===")

# Valid range: 500g – 9000g (physiologically plausible)
# Missing codes: 9996, 9997, 9998, 9999 → set to NaN
bw_g = m19.where((m19 >= 500) & (m19 <= 9000), np.nan)

print(f"Valid birth weights: {bw_g.notna().sum():,} ({bw_g.notna().mean():.1%})")
print(f"Birth weight range: {bw_g.min():.0f}g – {bw_g.max():.0f}g")
print(f"Mean birth weight: {bw_g.mean():.0f}g ({bw_g.mean()/1000:.3f}kg)")

# LBW = birth weight < 2500 grams
lbw = (bw_g < 2500).astype("Int64")
lbw[bw_g.isna()] = pd.NA

print(f"\nLBW rate (<2500g): {lbw.mean():.4f}  (expect 0.15-0.30)")
print(f"LBW count: {lbw.sum():,}")
print(f"LBW missing: {lbw.isna().mean():.1%}")

# Assign to dataframe
df["birthweight_g"] = bw_g
df["birthweight_kg"] = bw_g / 1000
df["lbw"] = lbw
df["vlbw"] = (bw_g < 1500).astype("Int64")
df.loc[bw_g.isna(), "vlbw"] = pd.NA
df["has_bw"] = bw_g.notna().astype("Int64")

print(f"\nVLBW rate (<1500g): {df['vlbw'].mean():.4f}")
print(f"Has birth weight:   {df['has_bw'].mean():.1%}")

# ── FIX PRETERM ───────────────────────────────────────────────────────────────
# s220a = pregnancy duration in months (we confirmed this earlier)
# Preterm = < 8 months (approximately < 37 weeks)
preg = pd.to_numeric(df["s220a"], errors="coerce")
preg_valid = preg.between(4, 10)
df["preg_months"] = preg.where(preg_valid, np.nan)
df["preg_valid"] = preg_valid

df["preterm"] = (df["preg_months"] < 8).astype("Int64")
df.loc[~preg_valid, "preterm"] = pd.NA
print(f"\nPreterm rate (<8mo): {df['preterm'].mean():.4f}")

# ── FIX ANALYSIS FLAGS ────────────────────────────────────────────────────────
print("\n=== FIXING ANALYSIS FLAGS ===")

df["analysis_ready"] = (
    df["lbw"].notna() & df["district"].notna() & df["birth_year"].between(2009, 2021)
).astype("Int64")

df["analysis_full"] = (
    df["lbw"].notna()
    & df["district"].notna()
    & df["birth_month_tmax_anomaly"].notna()
    & df["birth_year"].between(2009, 2021)
).astype("Int64")

df["analysis_trimester"] = (
    df["analysis_full"].eq(1)
    & df["t3_tmax_anomaly"].notna()
    & df["preg_valid"].eq(True)
).astype("Int64")

print(f"analysis_ready:     {df['analysis_ready'].sum():,}")
print(f"analysis_full:      {df['analysis_full'].sum():,}")
print(f"analysis_trimester: {df['analysis_trimester'].sum():,}")

# ── VALIDATION CHECKS ─────────────────────────────────────────────────────────
print("\n=== VALIDATION ===")

print("\nLBW rate by wave:")
for w, lbl in [(4, "NFHS-4"), (5, "NFHS-5")]:
    sub = df[df.wave == w]
    print(f"  {lbl}: LBW={sub['lbw'].mean():.4f}  " f"n_births={sub['has_bw'].sum():,}")

print("\nLBW rate by wealth quintile (expect Q1 highest):")
lbw_q = df.groupby("wealth_q")["lbw"].mean().round(4)
for q, v in lbw_q.items():
    bar = "█" * int(v * 100)
    print(f"  Q{q}: {v:.4f}  {bar}")

print("\nLBW rate by rural/urban:")
print(df.groupby("rural")["lbw"].mean().round(4).rename({0: "Urban", 1: "Rural"}))

print("\nLBW rate by T3 tmax anomaly quartile:")
valid = df[df["t3_tmax_anomaly"].notna() & df["lbw"].notna()].copy()
if len(valid) > 1000:
    valid["t3_q"] = pd.qcut(
        valid["t3_tmax_anomaly"], q=4, labels=["Q1 cool", "Q2", "Q3", "Q4 hot"]
    )
    by_q = valid.groupby("t3_q", observed=True)["lbw"].mean().round(4)
    for q, v in by_q.items():
        bar = "█" * int(v * 100)
        print(f"  {q}: {v:.4f}  {bar}")
    print("  ↑ Should increase Q1→Q4 if heat drives LBW")
else:
    print("  Not enough births with both T3 and LBW data")

print("\nBirth weight distribution check:")
bw_check = df["birthweight_g"].describe()
print(bw_check.round(1))

# ── CHECK DIGIT PREFERENCE ────────────────────────────────────────────────────
print("\nDigit preference in birth weight (g):")
for w in [2000, 2500, 3000, 3500, 4000]:
    n = (df["birthweight_g"] == w).sum()
    print(f"  Exactly {w}g: {n:,}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
print("\n=== SAVING ===")
out_pq = f"{PROCESSED}/final_analytical_dataset.parquet"
out_csv = f"{PROCESSED}/final_analytical_dataset.csv"

df.to_parquet(out_pq, index=False, compression="snappy")
df.to_csv(out_csv, index=False)

print(f"Parquet: {out_pq}")
print(f"CSV:     {out_csv}")
print(f"\nFinal shape: {df.shape}")
print(f"Ready for ML/regression: {df['analysis_full'].sum():,} births")
print("\nDone ✓")
