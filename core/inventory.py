"""
DATASET VERIFICATION CHECK
python3 verify_dataset.py
"""

import pandas as pd, numpy as np

PROCESSED = "Dataset/processed"

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

df = pd.read_parquet(f"{PROCESSED}/final_analytical_dataset.parquet")

print(f"\nShape: {df.shape}")
print(f"Columns: {len(df.columns)}")

print(f"\n--- WAVE COUNTS ---")
print(f"N4: {(df.wave==4).sum():,}")
print(f"N5: {(df.wave==5).sum():,}")
print(f"Total: {len(df):,}")

print(f"\n--- LBW ---")
print(
    f"N4 LBW rate: {df[df.wave==4]['lbw'].mean():.4f}  non-null: {df[df.wave==4]['lbw'].notna().sum():,}"
)
print(
    f"N5 LBW rate: {df[df.wave==5]['lbw'].mean():.4f}  non-null: {df[df.wave==5]['lbw'].notna().sum():,}"
)

print(f"\n--- MORTALITY ---")
print(f"N4 neonatal: {df[df.wave==4]['neonatal_death'].notna().sum():,}")
print(f"N5 neonatal: {df[df.wave==5]['neonatal_death'].notna().sum():,}")

print(f"\n--- CLIMATE ---")
print(f"N4 T3 hot days: {df[df.wave==4]['t3_hot_days_35'].notna().sum():,}")
print(
    f"N5 T3 hot days: {df[df.wave==5]['t3_hot_days_35'].notna().sum():,}  (expected 0 — no preg duration in N5)"
)
print(
    f"N4 birth-month tmax: {df[df.wave==4]['birth_month_tmax_anomaly'].notna().sum():,}"
)
print(
    f"N5 birth-month tmax: {df[df.wave==5]['birth_month_tmax_anomaly'].notna().sum():,}"
)

print(f"\n--- HOUSEHOLD ---")
print(f"N4 housing: {df[df.wave==4]['housing_quality_score'].notna().mean():.0%}")
print(f"N5 housing: {df[df.wave==5]['housing_quality_score'].notna().mean():.0%}")
print(f"N4 electricity: {df[df.wave==4]['has_electricity'].notna().mean():.0%}")
print(f"N5 electricity: {df[df.wave==5]['has_electricity'].notna().mean():.0%}")

print(f"\n--- INDIVIDUAL ---")
print(f"N4 anaemia: {df[df.wave==4]['anaemic'].notna().mean():.0%}")
print(f"N5 anaemia: {df[df.wave==5]['anaemic'].notna().mean():.0%}")
print(f"N4 BMI: {df[df.wave==4]['bmi'].notna().mean():.0%}")
print(f"N5 BMI: {df[df.wave==5]['bmi'].notna().mean():.0%}")

print(f"\n--- NHS ---")
nhs_cols = [c for c in df.columns if c.startswith("nhs_")]
print(f"NHS cols: {len(nhs_cols)}")
if nhs_cols:
    print(f"Coverage N4: {df[df.wave==4][nhs_cols[0]].notna().mean():.0%}")
    print(f"Coverage N5: {df[df.wave==5][nhs_cols[0]].notna().mean():.0%}")

print(f"\n--- IGP ---")
igp_n4 = (df[df.wave == 4]["igp_state"] == 1).sum()
igp_n5 = (df[df.wave == 5]["igp_state"] == 1).sum()
print(f"N4 IGP births: {igp_n4:,} ({igp_n4/(df.wave==4).sum():.1%})")
print(f"N5 IGP births: {igp_n5:,} ({igp_n5/(df.wave==5).sum():.1%})")

# Check IGP state codes
igp_states = sorted(
    df[df["igp_state"] == 1]["state"].dropna().unique().astype(int).tolist()
)
print(f"IGP v024 codes: {igp_states}")

print(f"\n--- BIRTH YEARS ---")
print(
    f"N4: {df[df.wave==4]['birth_year'].min():.0f}–{df[df.wave==4]['birth_year'].max():.0f}"
)
print(
    f"N5: {df[df.wave==5]['birth_year'].min():.0f}–{df[df.wave==5]['birth_year'].max():.0f}"
)

print(f"\n--- WEALTH GRADIENT (N4+N5) ---")
for q in [1, 2, 3, 4, 5]:
    s = df[df["wealth_q"] == q]
    print(f"  Q{q}: LBW={s['lbw'].mean():.4f}  n={len(s):,}")

print(f"\n--- MISSING KEY VARS ---")
for col in [
    "lbw",
    "district",
    "state",
    "birth_year",
    "birth_month",
    "maternal_age",
    "wealth_q",
    "rural",
    "birth_order",
]:
    pct = df[col].isna().mean() * 100
    print(f"  {col:25s}: {pct:.1f}% missing")

print(f"\n{'='*60}")
print("VERDICT")
print("=" * 60)
checks = {
    "N5 births present": (df.wave == 5).sum() > 500000,
    "N5 LBW available": df[df.wave == 5]["lbw"].notna().sum() > 100000,
    "N5 mortality available": df[df.wave == 5]["neonatal_death"].notna().sum() > 500000,
    "N5 household merged": df[df.wave == 5]["housing_quality_score"].notna().mean()
    > 0.8,
    "N5 individual merged": df[df.wave == 5]["anaemic"].notna().mean() > 0.8,
    "N4 T3 climate present": df[df.wave == 4]["t3_hot_days_35"].notna().sum() > 100000,
    "Birth-month N4+N5": df["birth_month_tmax_anomaly"].notna().sum() > 500000,
    "IGP correctly defined": len(igp_states) >= 6,
    "NHS merged": len(nhs_cols) > 0,
    "Wealth gradient correct": df[df.wealth_q == 1]["lbw"].mean()
    > df[df.wealth_q == 5]["lbw"].mean(),
}
all_pass = True
for check, result in checks.items():
    status = "✅" if result else "❌"
    if not result:
        all_pass = False
    print(f"  {status} {check}")
print(
    f"\n  {'ALL CHECKS PASS ✅' if all_pass else 'SOME CHECKS FAILED — see ❌ above'}"
)
