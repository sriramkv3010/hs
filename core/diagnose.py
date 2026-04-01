"""
Fixed diagnostic — avoid qcut ties issue
Run: python3 diagnose.py
"""

import pandas as pd
import numpy as np

df = pd.read_parquet("Dataset/processed/final_analytical_dataset.parquet")

for c in [
    "lbw",
    "t3_hot_days_35",
    "t3_tmax_anomaly",
    "birth_month_tmax_anomaly",
    "wealth_q",
    "district",
    "birth_year",
    "birth_month",
    "rural",
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

valid = df[df["lbw"].notna() & df["t3_hot_days_35"].notna()].copy()
print(f"Sample: {len(valid):,}")

# ── 1. Raw correlations ────────────────────────────────────────────────────────
print("\n=== 1. RAW CORRELATIONS (no controls) ===")
print(f"hot_days_35  vs lbw: {valid['t3_hot_days_35'].corr(valid['lbw']):.4f}")
print(f"tmax_anomaly vs lbw: {valid['t3_tmax_anomaly'].corr(valid['lbw']):.4f}")

# ── 2. Hot days distribution — WHY qcut fails ─────────────────────────────────
print("\n=== 2. HOT DAYS DISTRIBUTION ===")
print(valid["t3_hot_days_35"].describe())
print(f"\nZero hot days: {(valid['t3_hot_days_35']==0).mean():.1%}")
print(f">0 hot days:   {(valid['t3_hot_days_35']>0).mean():.1%}")
print(f">30 hot days:  {(valid['t3_hot_days_35']>30).mean():.1%}")
print(f">60 hot days:  {(valid['t3_hot_days_35']>60).mean():.1%}")

# ── 3. LBW by hot days — use cut not qcut ────────────────────────────────────
print("\n=== 3. LBW RATE BY HOT DAYS (fixed cut points) ===")
bins = [-1, 0, 10, 30, 60, 125]
labels = ["0 days", "1-10", "11-30", "31-60", "61+"]
valid["heat_grp"] = pd.cut(valid["t3_hot_days_35"], bins=bins, labels=labels)
tbl = valid.groupby("heat_grp", observed=True).agg(
    n=("lbw", "count"), lbw_rate=("lbw", "mean")
)
tbl["lbw_pct"] = (tbl["lbw_rate"] * 100).round(2)
print(tbl[["n", "lbw_pct"]])
print("\n↑ KEY: if sign is correct, LBW% should RISE from '0 days' to '61+'")

# ── 4. LBW coding check ───────────────────────────────────────────────────────
print("\n=== 4. LBW CODING CHECK ===")
print(f"lbw=1 count: {(valid['lbw']==1).sum():,}  ({(valid['lbw']==1).mean():.1%})")
print(f"lbw=0 count: {(valid['lbw']==0).sum():,}  ({(valid['lbw']==0).mean():.1%})")
print(
    f"Mean birthweight where lbw=1: {valid[valid.lbw==1]['birthweight_g'].mean():.0f}g"
)
print(
    f"Mean birthweight where lbw=0: {valid[valid.lbw==0]['birthweight_g'].mean():.0f}g"
)
print("↑ lbw=1 should have LOWER birthweight than lbw=0")

# ── 5. Geography confounding ──────────────────────────────────────────────────
print("\n=== 5. GEOGRAPHY CONFOUNDING ===")
dist = (
    valid.groupby("district")
    .agg(
        hot=("t3_hot_days_35", "mean"), wealth=("wealth_q", "mean"), lbw=("lbw", "mean")
    )
    .reset_index()
)
print(f"Corr(district heat, wealth): {dist['hot'].corr(dist['wealth']):.4f}")
print(f"Corr(district heat, LBW):    {dist['hot'].corr(dist['lbw']):.4f}")
print("\nDistrict heat tercile → wealth and LBW:")
dist["ht"] = pd.qcut(
    dist["hot"], q=3, labels=["cool", "medium", "hot"], duplicates="drop"
)
print(dist.groupby("ht", observed=True)[["wealth", "lbw"]].mean().round(4))

# ── 6. Seasonal confounding ───────────────────────────────────────────────────
print("\n=== 6. BIRTH MONTH vs LBW ===")
bm = valid.groupby("birth_month")["lbw"].mean().round(4)
print(bm)
print("\n=== BIRTH MONTH vs HOT DAYS T3 ===")
bm_h = valid.groupby("birth_month")["t3_hot_days_35"].mean().round(1)
print(bm_h)
print(
    "\nCorr(birth_month, hot_days): "
    f"{valid['birth_month'].corr(valid['t3_hot_days_35']):.4f}"
)

# ── 7. Grid resolution check ──────────────────────────────────────────────────
print("\n=== 7. GRID RESOLUTION ===")
print(f"Unique tmax_lat values: {df['tmax_lat'].nunique()}")
print(f"Unique tmax_lon values: {df['tmax_lon'].nunique()}")
print(f"Unique (lat,lon) pairs: {df.groupby(['tmax_lat','tmax_lon']).ngroups}")
print(f"Total districts: {df['district'].nunique()}")
print(
    f"→ Multiple districts share same grid cell: "
    f"{df['district'].nunique() > df.groupby(['tmax_lat','tmax_lon']).ngroups}"
)

# ── 8. Year trend ─────────────────────────────────────────────────────────────
print("\n=== 8. YEAR TRENDS ===")
yr = (
    valid.groupby("birth_year")
    .agg(lbw=("lbw", "mean"), hot_days=("t3_hot_days_35", "mean"))
    .round(4)
)
print(yr)
print(f"\nCorr(year, lbw):      {valid['birth_year'].corr(valid['lbw']):.4f}")
print(f"Corr(year, hot_days): {valid['birth_year'].corr(valid['t3_hot_days_35']):.4f}")
