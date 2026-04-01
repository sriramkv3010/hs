"""
IMD Rainfall GRD Combiner — Simple Version
===========================================
Combines all Rainfall_ind{YYYY}_rfp25.grd files into one CSV.
NO shapefile. NO district matching. Just raw grid data.

Output: imd_rainfall_combined.csv
        imd_rainfall_combined.parquet

Run: python3 combine_rainfall_simple.py
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────────────────────────
RAINFALL_DIR = Path.home() / "Desktop/HS/Dataset/2.IMD/imd_rainfall"
OUTPUT_DIR = Path.home() / "Desktop/HS/Dataset/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IMD 0.25° grid — confirmed from your Step 1 output
N_LAT, N_LON = 129, 135
LAT_START, LAT_END = 6.5, 38.5
LON_START, LON_END = 66.5, 100.0
FILL_VALUE = -999.0

LATS = np.round(np.linspace(LAT_START, LAT_END, N_LAT), 2)  # 129 values
LONS = np.round(np.linspace(LON_START, LON_END, N_LON), 2)  # 135 values


# ── HELPERS ────────────────────────────────────────────────────────────────────
def parse_year(filepath):
    stem = Path(filepath).stem
    # Strategy 1: 4-digit year in 2000-2025 range
    matches = re.findall(r"(200[0-9]|201[0-9]|202[0-5])", stem)
    if matches:
        return int(matches[0])
    # Strategy 2: any 4-digit sequence in 1990-2030
    matches = re.findall(r"(\d{4})", stem)
    for m in matches:
        y = int(m)
        if 1990 <= y <= 2030:
            return y
    return None


def is_leap(y):
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


# ── FIND FILES ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("IMD RAINFALL GRD → CSV COMBINER")
print("=" * 55)

grd_files = sorted(RAINFALL_DIR.glob("Rainfall_ind*_rfp25.grd"))
print(f"\nFound {len(grd_files)} .grd files in:")
print(f"  {RAINFALL_DIR}")

file_info = []
for p in grd_files:
    year = parse_year(p.name)
    if year is None:
        print(f"  SKIP: cannot parse year from {p.name}")
        continue
    n_days = 366 if is_leap(year) else 365
    file_info.append({"path": p, "year": year, "n_days": n_days})

file_info = sorted(file_info, key=lambda x: x["year"])
years = [f["year"] for f in file_info]
print(f"Years: {years[0]} – {years[-1]}")
print(f"Files: {[f['path'].name for f in file_info]}")

# ── PROCESS EACH FILE → MONTHLY TOTALS ────────────────────────────────────────
print(f"\nProcessing {len(file_info)} files → monthly totals per grid cell...")
print("(Each file: 365 days × 129 lat × 135 lon = 17,415 grid cells)")

all_records = []

for fi in tqdm(file_info, desc="Years"):
    year = fi["year"]
    n_days = fi["n_days"]
    path = fi["path"]

    # Read binary float32
    raw = np.fromfile(path, dtype=np.float32)
    grid = raw.reshape(n_days, N_LAT, N_LON).astype(np.float64)

    # Mask fill values and negatives
    grid = np.where(grid <= FILL_VALUE, np.nan, grid)
    grid = np.where(grid < 0, np.nan, grid)

    # Build date array
    dates = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
    months = dates.month.values

    # Monthly aggregates per grid cell
    for month in range(1, 13):
        mask = months == month
        n_days_in_month = int(mask.sum())
        if n_days_in_month == 0:
            continue

        monthly_slice = grid[mask, :, :]  # (days_in_month, 129, 135)

        # Monthly total rainfall per grid cell (mm)
        total_mm = np.nansum(monthly_slice, axis=0)  # (129, 135)

        # Count valid days (non-NaN)
        valid_days = np.sum(~np.isnan(monthly_slice), axis=0)

        # Set cells with zero valid days to NaN
        total_mm = np.where(valid_days == 0, np.nan, total_mm)

        # Flatten to rows: one row per grid cell
        for i in range(N_LAT):
            for j in range(N_LON):
                rf = total_mm[i, j]
                if np.isnan(rf):
                    continue  # skip ocean/outside-India cells

                all_records.append(
                    {
                        "year": year,
                        "month": month,
                        "lat": float(LATS[i]),
                        "lon": float(LONS[j]),
                        "total_rainfall_mm": round(float(rf), 2),
                        "valid_days": int(valid_days[i, j]),
                        "expected_days": n_days_in_month,
                    }
                )

print(f"\nTotal records: {len(all_records):,}")
print("Building DataFrame...")
df = pd.DataFrame(all_records)

# ── ADD RAINFALL ANOMALY ───────────────────────────────────────────────────────
print("Computing climatology and anomaly...")

# Climatology = mean total_rainfall_mm for each (lat, lon, month) across all years
clim = (
    df.groupby(["lat", "lon", "month"])["total_rainfall_mm"]
    .mean()
    .reset_index()
    .rename(columns={"total_rainfall_mm": "clim_mean_mm"})
)
df = df.merge(clim, on=["lat", "lon", "month"], how="left")
df["rainfall_anomaly_mm"] = (df["total_rainfall_mm"] - df["clim_mean_mm"]).round(2)

# Drought flag: below 20th percentile of local distribution
pct20 = (
    df.groupby(["lat", "lon", "month"])["total_rainfall_mm"]
    .quantile(0.20)
    .reset_index()
    .rename(columns={"total_rainfall_mm": "drought_threshold_mm"})
)
df = df.merge(pct20, on=["lat", "lon", "month"], how="left")
df["drought_flag"] = (df["total_rainfall_mm"] < df["drought_threshold_mm"]).astype(int)

# ── QUALITY CHECKS ─────────────────────────────────────────────────────────────
print("\n--- Quality Checks ---")
print(f"Shape:          {df.shape}")
print(f"Years:          {sorted(df['year'].unique().tolist())}")
print(f"Months:         {sorted(df['month'].unique().tolist())}")
print(f"Lat range:      {df['lat'].min()} – {df['lat'].max()}")
print(f"Lon range:      {df['lon'].min()} – {df['lon'].max()}")
print(f"Grid cells/mo:  {df[df.year==2015][df.month==1]['lat'].count():,}")
print(f"Missing rain:   {df['total_rainfall_mm'].isna().mean():.1%}")
print(f"Drought months: {df['drought_flag'].mean():.1%} (expect ~20%)")
print(f"\nRainfall stats (mm/month):")
print(df["total_rainfall_mm"].describe().round(2))
print(f"\nAnomaly stats (mm/month):")
print(df["rainfall_anomaly_mm"].describe().round(2))

# Monsoon sanity check — June-September should be much wetter
print(f"\nMean monthly rainfall by season (India average):")
season_check = df.groupby("month")["total_rainfall_mm"].mean().round(1)
for m, mm in season_check.items():
    bar = "█" * int(mm / 10)
    print(f"  Month {m:2d}: {mm:6.1f} mm  {bar}")

# ── SAVE ───────────────────────────────────────────────────────────────────────
print("\n--- Saving ---")

csv_path = OUTPUT_DIR / "imd_rainfall_combined.csv"
df.to_csv(csv_path, index=False)
print(f"CSV:     {csv_path}")
print(f"  Size:  {csv_path.stat().st_size / 1e6:.1f} MB")
print(f"  Rows:  {len(df):,}")

parquet_path = OUTPUT_DIR / "imd_rainfall_combined.parquet"
df.to_parquet(parquet_path, index=False, compression="snappy")
print(f"Parquet: {parquet_path}")
print(f"  Size:  {parquet_path.stat().st_size / 1e6:.1f} MB")

# ── SUMMARY ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("DONE")
print("=" * 55)
print(f"Combined {len(file_info)} years: {years[0]}–{years[-1]}")
print(f"Total rows:    {len(df):,}")
print(f"Columns:       {df.columns.tolist()}")
print(f"\nColumn descriptions:")
print(f"  year                 — calendar year")
print(f"  month                — calendar month (1-12)")
print(f"  lat                  — grid cell latitude (0.25° steps)")
print(f"  lon                  — grid cell longitude (0.25° steps)")
print(f"  total_rainfall_mm    — monthly total rainfall in mm")
print(f"  valid_days           — days with non-missing data")
print(f"  expected_days        — total days in that month")
print(f"  clim_mean_mm         — long-run mean for this lat/lon/month")
print(f"  rainfall_anomaly_mm  — deviation from climatological mean")
print(f"  drought_threshold_mm — 20th percentile threshold")
print(f"  drought_flag         — 1 if below 20th percentile")
print(f"\nLoad in notebooks:")
print(f"  import pandas as pd")
print(f"  df = pd.read_parquet('{parquet_path}')")
