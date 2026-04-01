"""
IMD Tmax GRD Combiner — FINAL CORRECT VERSION
===============================================
Files : Maxtemp_MaxT_{YYYY}.GRD (2009-2021)
Grid  : 31 lat × 31 lon at 1.0° resolution
        Lat  7.5°N – 37.5°N
        Lon 67.5°E – 97.5°E
Fill  : 99.9 (missing/ocean — NOT -999)

Run: python3 combine_imd_tmax.py
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TMAX_DIR = Path.home() / "Desktop/HS/Dataset/2.IMD/imd_tmax"
OUTPUT_DIR = Path.home() / "Desktop/HS/Dataset/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_LAT, N_LON = 31, 31
LAT_START, LAT_END = 7.5, 37.5
LON_START, LON_END = 67.5, 97.5

LATS = np.round(np.linspace(LAT_START, LAT_END, N_LAT), 1)
LONS = np.round(np.linspace(LON_START, LON_END, N_LON), 1)

# CORRECTED fill values — IMD Tmax uses 99.9 not -999
FILL_HIGH = 99.0  # anything >= 99.0 is missing (fill = 99.9)
FILL_LOW = -9.0  # anything <= -9.0 is missing (fill = -99.9 in some files)
VALID_MIN = -5.0  # coldest realistic India Tmax (Himalayan winters)
VALID_MAX = 52.0  # hottest realistic India Tmax ever recorded ~51°C


# ── HELPERS ────────────────────────────────────────────────────────────────────
def is_leap(y):
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def parse_year(filepath):
    stem = Path(filepath).stem
    matches = re.findall(r"(200[0-9]|201[0-9]|202[0-5])", stem)
    if matches:
        return int(matches[0])
    for m in re.findall(r"(\d{4})", stem):
        y = int(m)
        if 1990 <= y <= 2030:
            return y
    return None


def mask_fill(arr):
    """Replace fill values and implausible temperatures with NaN."""
    arr = arr.copy().astype(np.float64)
    arr[arr >= FILL_HIGH] = np.nan  # 99.9 fill value
    arr[arr <= FILL_LOW] = np.nan  # -99.9 fill value
    arr[arr > VALID_MAX] = np.nan  # above record India temperature
    arr[arr < VALID_MIN] = np.nan  # below plausible India temperature
    return arr


# ── FIND + VALIDATE FILES ──────────────────────────────────────────────────────
print("=" * 55)
print("IMD TMAX GRD → CSV COMBINER  (31×31, 1° grid)")
print("=" * 55)

grd_files = sorted(TMAX_DIR.glob("*.GRD")) or sorted(TMAX_DIR.glob("*.grd"))
print(f"\nFound {len(grd_files)} files in:\n  {TMAX_DIR}")

file_info = []
print(f"\nValidating:")
for p in grd_files:
    year = parse_year(p.name)
    if year is None:
        print(f"  SKIP: {p.name}")
        continue
    n_days = 366 if is_leap(year) else 365
    expected_size = n_days * N_LAT * N_LON * 4
    actual_size = p.stat().st_size
    size_ok = actual_size == expected_size
    print(
        f"  {p.name} → year={year}  days={n_days}  " f"{'✓' if size_ok else 'MISMATCH'}"
    )
    file_info.append({"path": p, "year": year, "n_days": n_days, "size_ok": size_ok})

if not file_info:
    raise SystemExit(f"No files found in {TMAX_DIR}")

file_info = sorted(file_info, key=lambda x: x["year"])
years = [f["year"] for f in file_info]
n_bad = sum(1 for f in file_info if not f["size_ok"])
print(f"\nYears: {years[0]}–{years[-1]}  Files: {len(file_info)}  Bad: {n_bad}")

if n_bad > 0:
    raise SystemExit("Fix size mismatches before continuing.")

# ── INSPECT FIRST FILE TO CONFIRM FILL VALUE ──────────────────────────────────
print(f"\n--- Inspecting first file ---")
s = file_info[0]
raw = np.fromfile(s["path"], dtype=np.float32)
g = raw.reshape(s["n_days"], N_LAT, N_LON)
day1_raw = g[0].ravel()

print(f"Day 1 raw stats (BEFORE masking):")
print(
    f"  min={day1_raw.min():.2f}  max={day1_raw.max():.2f}  "
    f"mean={day1_raw.mean():.2f}"
)
print(f"  Unique high values: " f"{sorted(set(day1_raw[day1_raw > 50].round(1)))[:10]}")
print(f"  Unique low values:  " f"{sorted(set(day1_raw[day1_raw < 0].round(1)))[:10]}")

# Apply masking
day1_clean = mask_fill(day1_raw.reshape(N_LAT, N_LON)).ravel()
valid = day1_clean[~np.isnan(day1_clean)]
print(f"\nDay 1 stats (AFTER masking fill values >= {FILL_HIGH}):")
print(f"  Valid cells: {len(valid)} / {N_LAT*N_LON}")
print(f"  Tmax range:  {valid.min():.1f}°C – {valid.max():.1f}°C")
print(f"  Tmax mean:   {valid.mean():.1f}°C")

# Confirm values are realistic
assert (
    valid.max() <= VALID_MAX
), f"Still too high after masking: {valid.max():.1f}°C. Adjust FILL_HIGH."
assert (
    valid.min() >= VALID_MIN
), f"Still too low after masking: {valid.min():.1f}°C. Adjust FILL_LOW."
assert len(valid) > 0, "No valid cells after masking — check fill values."
print("Grid verified ✓")

# ── PROCESS ALL FILES → MONTHLY AGGREGATES ────────────────────────────────────
print(f"\nProcessing {len(file_info)} years...")

all_records = []

for fi in tqdm(file_info, desc="Years"):
    year = fi["year"]
    n_days = fi["n_days"]

    raw = np.fromfile(fi["path"], dtype=np.float32)
    grid = mask_fill(raw.reshape(n_days, N_LAT, N_LON))

    dates = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
    months = dates.month.values

    for month in range(1, 13):
        mask = months == month
        n_days_in_month = int(mask.sum())
        if n_days_in_month == 0:
            continue

        sl = grid[mask, :, :]  # (days_in_month, 31, 31)

        tmax_mean = np.nanmean(sl, axis=0)
        tmax_max = np.nanmax(sl, axis=0)
        hot_35 = np.nansum(sl > 35, axis=0).astype(int)
        hot_33 = np.nansum(sl > 33, axis=0).astype(int)
        valid_days = np.sum(~np.isnan(sl), axis=0).astype(int)

        tmax_mean = np.where(valid_days == 0, np.nan, tmax_mean)
        tmax_max = np.where(valid_days == 0, np.nan, tmax_max)

        for i in range(N_LAT):
            for j in range(N_LON):
                if np.isnan(tmax_mean[i, j]):
                    continue
                all_records.append(
                    {
                        "year": year,
                        "month": month,
                        "lat": float(LATS[i]),
                        "lon": float(LONS[j]),
                        "tmax_mean": round(float(tmax_mean[i, j]), 3),
                        "tmax_max": round(float(tmax_max[i, j]), 3),
                        "hot_days_35": int(hot_35[i, j]),
                        "hot_days_33": int(hot_33[i, j]),
                        "valid_days": int(valid_days[i, j]),
                        "expected_days": n_days_in_month,
                    }
                )

print(f"\nTotal records: {len(all_records):,}")
df = pd.DataFrame(all_records)

# ── CLIMATOLOGY + ANOMALY ─────────────────────────────────────────────────────
print("Computing climatology and anomaly...")

clim = (
    df.groupby(["lat", "lon", "month"])["tmax_mean"]
    .mean()
    .reset_index()
    .rename(columns={"tmax_mean": "tmax_clim_mean"})
)
df = df.merge(clim, on=["lat", "lon", "month"], how="left")
df["tmax_anomaly"] = (df["tmax_mean"] - df["tmax_clim_mean"]).round(3)

anom_sd = (
    df.groupby(["lat", "lon", "month"])["tmax_anomaly"]
    .std()
    .reset_index()
    .rename(columns={"tmax_anomaly": "tmax_anomaly_sd"})
)
df = df.merge(anom_sd, on=["lat", "lon", "month"], how="left")
df["tmax_anomaly_std"] = (df["tmax_anomaly"] / df["tmax_anomaly_sd"]).round(3)

# ── QUALITY CHECKS ────────────────────────────────────────────────────────────
print("\n--- Quality Checks ---")
print(f"Shape:          {df.shape}")
print(f"Years:          {sorted(df['year'].unique().tolist())}")
print(f"Grid cells/mo:  {df[(df.year==2015) & (df.month==1)].shape[0]}")
print(
    f"Tmax range:     {df['tmax_mean'].min():.1f} – " f"{df['tmax_mean'].max():.1f}°C"
)
print(f"Anomaly mean:   {df['tmax_anomaly'].mean():.4f}°C (should be ~0)")
print(f"Missing tmax:   {df['tmax_mean'].isna().mean():.1%}")

print(f"\nMean Tmax by month (°C):")
by_m = df.groupby("month")["tmax_mean"].mean().round(1)
for m, v in by_m.items():
    bar = "█" * int((v - 10) / 1.5)
    print(f"  Month {m:2d}: {v:5.1f}°C  {bar}")

print(f"\nMean hot days >35°C by month:")
by_h = df.groupby("month")["hot_days_35"].mean().round(1)
for m, v in by_h.items():
    bar = "█" * int(v)
    print(f"  Month {m:2d}: {v:4.1f} days  {bar}")

print(f"\nAnomaly distribution:")
print(df["tmax_anomaly"].describe().round(3))

# ── SAVE ──────────────────────────────────────────────────────────────────────
print("\n--- Saving ---")

csv_path = OUTPUT_DIR / "imd_tmax_combined.csv"
df.to_csv(csv_path, index=False)
print(f"CSV:     {csv_path}")
print(f"  Rows:  {len(df):,}")
print(f"  Size:  {csv_path.stat().st_size/1e6:.1f} MB")

parquet_path = OUTPUT_DIR / "imd_tmax_combined.parquet"
df.to_parquet(parquet_path, index=False, compression="snappy")
print(f"Parquet: {parquet_path}")
print(f"  Size:  {parquet_path.stat().st_size/1e6:.1f} MB")

print("\n" + "=" * 55)
print("DONE")
print("=" * 55)
print(f"Years:   {df['year'].min()} – {df['year'].max()}")
print(f"Rows:    {len(df):,}")
print(f"Columns: {df.columns.tolist()}")
print(
    f"""
Column descriptions:
  year              — calendar year
  month             — calendar month (1–12)
  lat               — latitude  (1° steps, 7.5–37.5°N)
  lon               — longitude (1° steps, 67.5–97.5°E)
  tmax_mean         — mean daily Tmax for month (°C)
  tmax_max          — highest single-day Tmax (°C)
  hot_days_35       — days Tmax > 35°C  ← PRIMARY heat variable
  hot_days_33       — days Tmax > 33°C  ← robustness check
  valid_days        — non-missing days
  expected_days     — total days in month
  tmax_clim_mean    — long-run monthly mean (°C)
  tmax_anomaly      — deviation from normal  ← KEY REGRESSION VARIABLE
  tmax_anomaly_sd   — SD of anomaly
  tmax_anomaly_std  — anomaly in SD units

Load: df = pd.read_parquet('{parquet_path}')
"""
)
