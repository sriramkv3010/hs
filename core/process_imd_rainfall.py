"""
IMD Rainfall GRD → Pandas DataFrame + CSV
==========================================
Files : Rainfall_ind{YYYY}_rfp25.grd  (2009–2021)
Format: Raw float32 binary, NO header
Grid  : 129 lat × 135 lon at 0.25° resolution
        Lat  6.5°N – 38.5°N  (129 values)
        Lon 66.5°E – 100.0°E (135 values)
Fill  : -999.0  (ocean / outside India)

Outputs (all in Dataset/processed/):
  imd_rainfall_monthly_2009_2021.csv
  imd_rainfall_monthly_2009_2021.parquet
  district_rainfall_mapping.csv
"""

import re, glob, json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from tqdm import tqdm

# ── CONFIGURATION ─────────────────────────────────────────────
BASE_DIR = Path("/Users/sriram/Desktop/HS")
RAINFALL_DIR = BASE_DIR / "Dataset/2.IMD/imd_rainfall"
SHAPEFILE_PATH = (
    BASE_DIR
    / "Dataset/Census/District Boundary Shapefile/India-Districts-2011Census.shp"
)
OUTPUT_DIR = BASE_DIR / "Dataset/processed"

N_LAT, N_LON = 129, 135
LAT_START, LAT_END = 6.5, 38.5
LON_START, LON_END = 66.5, 100.0
FILL_VALUE = -999.0
DROUGHT_PCT = 0.20
# ──────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LATS = np.round(np.linspace(LAT_START, LAT_END, N_LAT), 2)
LONS = np.round(np.linspace(LON_START, LON_END, N_LON), 2)


def section(t):
    print(f"\n{'─'*60}\n  {t}\n{'─'*60}")


def is_leap(y):
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def parse_year(filename):
    """
    Correctly extract the 4-digit year from filenames like:
      Rainfall_ind2009_rfp25.grd  →  2009
    Uses the FIRST 4-consecutive-digit group in the stem.
    Avoids the bug of taking digits[-4:] which picks up trailing numbers.
    """
    m = re.search(r"\b(19\d{2}|20\d{2})\b", Path(filename).stem)
    if m:
        return int(m.group(1))
    # fallback: first group of exactly 4 digits
    m = re.search(r"(\d{4})", Path(filename).stem)
    return int(m.group(1)) if m else None


# ══════════════════════════════════════════════════════════════
section("STEP 0 — Discover + validate files")
# ══════════════════════════════════════════════════════════════

grd_files = sorted(RAINFALL_DIR.glob("Rainfall_ind*_rfp25.grd"))
if not grd_files:
    grd_files = sorted(RAINFALL_DIR.glob("*.grd"))
if not grd_files:
    raise FileNotFoundError(f"No .grd files in {RAINFALL_DIR}")

file_info = []
for p in grd_files:
    year = parse_year(p.name)
    if year is None:
        print(f"  SKIP — cannot parse year: {p.name}")
        continue
    n_days = 366 if is_leap(year) else 365
    expected_size = n_days * N_LAT * N_LON * 4  # float32 = 4 bytes
    actual_size = p.stat().st_size
    size_ok = actual_size == expected_size
    if not size_ok:
        print(f"  SIZE MISMATCH  {p.name}")
        print(f"    actual   = {actual_size:,} bytes")
        print(
            f"    expected = {expected_size:,}  " f"({n_days}d × {N_LAT}×{N_LON} × 4)"
        )
        print(f"    values/day = {actual_size/4/n_days:.2f}  " f"(need {N_LAT*N_LON})")
    file_info.append({"path": p, "year": year, "n_days": n_days, "size_ok": size_ok})

files_df = pd.DataFrame(file_info).sort_values("year").reset_index(drop=True)
print(f"Files    : {len(files_df)}")
print(f"Years    : {files_df['year'].min()} – {files_df['year'].max()}")
print(f"Size OK  : {files_df['size_ok'].sum()} / {len(files_df)}")
print(files_df[["year", "n_days", "size_ok"]].to_string(index=False))

# Leap year check
leap_years = [y for y in files_df["year"] if is_leap(y)]
print(f"\nLeap years in dataset: {leap_years}  (these have 366 days — confirmed)")

if not files_df["size_ok"].all():
    bad = files_df[~files_df["size_ok"]]
    print(f"\n  {len(bad)} file(s) still have size mismatches.")
    print(f"  Update N_LAT/N_LON in CONFIG and rerun.")
    raise SystemExit(1)

print("\nAll files validated ✓")

# ══════════════════════════════════════════════════════════════
section("STEP 1 — Verify grid on one file")
# ══════════════════════════════════════════════════════════════

s = files_df.iloc[0]
raw_s = np.fromfile(s["path"], dtype=np.float32)
grid_s = raw_s.reshape(s["n_days"], N_LAT, N_LON).astype(np.float64)
grid_s = np.where(grid_s <= FILL_VALUE, np.nan, grid_s)
day1 = grid_s[0]
valid = day1[~np.isnan(day1)]

print(f"File      : {s['path'].name}  (year={s['year']}, days={s['n_days']})")
print(f"Grid      : {N_LAT} lat × {N_LON} lon")
print(f"Lat range : {LATS[0]}°N → {LATS[-1]}°N  step 0.25°")
print(f"Lon range : {LONS[0]}°E → {LONS[-1]}°E  step 0.25°")
print(f"Day 1     : {len(valid)} valid cells / {N_LAT*N_LON} total")
print(
    f"Day 1     : rainfall {valid.min():.1f}–{valid.max():.1f} mm  "
    f"mean {valid.mean():.2f} mm"
)

assert valid.min() >= 0, "Negative rainfall after masking — check FILL_VALUE"
assert valid.max() < 1500, "Rainfall > 1500mm/day — check FILL_VALUE"
print("Grid verified ✓")

# ══════════════════════════════════════════════════════════════
section("STEP 2 — Load shapefile + KDTree district matching")
# ══════════════════════════════════════════════════════════════

districts = gpd.read_file(SHAPEFILE_PATH)
print(f"Districts : {len(districts)}")
print(f"CRS       : {districts.crs}")
print(f"Columns   : {districts.columns.tolist()}")

if districts.crs and districts.crs.to_epsg() != 4326:
    districts = districts.to_crs(epsg=4326)
    print("Reprojected → WGS84")

districts = districts.copy()
districts["centroid_lat"] = districts.geometry.centroid.y
districts["centroid_lon"] = districts.geometry.centroid.x


def find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
        for col in df.columns:
            if c.lower() == col.lower():
                return col
    return None


id_c = find_col(districts, ["censuscode", "PC11_ID", "C_CODE11", "Dist_LGD", "DIST_ID"])
nm_c = find_col(districts, ["DISTRICT", "Dist_Name", "NAME_2", "district", "NAME"])
st_c = find_col(districts, ["STATE", "ST_NAME", "State_Name", "state", "NAME_1"])
print(f"\nDetected → id:'{id_c}'  name:'{nm_c}'  state:'{st_c}'")

if id_c is None:
    districts["district_id"] = districts.index.astype(str).str.zfill(4)
    id_c = "district_id"
if nm_c is None:
    districts["district_name"] = "District_" + districts.index.astype(str)
    nm_c = "district_name"
if st_c is None:
    districts["state_name"] = "Unknown"
    st_c = "state_name"

districts = districts.rename(
    columns={id_c: "district_id", nm_c: "district_name", st_c: "state_name"}
)

# KDTree over IMD grid centres
glat, glon = np.meshgrid(LATS, LONS, indexing="ij")
tree = cKDTree(np.column_stack([glat.ravel(), glon.ravel()]))

dists_deg, flat = tree.query(
    np.column_stack(
        [districts["centroid_lat"].values, districts["centroid_lon"].values]
    )
)
lat_idx = flat // N_LON
lon_idx = flat % N_LON

districts["lat_idx"] = lat_idx
districts["lon_idx"] = lon_idx
districts["grid_lat"] = LATS[lat_idx]
districts["grid_lon"] = LONS[lon_idx]
districts["distance_km"] = (dists_deg * 111.0).round(2)

print(f"\nKDTree matching:")
print(f"  Districts   : {len(districts)}")
print(f"  Mean dist   : {districts['distance_km'].mean():.2f} km")
print(
    f"  Max  dist   : {districts['distance_km'].max():.2f} km  "
    f"({districts.loc[districts['distance_km'].idxmax(),'district_name']})"
)

outside = (
    (districts["centroid_lat"] < LATS.min())
    | (districts["centroid_lat"] > LATS.max())
    | (districts["centroid_lon"] < LONS.min())
    | (districts["centroid_lon"] > LONS.max())
)
if outside.any():
    print(f"\n  WARNING: {outside.sum()} centroids outside IMD grid extent:")
    print(f"  {districts.loc[outside,'district_name'].tolist()}")

map_csv = OUTPUT_DIR / "district_rainfall_mapping.csv"
districts[
    [
        "district_id",
        "district_name",
        "state_name",
        "centroid_lat",
        "centroid_lon",
        "grid_lat",
        "grid_lon",
        "distance_km",
    ]
].to_csv(map_csv, index=False)
print(f"\n  Mapping saved: {map_csv}")

n_dist = len(districts)
lat_idxs = districts["lat_idx"].values
lon_idxs = districts["lon_idx"].values

# ══════════════════════════════════════════════════════════════
section("STEP 3 — Read GRD files → monthly district aggregates")
# ══════════════════════════════════════════════════════════════

records = []

for _, row in tqdm(files_df.iterrows(), total=len(files_df), desc="Years"):
    year = int(row["year"])
    n_days = int(row["n_days"])

    # Read binary → reshape → mask fill values
    raw = np.fromfile(row["path"], dtype=np.float32)
    grid = raw.reshape(n_days, N_LAT, N_LON).astype(np.float64)
    grid = np.where(grid <= FILL_VALUE, np.nan, grid)
    grid = np.where(grid < 0, np.nan, grid)

    dates = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
    months = dates.month.values

    # Extract all districts at once: (n_days, n_districts)
    dist_series = grid[:, lat_idxs, lon_idxs]

    for month in range(1, 13):
        mask = months == month
        n_in_month = int(mask.sum())
        if n_in_month == 0:
            continue

        md = dist_series[mask, :]  # (days_in_month, n_districts)
        total_mm = np.nansum(md, axis=0)
        rainy_days = np.nansum(md > 0, axis=0).astype(int)
        heavy_days = np.nansum(md > 50, axis=0).astype(int)
        max_1day = np.nanmax(md, axis=0)
        valid_days = np.sum(~np.isnan(md), axis=0).astype(int)

        for d in range(n_dist):
            records.append(
                {
                    "district_id": districts["district_id"].iat[d],
                    "district_name": districts["district_name"].iat[d],
                    "state_name": districts["state_name"].iat[d],
                    "year": year,
                    "month": month,
                    "total_rainfall_mm": round(float(total_mm[d]), 2),
                    "rainy_days": int(rainy_days[d]),
                    "heavy_rain_days": int(heavy_days[d]),
                    "max_1day_rainfall_mm": round(float(max_1day[d]), 2),
                    "valid_days": int(valid_days[d]),
                    "expected_days": n_in_month,
                    "grid_lat": float(districts["grid_lat"].iat[d]),
                    "grid_lon": float(districts["grid_lon"].iat[d]),
                    "distance_km": float(districts["distance_km"].iat[d]),
                }
            )

print(f"\nBuilding DataFrame from {len(records):,} records …")
df = pd.DataFrame(records)

df["data_completeness"] = (df["valid_days"] / df["expected_days"]).clip(0, 1).round(3)
df["mean_daily_rainfall_mm"] = (
    (df["total_rainfall_mm"] / df["valid_days"])
    .replace([np.inf, -np.inf], np.nan)
    .round(3)
)

print(f"Shape  : {df.shape}")
print(f"Memory : {df.memory_usage(deep=True).sum()/1e6:.1f} MB")

# ══════════════════════════════════════════════════════════════
section("STEP 4 — Climatology, anomaly, drought indicator")
# ══════════════════════════════════════════════════════════════

clim_mean = (
    df.groupby(["district_id", "month"])["total_rainfall_mm"]
    .mean()
    .rename("climatology_mean_mm")
    .reset_index()
)
df = df.merge(clim_mean, on=["district_id", "month"], how="left")

df["rainfall_anomaly_mm"] = (df["total_rainfall_mm"] - df["climatology_mean_mm"]).round(
    2
)
df["rainfall_anomaly_pct"] = (
    (df["rainfall_anomaly_mm"] / df["climatology_mean_mm"] * 100)
    .replace([np.inf, -np.inf], np.nan)
    .round(1)
)

drought_thr = (
    df.groupby(["district_id", "month"])["total_rainfall_mm"]
    .quantile(DROUGHT_PCT)
    .rename("drought_threshold_mm")
    .reset_index()
)
df = df.merge(drought_thr, on=["district_id", "month"], how="left")
df["drought"] = (df["total_rainfall_mm"] < df["drought_threshold_mm"]).astype(int)

print(f"Drought frequency: {df['drought'].mean()*100:.1f}% of district-months")

# ══════════════════════════════════════════════════════════════
section("STEP 5 — Season + water year")
# ══════════════════════════════════════════════════════════════

season_map = {
    1: "Winter",
    2: "Winter",
    3: "Pre-monsoon",
    4: "Pre-monsoon",
    5: "Pre-monsoon",
    6: "Kharif",
    7: "Kharif",
    8: "Kharif",
    9: "Kharif",
    10: "Kharif",
    11: "Rabi",
    12: "Rabi",
}
df["season"] = df["month"].map(season_map)
df["water_year"] = np.where(df["month"] >= 6, df["year"] + 1, df["year"])

# ══════════════════════════════════════════════════════════════
section("STEP 6 — Final column order + sort")
# ══════════════════════════════════════════════════════════════

final_cols = [
    "district_id",
    "district_name",
    "state_name",
    "year",
    "month",
    "season",
    "water_year",
    "total_rainfall_mm",
    "mean_daily_rainfall_mm",
    "rainy_days",
    "heavy_rain_days",
    "max_1day_rainfall_mm",
    "valid_days",
    "expected_days",
    "data_completeness",
    "climatology_mean_mm",
    "rainfall_anomaly_mm",
    "rainfall_anomaly_pct",
    "drought_threshold_mm",
    "drought",
    "grid_lat",
    "grid_lon",
    "distance_km",
]
df = df[final_cols].sort_values(["district_id", "year", "month"]).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════
section("STEP 7 — Save outputs")
# ══════════════════════════════════════════════════════════════

csv_path = OUTPUT_DIR / "imd_rainfall_monthly_2009_2021.csv"
df.to_csv(csv_path, index=False)
print(f"CSV     : {csv_path}")
print(f"  size  : {csv_path.stat().st_size/1e6:.1f} MB  |  rows: {len(df):,}")

pq_path = OUTPUT_DIR / "imd_rainfall_monthly_2009_2021.parquet"
df.to_parquet(pq_path, index=False, compression="snappy")
print(f"Parquet : {pq_path}")
print(f"  size  : {pq_path.stat().st_size/1e6:.1f} MB")

json.dump(
    {
        "grid": f"{N_LAT}×{N_LON}",
        "fill": FILL_VALUE,
        "lat": f"{LAT_START}–{LAT_END}°N 0.25°",
        "lon": f"{LON_START}–{LON_END}°E 0.25°",
        "drought_percentile": DROUGHT_PCT,
        "years": f"{df['year'].min()}–{df['year'].max()}",
        "n_districts": int(df["district_id"].nunique()),
        "total_rows": len(df),
        "drought_pct": round(float(df["drought"].mean() * 100), 2),
    },
    open(OUTPUT_DIR / "imd_rainfall_metadata.json", "w"),
    indent=2,
)
print(f"Metadata: {OUTPUT_DIR}/imd_rainfall_metadata.json")

# ══════════════════════════════════════════════════════════════
section("DONE — preview")
# ══════════════════════════════════════════════════════════════

print(f"\nColumn dtypes:\n{df.dtypes.to_string()}")
print(f"\nFirst 5 rows:\n{df.head().to_string(index=False)}")
print(f"\nJune 2015 — 5 districts:")
print(
    df[(df["year"] == 2015) & (df["month"] == 6)]
    .head(5)[
        [
            "district_name",
            "state_name",
            "total_rainfall_mm",
            "rainy_days",
            "rainfall_anomaly_mm",
            "drought",
        ]
    ]
    .to_string(index=False)
)

print(
    f"""
─────────────────────────────────────────────
Merge key into NFHS analysis dataset:
    ON  district_id + year + month

Key regression variables:
    total_rainfall_mm    — monthly total (confounder §5.2)
    rainfall_anomaly_mm  — deviation from district-month mean
    drought              — binary (1 = below {int(DROUGHT_PCT*100)}th percentile)
─────────────────────────────────────────────
"""
)
