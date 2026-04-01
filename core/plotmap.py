"""
INDIA DISTRICT MAP — Heat Effect on LBW
White background, correct India boundary (no PoK)
Run: python3 plot_india_map.py
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

try:
    import geopandas as gpd
except ImportError:
    print("Run: pip install geopandas")
    exit(1)

PROCESSED = "Dataset/processed"
SHP_PATH = (
    "Dataset/6.Census district boundaries/"
    "District Boundary Shapefile/India-Districts-2011Census.shp"
)
OUTPUT_DIR = "Dataset/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("INDIA DISTRICT MAP — LBW & Heat Effect")
print("=" * 60)

# ── 1. Load shapefile ──────────────────────────────────────────────────────────
print("\n[1] Loading shapefile...")
shp = gpd.read_file(SHP_PATH)
if shp.crs.to_epsg() != 4326:
    shp = shp.to_crs(epsg=4326)
shp["district_code"] = pd.to_numeric(shp["censuscode"], errors="coerce")

# Fix PoK / disputed regions — clip to India proper bounds
# India proper: lon 68-98, lat 6-37 (excludes PoK which is ~73-78 lon, 33-37 lat)
# We mark PoK districts as white (no data) by NOT filling them
# Indian territory bounding box
INDIA_BOUNDS = {"minx": 68.0, "maxx": 97.5, "miny": 6.5, "maxy": 37.5}
shp = shp.cx[
    INDIA_BOUNDS["minx"] : INDIA_BOUNDS["maxx"],
    INDIA_BOUNDS["miny"] : INDIA_BOUNDS["maxy"],
]
print(f"    Districts: {len(shp)}")

# ── 2. Load data ───────────────────────────────────────────────────────────────
print("\n[2] Loading dataset...")
raw = pd.read_parquet(f"{PROCESSED}/final_analytical_dataset.parquet")
for c in [
    "lbw",
    "t3_hot_days_35",
    "district",
    "wealth_q",
    "birth_month",
    "birth_year",
    "wave",
]:
    if c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").astype(float)

# ── 3. District stats ──────────────────────────────────────────────────────────
print("\n[3] Computing district statistics...")

dist_lbw = (
    raw[raw["lbw"].notna()]
    .groupby("district")
    .agg(
        lbw_rate=("lbw", "mean"),
        n_births=("lbw", "count"),
        mean_hot_days=("t3_hot_days_35", "mean"),
    )
    .reset_index()
)
dist_lbw = dist_lbw[dist_lbw["n_births"] >= 30]
dist_lbw["lbw_pct"] = dist_lbw["lbw_rate"] * 100
print(f"    Districts with LBW data: {len(dist_lbw)}")


# Heat effect per district: hot years vs cool years LBW difference
def heat_effect(group):
    g = group.dropna(subset=["lbw", "t3_hot_days_35"])
    if len(g) < 20:
        return np.nan
    med = g["t3_hot_days_35"].median()
    hot = g[g["t3_hot_days_35"] > med]["lbw"].mean()
    cool = g[g["t3_hot_days_35"] <= med]["lbw"].mean()
    return (hot - cool) * 100


dist_eff = (
    raw[raw["lbw"].notna() & raw["t3_hot_days_35"].notna()]
    .groupby("district")
    .apply(heat_effect)
    .reset_index()
    .rename(columns={0: "heat_effect_pp"})
)
dist_eff = dist_eff[dist_eff["heat_effect_pp"].notna()]

dist_hot = (
    raw[raw["t3_hot_days_35"].notna()]
    .groupby("district")
    .agg(mean_hot_days=("t3_hot_days_35", "mean"), n=("t3_hot_days_35", "count"))
    .reset_index()
)
dist_hot = dist_hot[dist_hot["n"] >= 10]

# ── 4. Merge ───────────────────────────────────────────────────────────────────
print("\n[4] Merging...")
shp = shp.merge(
    dist_lbw[["district", "lbw_pct", "n_births"]],
    left_on="district_code",
    right_on="district",
    how="left",
)
shp = shp.merge(
    dist_eff,
    left_on="district_code",
    right_on="district",
    how="left",
    suffixes=("", "_e"),
)
shp = shp.merge(
    dist_hot[["district", "mean_hot_days"]],
    left_on="district_code",
    right_on="district",
    how="left",
    suffixes=("", "_h"),
)
print(f"    LBW data: {shp['lbw_pct'].notna().sum()} districts")
print(f"    Heat effect: {shp['heat_effect_pp'].notna().sum()} districts")

# ── 5. Colormaps ───────────────────────────────────────────────────────────────
# LBW: white → light orange → deep red
cmap_lbw = LinearSegmentedColormap.from_list(
    "lbw",
    ["#FFFFFF", "#FEE5D0", "#FCAE91", "#FB6A4A", "#DE2D26", "#A50F15", "#67000D"],
    N=256,
)

# Heat effect: white centre, red for positive effect
cmap_eff = LinearSegmentedColormap.from_list(
    "effect", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#CA0020", "#67001F"], N=256
)

# Hot days: white → amber → deep orange
cmap_hot = LinearSegmentedColormap.from_list(
    "hotdays", ["#FFFFFF", "#FFF3CC", "#FFD700", "#FF8C00", "#CC3300", "#7B0000"], N=256
)

# ── 6. MAP 1: LBW Rate ────────────────────────────────────────────────────────
print("\n[5] Map 1: LBW Rate...")
fig, ax = plt.subplots(1, 1, figsize=(13, 15))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# No-data districts = light grey
shp[shp["lbw_pct"].isna()].plot(
    ax=ax, color="#F0F0F0", edgecolor="#CCCCCC", linewidth=0.3
)

# Data districts
shp_d = shp[shp["lbw_pct"].notna()]
v1 = shp_d["lbw_pct"].quantile(0.05)
v2 = shp_d["lbw_pct"].quantile(0.95)
shp_d.plot(
    column="lbw_pct",
    ax=ax,
    cmap=cmap_lbw,
    vmin=v1,
    vmax=v2,
    edgecolor="#999999",
    linewidth=0.15,
    legend=False,
)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap_lbw, norm=plt.Normalize(vmin=v1, vmax=v2))
sm.set_array([])
cbar = plt.colorbar(
    sm, ax=ax, fraction=0.025, pad=0.02, aspect=35, orientation="vertical"
)
cbar.set_label("LBW Rate (%)", fontsize=12, labelpad=10, color="#333333")
cbar.ax.tick_params(labelsize=10, color="#555555")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#333333")

# Annotations
ax.set_title(
    "Low Birth Weight Rate by District\nIndia — NFHS-4 + NFHS-5 (2009–2021)",
    fontsize=16,
    fontweight="bold",
    pad=15,
    color="#1A1A1A",
)
ax.text(
    0.02,
    0.02,
    f"Districts: {shp_d.shape[0]}\n"
    f"National mean: {shp_d['lbw_pct'].mean():.1f}%\n"
    f"Range: {shp_d['lbw_pct'].min():.1f}–{shp_d['lbw_pct'].max():.1f}%\n"
    f"Grey = insufficient data",
    transform=ax.transAxes,
    fontsize=9,
    color="#555555",
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#DDDDDD"
    ),
)

# Legend patches
legend_elements = [
    mpatches.Patch(facecolor="#DE2D26", edgecolor="#999999", label="High LBW (>20%)"),
    mpatches.Patch(
        facecolor="#FCAE91", edgecolor="#999999", label="Medium LBW (15-20%)"
    ),
    mpatches.Patch(facecolor="#FEE5D0", edgecolor="#999999", label="Low LBW (<15%)"),
    mpatches.Patch(facecolor="#F0F0F0", edgecolor="#CCCCCC", label="Insufficient data"),
]
ax.legend(
    handles=legend_elements,
    loc="lower right",
    fontsize=9,
    framealpha=0.9,
    edgecolor="#DDDDDD",
)
ax.axis("off")
plt.tight_layout()
out1 = f"{OUTPUT_DIR}/figure_india_district_lbw.png"
plt.savefig(out1, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"    Saved: {out1}")

# ── 7. MAP 2: Heat Effect ──────────────────────────────────────────────────────
print("\n[6] Map 2: Heat Effect on LBW...")
fig, ax = plt.subplots(1, 1, figsize=(13, 15))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

shp[shp["heat_effect_pp"].isna()].plot(
    ax=ax, color="#F0F0F0", edgecolor="#CCCCCC", linewidth=0.3
)

shp_e = shp[shp["heat_effect_pp"].notna()]
vmax_e = max(
    abs(shp_e["heat_effect_pp"].quantile(0.05)),
    abs(shp_e["heat_effect_pp"].quantile(0.95)),
)
shp_e.plot(
    column="heat_effect_pp",
    ax=ax,
    cmap=cmap_eff,
    vmin=-vmax_e,
    vmax=vmax_e,
    edgecolor="#999999",
    linewidth=0.15,
    legend=False,
)

sm2 = plt.cm.ScalarMappable(
    cmap=cmap_eff, norm=plt.Normalize(vmin=-vmax_e, vmax=vmax_e)
)
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=ax, fraction=0.025, pad=0.02, aspect=35)
cbar2.set_label(
    "Heat Effect on LBW\n(pp: hot vs cool years within district)",
    fontsize=11,
    labelpad=10,
    color="#333333",
)
cbar2.ax.tick_params(labelsize=10)
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="#333333")

pos_d = (shp_e["heat_effect_pp"] > 0).sum()
ax.set_title(
    "Ecological Heat Effect on Low Birth Weight\n"
    "Red = Stronger heat→LBW link | Blue = Weaker/null | White = Neutral",
    fontsize=15,
    fontweight="bold",
    pad=15,
    color="#1A1A1A",
)
ax.text(
    0.02,
    0.02,
    f"Districts with positive effect: {pos_d}/{len(shp_e)} ({pos_d/len(shp_e)*100:.0f}%)\n"
    f"Mean effect: {shp_e['heat_effect_pp'].mean():+.2f}pp\n"
    f"Grey = insufficient data",
    transform=ax.transAxes,
    fontsize=9,
    color="#555555",
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#DDDDDD"
    ),
)

legend_elements2 = [
    mpatches.Patch(
        facecolor="#CA0020", edgecolor="#999999", label="Strong positive effect"
    ),
    mpatches.Patch(facecolor="#F7F7F7", edgecolor="#999999", label="No effect"),
    mpatches.Patch(facecolor="#2166AC", edgecolor="#999999", label="Negative/null"),
    mpatches.Patch(facecolor="#F0F0F0", edgecolor="#CCCCCC", label="Insufficient data"),
]
ax.legend(
    handles=legend_elements2,
    loc="lower right",
    fontsize=9,
    framealpha=0.9,
    edgecolor="#DDDDDD",
)
ax.axis("off")
plt.tight_layout()
out2 = f"{OUTPUT_DIR}/figure_india_district_heat_effect.png"
plt.savefig(out2, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"    Saved: {out2}")

# ── 8. MAP 3: Side-by-side ─────────────────────────────────────────────────────
print("\n[7] Map 3: Side-by-side ecology + health...")
fig, axes = plt.subplots(1, 2, figsize=(22, 14))
fig.patch.set_facecolor("white")

for ax in axes:
    ax.set_facecolor("white")

# Left: Hot days exposure
ax = axes[0]
shp[shp["mean_hot_days"].isna()].plot(
    ax=ax, color="#F0F0F0", edgecolor="#CCCCCC", linewidth=0.3
)
shp_hot2 = shp[shp["mean_hot_days"].notna()]
vh1 = shp_hot2["mean_hot_days"].quantile(0.05)
vh2 = shp_hot2["mean_hot_days"].quantile(0.95)
shp_hot2.plot(
    column="mean_hot_days",
    ax=ax,
    cmap=cmap_hot,
    vmin=vh1,
    vmax=vh2,
    edgecolor="#BBBBBB",
    linewidth=0.15,
)
smh = plt.cm.ScalarMappable(cmap=cmap_hot, norm=plt.Normalize(vmin=vh1, vmax=vh2))
smh.set_array([])
cbh = plt.colorbar(smh, ax=ax, fraction=0.025, pad=0.02, aspect=30)
cbh.set_label("Mean T3 Hot Days >35°C", fontsize=11, labelpad=8, color="#333333")
cbh.ax.tick_params(labelsize=9)
plt.setp(cbh.ax.yaxis.get_ticklabels(), color="#333333")
ax.set_title(
    "A. Ecological Exposure\nMean T3 Days >35°C per District (IMD 2009–2021)",
    fontsize=13,
    fontweight="bold",
    color="#1A1A1A",
    pad=12,
)
ax.text(
    0.02,
    0.02,
    f"White = cool districts\nDeep red = hottest districts\n" f"Grey = no climate data",
    transform=ax.transAxes,
    fontsize=9,
    color="#555555",
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#DDDDDD"
    ),
)
ax.axis("off")

# Right: LBW rate
ax2 = axes[1]
shp[shp["lbw_pct"].isna()].plot(
    ax=ax2, color="#F0F0F0", edgecolor="#CCCCCC", linewidth=0.3
)
shp_l2 = shp[shp["lbw_pct"].notna()]
vl1 = shp_l2["lbw_pct"].quantile(0.05)
vl2 = shp_l2["lbw_pct"].quantile(0.95)
shp_l2.plot(
    column="lbw_pct",
    ax=ax2,
    cmap=cmap_lbw,
    vmin=vl1,
    vmax=vl2,
    edgecolor="#BBBBBB",
    linewidth=0.15,
)
sml = plt.cm.ScalarMappable(cmap=cmap_lbw, norm=plt.Normalize(vmin=vl1, vmax=vl2))
sml.set_array([])
cbl = plt.colorbar(sml, ax=ax2, fraction=0.025, pad=0.02, aspect=30)
cbl.set_label("LBW Rate (%)", fontsize=11, labelpad=8, color="#333333")
cbl.ax.tick_params(labelsize=9)
plt.setp(cbl.ax.yaxis.get_ticklabels(), color="#333333")
ax2.set_title(
    "B. Health Outcome\nLow Birth Weight Rate by District (NFHS-4 + NFHS-5)",
    fontsize=13,
    fontweight="bold",
    color="#1A1A1A",
    pad=12,
)

# Compute correlation for annotation
merged_corr = shp[shp["mean_hot_days"].notna() & shp["lbw_pct"].notna()]
corr_val = merged_corr["mean_hot_days"].corr(merged_corr["lbw_pct"])
ax2.text(
    0.02,
    0.02,
    f"White = lowest LBW\nDeep red = highest LBW\n"
    f"r = {corr_val:.3f} (heat vs LBW across districts)",
    transform=ax2.transAxes,
    fontsize=9,
    color="#555555",
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#DDDDDD"
    ),
)
ax2.axis("off")

fig.suptitle(
    "Ecology → Health: Thermal Stress and Low Birth Weight Across India\n"
    "Districts with greater heat exposure show higher LBW rates  "
    f"(district correlation r = {corr_val:.3f})",
    fontsize=15,
    fontweight="bold",
    color="#1A1A1A",
    y=1.02,
)

# Bottom note
fig.text(
    0.5,
    -0.01,
    "Source: IMD Gridded Temperature Data + NFHS-4 & NFHS-5 | "
    "White/light = low intensity | Deep red = high intensity | Grey = insufficient data",
    ha="center",
    color="#777777",
    fontsize=10,
)

plt.tight_layout(rect=[0, 0, 1, 1])
out3 = f"{OUTPUT_DIR}/figure_india_ecology_health_map.png"
plt.savefig(out3, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"    Saved: {out3}")

print(f"\n{'='*60}")
print("DONE — 3 maps saved to Dataset/results/")
print("=" * 60)
print(f"  1. figure_india_district_lbw.png      — LBW rate choropleth")
print(f"  2. figure_india_district_heat_effect.png — Heat effect strength")
print(f"  3. figure_india_ecology_health_map.png — Side-by-side for paper")
print(f"\n  District correlation (heat vs LBW): r = {corr_val:.3f}")
