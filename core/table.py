"""
TABLE 1 — DESCRIPTIVE STATISTICS
Generates publication-ready Table 1 matching the exact format in the screenshot.
Fills with actual values from final_analytical_dataset.parquet.

Run: python3 generate_table1.py
Output: Dataset/results/table1_publication.csv
        Dataset/results/table1_publication.png  (formatted image)
        Dataset/results/table1_latex.tex        (LaTeX code)
"""

import os, warnings
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

PROCESSED = "Dataset/processed"
OUTPUT_DIR = "Dataset/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
raw = pd.read_parquet(f"{PROCESSED}/final_analytical_dataset.parquet")

# ── Convert all needed columns to float ──────────────────────────────────────
cols = [
    "lbw",
    "birthweight_g",
    "t3_hot_days_35",
    "t3_hot_days_33",
    "t3_tmax_anomaly",
    "maternal_age",
    "education",
    "wealth_q",
    "rural",
    "birth_order",
    "has_electricity",
    "good_housing",
    "anaemic",
    "bmi",
    "neonatal_death",
    "infant_death",
    "housing_quality_score",
    "clean_fuel",
    "wave",
    "birth_year",
    "t3_hot_days_35",
    "t1_hot_days_35",
    "t2_hot_days_35",
    "t3_rainfall_mm",
]
for c in cols:
    if c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").astype(float)

# ── Build the LBW analytical sample (matching main analysis) ─────────────────
raw["state"] = pd.to_numeric(
    raw.get("v024", pd.Series(dtype=float)), errors="coerce"
).astype(float)
lbw_s = (
    raw[
        raw["lbw"].notna()
        & raw["t3_hot_days_35"].notna()
        & raw["state"].notna()
        & raw["birth_year"].between(2009, 2021)
    ]
    .copy()
    .reset_index(drop=True)
)

full_s = (
    raw[
        raw["neonatal_death"].notna()
        & raw["t3_hot_days_35"].notna()
        & raw["state"].notna()
        & raw["birth_year"].between(2009, 2021)
    ]
    .copy()
    .reset_index(drop=True)
)

print(f"LBW sample: {len(lbw_s):,}")
print(f"Mortality sample: {len(full_s):,}")


# ── Function to compute row stats ─────────────────────────────────────────────
def stats_row(df, col, label, group):
    s = (
        pd.to_numeric(df[col], errors="coerce").dropna()
        if col in df.columns
        else pd.Series(dtype=float)
    )
    if len(s) < 10:
        return None
    return {
        "Group": group,
        "Variable": label,
        "N": len(s),
        "Mean": float(s.mean()),
        "SD": float(s.std()),
        "p25": float(s.quantile(0.25)),
        "Median": float(s.median()),
        "p75": float(s.quantile(0.75)),
    }


# ── Build table rows in exact order matching screenshot ───────────────────────
rows = []

# OUTCOME VARIABLES
rows.append(stats_row(lbw_s, "lbw", "Low Birth Weight (LBW)", "Outcome Variables"))
rows.append(
    stats_row(lbw_s, "birthweight_g", "Birth Weight (grams)", "Outcome Variables")
)
rows.append(
    stats_row(full_s, "neonatal_death", "Neonatal Mortality", "Outcome Variables")
)
rows.append(stats_row(full_s, "infant_death", "Infant Mortality", "Outcome Variables"))

# TREATMENT VARIABLES
rows.append(
    stats_row(lbw_s, "t3_hot_days_35", "T3 Hot Days > 35°C", "Treatment Variables")
)
rows.append(
    stats_row(lbw_s, "t3_hot_days_33", "T3 Hot Days > 33°C", "Treatment Variables")
)
rows.append(
    stats_row(lbw_s, "t3_tmax_anomaly", "T3 Tmax Anomaly (°C)", "Treatment Variables")
)

# CONTROL VARIABLES
rows.append(
    stats_row(lbw_s, "maternal_age", "Maternal Age (years)", "Control Variables")
)
rows.append(stats_row(lbw_s, "education", "Education (0–3)", "Control Variables"))
rows.append(stats_row(lbw_s, "wealth_q", "Wealth Quintile (1–5)", "Control Variables"))
rows.append(stats_row(lbw_s, "rural", "Rural (1 = yes)", "Control Variables"))
rows.append(stats_row(lbw_s, "birth_order", "Birth Order", "Control Variables"))

# ADAPTIVE CAPACITY
rows.append(stats_row(lbw_s, "has_electricity", "Has Electricity", "Adaptive Capacity"))
rows.append(stats_row(lbw_s, "good_housing", "Good Housing", "Adaptive Capacity"))
rows.append(stats_row(lbw_s, "anaemic", "Anaemic (1 = yes)", "Adaptive Capacity"))
rows.append(stats_row(lbw_s, "bmi", "Maternal BMI (kg/m²)", "Adaptive Capacity"))
rows.append(
    stats_row(
        lbw_s, "housing_quality_score", "Housing Quality Score", "Adaptive Capacity"
    )
)

# Filter out None rows
rows = [r for r in rows if r is not None]
df_table = pd.DataFrame(rows)


# ── Format for display ────────────────────────────────────────────────────────
def fmt_n(x):
    return f"{int(x):,}"


def fmt_val(x, col_name, var_name):
    """Smart formatting: integers for counts/binary, 2-4 decimals for others"""
    if var_name in [
        "Low Birth Weight (LBW)",
        "Neonatal Mortality",
        "Infant Mortality",
        "Rural (1 = yes)",
        "Has Electricity",
        "Good Housing",
        "Anaemic (1 = yes)",
    ]:
        return f"{x:.4f}"  # show as proportion
    elif var_name in ["Birth Weight (grams)"]:
        return f"{x:.0f}"  # whole numbers for grams
    elif var_name in ["T3 Hot Days > 35°C", "T3 Hot Days > 33°C"]:
        return f"{x:.1f}"  # 1 decimal for days
    elif var_name in ["T3 Tmax Anomaly (°C)", "Maternal BMI (kg/m²)"]:
        return f"{x:.2f}"
    elif var_name in ["Maternal Age (years)", "Birth Order"]:
        return f"{x:.1f}"
    elif var_name in [
        "Education (0–3)",
        "Wealth Quintile (1–5)",
        "Housing Quality Score",
    ]:
        return f"{x:.1f}"
    else:
        return f"{x:.4f}"


# Build display table
display_rows = []
for _, r in df_table.iterrows():
    vn = r["Variable"]
    display_rows.append(
        {
            "Variable": vn,
            "N": fmt_n(r["N"]),
            "Mean": fmt_val(r["Mean"], "Mean", vn),
            "SD": fmt_val(r["SD"], "SD", vn),
            "p25": fmt_val(r["p25"], "p25", vn),
            "Median": fmt_val(r["Median"], "Median", vn),
            "p75": fmt_val(r["p75"], "p75", vn),
            "Group": r["Group"],
        }
    )
df_display = pd.DataFrame(display_rows)

# Save CSV
df_table.to_csv(f"{OUTPUT_DIR}/table1_publication.csv", index=False)
print(f"\nTable 1 data saved.")

# Print to console
print(f"\n{'='*80}")
print(f"TABLE 1: DESCRIPTIVE STATISTICS")
print(f"{'='*80}")
print(
    f"{'Variable':<35} {'N':>10} {'Mean':>8} {'SD':>7} {'p25':>7} {'Median':>8} {'p75':>7}"
)
print(f"{'─'*80}")
current_group = None
for _, r in df_display.iterrows():
    if r["Group"] != current_group:
        current_group = r["Group"]
        print(f"\n{current_group}")
    print(
        f"  {r['Variable']:<33} {r['N']:>10} {r['Mean']:>8} {r['SD']:>7} "
        f"{r['p25']:>7} {r['Median']:>8} {r['p75']:>7}"
    )

# ── Publication-quality figure (matching screenshot style) ────────────────────
print("\nGenerating publication-ready table image...")

fig_h = 0.42 * (len(df_display) + 6)
fig, ax = plt.subplots(figsize=(11, fig_h))
ax.set_xlim(0, 10)
ax.set_ylim(0, len(df_display) + 5)
ax.axis("off")
fig.patch.set_facecolor("white")

# Title
ax.text(
    5,
    len(df_display) + 4.2,
    "Table 1: Descriptive Statistics",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
    fontfamily="serif",
)

# Header line
y_header = len(df_display) + 3.3
ax.axhline(y=y_header + 0.5, xmin=0.02, xmax=0.98, color="black", lw=1.2)
ax.axhline(y=y_header - 0.3, xmin=0.02, xmax=0.98, color="black", lw=0.8)

# Column headers
col_x = [0.15, 4.0, 5.3, 6.5, 7.6, 8.7, 9.75]
col_labels = ["Variable", "N", "Mean", "SD", "p25", "Median", "p75"]
col_align = ["left", "right", "right", "right", "right", "right", "right"]
for cx, cl, ca in zip(col_x, col_labels, col_align):
    ax.text(
        cx,
        y_header,
        cl,
        ha=ca,
        va="center",
        fontsize=10,
        fontweight="normal",
        fontfamily="serif",
    )

# Rows
current_group = None
y = y_header - 0.8
row_data = df_display.to_dict("records")

for r in row_data:
    # Group header
    if r["Group"] != current_group:
        current_group = r["Group"]
        ax.text(
            0.15,
            y,
            current_group,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            fontfamily="serif",
        )
        y -= 0.55

    # Data row
    vals = [r["Variable"], r["N"], r["Mean"], r["SD"], r["p25"], r["Median"], r["p75"]]
    aligns = ["left", "right", "right", "right", "right", "right", "right"]
    for cx, val, ca in zip(col_x, vals, aligns):
        indent = 0.25 if ca == "left" else 0
        ax.text(
            cx + indent, y, val, ha=ca, va="center", fontsize=9.5, fontfamily="serif"
        )
    y -= 0.48

# Bottom lines
ax.axhline(y=y + 0.25, xmin=0.02, xmax=0.98, color="black", lw=1.2)

# Note
note = (
    f"Note: LBW sample N={len(lbw_s):,} (births with T3 climate data, 2009–2021). "
    f"Mortality sample N={len(full_s):,}.\n"
    f"LBW = birth weight < 2,500g. Hot days = days exceeding threshold in T3 trimester. "
    f"Tmax anomaly = deviation from 1950–2000 baseline."
)
ax.text(
    0.15,
    y - 0.1,
    note,
    ha="left",
    va="top",
    fontsize=7.5,
    fontfamily="serif",
    color="#333333",
    wrap=True,
)

plt.tight_layout(pad=0.5)
out_png = f"{OUTPUT_DIR}/table1_publication.png"
plt.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Table image saved: {out_png}")

# ── LaTeX output ──────────────────────────────────────────────────────────────
latex = r"""\begin{table}[htbp]
\centering
\caption{Descriptive Statistics}
\label{tab:descriptive}
\begin{tabular}{lrrrrrr}
\hline
Variable & N & Mean & SD & p25 & Median & p75 \\
\hline
"""
current_group = None
for r in row_data:
    if r["Group"] != current_group:
        current_group = r["Group"]
        latex += f"\\multicolumn{{7}}{{l}}{{\\textbf{{{current_group}}}}} \\\\\n"
    vn = r["Variable"].replace("°", "$^\\circ$").replace("–", "--").replace("²", "$^2$")
    latex += (
        f"\\quad {vn} & {r['N']} & {r['Mean']} & {r['SD']} & "
        f"{r['p25']} & {r['Median']} & {r['p75']} \\\\\n"
    )
latex += (
    r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note:} LBW sample N="""
    + fmt_n(len(lbw_s))
    + r""". Mortality sample N="""
    + fmt_n(len(full_s))
    + r""".
LBW = birth weight $<$ 2,500g. Hot days = days exceeding threshold in T3 trimester.
Tmax anomaly = deviation from 1950--2000 baseline. Source: NFHS-4 and NFHS-5, IMD gridded temperature.
\end{tablenotes}
\end{table}
"""
)
out_tex = f"{OUTPUT_DIR}/table1_latex.tex"
with open(out_tex, "w") as f:
    f.write(latex)
print(f"LaTeX saved: {out_tex}")

print(f"\n{'='*60}")
print("DONE — Table 1 complete")
print(f"  PNG:   {out_png}")
print(f"  CSV:   {OUTPUT_DIR}/table1_publication.csv")
print(f"  LaTeX: {out_tex}")
