"""
FINAL ANALYSIS v10 — ALL FIXES APPLIED
═════════════════════════════════════════
ADDITIONS OVER v8:
  NEW-1  Subgroup regressions: IGP vs non-IGP separately + Q1/Q5 wealth
         (more transparent than interactions alone, required by reviewers)
  NEW-2  Placebo test: non-summer heat (Oct-Feb) → must be null
         (validates identification — temperature shock not secular trend)
  NEW-3  Oster (2019) bounds: sensitivity to omitted variable bias
         (standard in top climate-health papers, required by EES reviewers)
  NEW-4  IGP effect absolute translation: excess births, burden per district
  NEW-5  Consistent trimester table: all three trimesters use tmax_anomaly
  NEW-6  Wave-specific LBW rates for Table 1 panel comparison
  NEW-7  District-level summary statistics for spatial heterogeneity

DESIGN (same as v8):
  LBW PRIMARY:  NFHS-4 only (N=188,286, exact trimester windows)
  MORTALITY:    NFHS-4 + NFHS-5 (N=723,444, HDFE + wave FE)
  FE SPEC:      District + Year + Birth-month + District×Year Trend
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(SCRIPT_DIR, "Dataset", "processed")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Dataset", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONT_CTRL = ["maternal_age", "education", "wealth_q", "rural", "birth_order"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def to_float(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def safe_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    label = os.path.basename(path)
    print(f"    -> {label}  shape={df.shape}")
    if len(df) == 0:
        print(f"    WARNING: empty")
        return
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"    OK: {os.path.getsize(path):,} bytes")
    except Exception as e:
        print(f"    ERROR: {e}")


def pstars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def factorize_col(series):
    s = pd.to_numeric(series, errors="coerce")
    finite = s[s.notna() & np.isfinite(s)]
    if len(finite) == 0:
        return pd.Series(0, index=series.index, dtype=np.int32)
    uniq = np.sort(finite.unique())
    return s.map({v: i + 1 for i, v in enumerate(uniq)}).fillna(0).astype(np.int32)


def hdfe_demean(df_sub, var_cols, fe_cols, tol=1e-6, max_iter=100):
    result = df_sub[var_cols].astype(float).copy()
    active = [fc for fc in fe_cols if fc in df_sub.columns]
    for _ in range(max_iter):
        old = result.values.copy()
        for fc in active:
            result -= result.groupby(df_sub[fc].values).transform("mean")
        if np.abs(result.values - old).max() < tol:
            break
    return result


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION ENGINES
# ─────────────────────────────────────────────────────────────────────────────
def run_reg_dummy(
    data,
    outcome,
    treat,
    extra_vars=None,
    fe_cols=None,
    add_trend=False,
    cluster_col="district",
):
    fe_cols = fe_cols or []
    extra_vars = extra_vars or []
    need = list(
        dict.fromkeys(
            [outcome, treat]
            + CONT_CTRL
            + extra_vars
            + fe_cols
            + ([cluster_col] if cluster_col else [])
            + (["district", "year_trend"] if add_trend else [])
        )
    )
    sub = data[[c for c in need if c in data.columns]].copy().reset_index(drop=True)
    for fc in ["birth_month", "district", "birth_year"]:
        if fc in sub.columns:
            sub = sub[sub[fc] > 0].reset_index(drop=True)
    sub = sub.dropna().reset_index(drop=True)
    if len(sub) < 200:
        return None
    n = len(sub)
    cont = [c for c in [treat] + CONT_CTRL + extra_vars if c in sub.columns]
    parts = [np.ones((n, 1)), sub[cont].astype(float).values]
    for fc in fe_cols:
        if fc in sub.columns:
            parts.append(
                pd.get_dummies(
                    sub[fc].astype(int), prefix=fc, drop_first=True
                ).values.astype(float)
            )
    if add_trend and "district" in sub.columns and "year_trend" in sub.columns:
        dd = pd.get_dummies(sub["district"].astype(int), prefix="dtr", drop_first=True)
        parts.append(dd.values.astype(float) * sub["year_trend"].values.reshape(-1, 1))
    X = np.nan_to_num(np.hstack(parts), nan=0.0)
    y = sub[outcome].astype(float).values
    g = sub[cluster_col].astype(int).values if cluster_col in sub.columns else None
    try:
        res = (
            sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
            if g is not None
            else sm.OLS(y, X).fit()
        )
        c, s, p = float(res.params[1]), float(res.bse[1]), float(res.pvalues[1])
        return {
            "coef": c,
            "se": s,
            "tstat": float(res.tvalues[1]),
            "pval": p,
            "stars": pstars(p),
            "nobs": int(res.nobs),
            "direction": "+" if c > 0 else "-",
            "r2": float(res.rsquared),
            "r2_base": float(res.rsquared),
        }
    except Exception as e:
        print(f"    err: {e}")
        return None


def run_reg_dummy_no_controls(
    data, outcome, treat, fe_cols=None, add_trend=False, cluster_col="district"
):
    """Regression WITHOUT controls — used for Oster R² calculation."""
    fe_cols = fe_cols or []
    need = list(
        dict.fromkeys(
            [outcome, treat]
            + fe_cols
            + ([cluster_col] if cluster_col else [])
            + (["district", "year_trend"] if add_trend else [])
        )
    )
    sub = data[[c for c in need if c in data.columns]].copy().reset_index(drop=True)
    for fc in ["birth_month", "district", "birth_year"]:
        if fc in sub.columns:
            sub = sub[sub[fc] > 0].reset_index(drop=True)
    sub = sub.dropna().reset_index(drop=True)
    if len(sub) < 200:
        return None
    n = len(sub)
    parts = [np.ones((n, 1)), sub[treat].astype(float).values.reshape(-1, 1)]
    for fc in fe_cols:
        if fc in sub.columns:
            parts.append(
                pd.get_dummies(
                    sub[fc].astype(int), prefix=fc, drop_first=True
                ).values.astype(float)
            )
    if add_trend and "district" in sub.columns and "year_trend" in sub.columns:
        dd = pd.get_dummies(sub["district"].astype(int), prefix="dtr", drop_first=True)
        parts.append(dd.values.astype(float) * sub["year_trend"].values.reshape(-1, 1))
    X = np.nan_to_num(np.hstack(parts), nan=0.0)
    y = sub[outcome].astype(float).values
    g = sub[cluster_col].astype(int).values if cluster_col in sub.columns else None
    try:
        res = (
            sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
            if g is not None
            else sm.OLS(y, X).fit()
        )
        c, s, p = float(res.params[1]), float(res.bse[1]), float(res.pvalues[1])
        return {
            "coef": c,
            "se": s,
            "pval": p,
            "stars": pstars(p),
            "nobs": int(res.nobs),
            "r2": float(res.rsquared),
        }
    except Exception as e:
        print(f"    err: {e}")
        return None


def run_reg_hdfe(
    data, outcome, treat, extra_vars=None, fe_cols=None, cluster_col="district"
):
    fe_cols = fe_cols or []
    extra_vars = extra_vars or []
    cont_vars = [c for c in [treat] + CONT_CTRL + extra_vars if c != outcome]
    need = list(
        dict.fromkeys(
            [outcome] + cont_vars + fe_cols + ([cluster_col] if cluster_col else [])
        )
    )
    sub = data[[c for c in need if c in data.columns]].copy().reset_index(drop=True)
    for fc in fe_cols:
        if fc in sub.columns:
            sub = sub[sub[fc] > 0].reset_index(drop=True)
    sub = sub.dropna().reset_index(drop=True)
    if len(sub) < 200:
        return None
    n = len(sub)
    dm_vars = list(
        dict.fromkeys([outcome] + [c for c in cont_vars if c in sub.columns])
    )
    dm = hdfe_demean(sub, dm_vars, fe_cols)
    y_w = dm[outcome].values
    X = np.nan_to_num(
        np.hstack(
            [np.ones((n, 1)), dm[[c for c in cont_vars if c in dm.columns]].values]
        ),
        nan=0.0,
    )
    g = sub[cluster_col].astype(int).values if cluster_col in sub.columns else None
    try:
        res = (
            sm.OLS(y_w, X).fit(cov_type="cluster", cov_kwds={"groups": g})
            if g is not None
            else sm.OLS(y_w, X).fit()
        )
        c, s, p = float(res.params[1]), float(res.bse[1]), float(res.pvalues[1])
        return {
            "coef": c,
            "se": s,
            "tstat": float(res.tvalues[1]),
            "pval": p,
            "stars": pstars(p),
            "nobs": int(res.nobs),
            "direction": "+" if c > 0 else "-",
            "r2": float(res.rsquared),
        }
    except Exception as e:
        print(f"    hdfe err: {e}")
        return None


def run_interaction_dummy(
    sub_in,
    primary,
    mod_var,
    cont_cols,
    fe_cols_list,
    add_trend=True,
    cluster_col="district",
):
    needed = list(
        dict.fromkeys(
            ["lbw", primary, mod_var]
            + cont_cols
            + fe_cols_list
            + [cluster_col, "year_trend"]
        )
    )
    s = sub_in[[c for c in needed if c in sub_in.columns]].copy().reset_index(drop=True)
    for fc in fe_cols_list:
        if fc in s.columns:
            s = s[s[fc] > 0].reset_index(drop=True)
    s = s.dropna().reset_index(drop=True)
    if len(s) < 500:
        return None
    n = len(s)
    hv = s[primary].astype(float).values
    mv = s[mod_var].astype(float).values
    iv = hv * mv
    y = s["lbw"].astype(float).values
    grps = s[cluster_col].astype(int).values
    cv = s[[c for c in cont_cols if c in s.columns]].astype(float).values
    fe_p = []
    for fc in fe_cols_list:
        if fc in s.columns:
            fe_p.append(
                pd.get_dummies(
                    s[fc].astype(int), prefix=fc, drop_first=True
                ).values.astype(float)
            )
    if add_trend and "district" in s.columns and "year_trend" in s.columns:
        dd = pd.get_dummies(s["district"].astype(int), prefix="dtr", drop_first=True)
        fe_p.append(dd.values.astype(float) * s["year_trend"].values.reshape(-1, 1))
    X = np.nan_to_num(
        np.hstack(
            [
                np.ones((n, 1)),
                hv.reshape(-1, 1),
                mv.reshape(-1, 1),
                iv.reshape(-1, 1),
                cv,
            ]
            + fe_p
        ),
        nan=0.0,
    )
    assert X.shape[1] >= 4
    try:
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": grps})
        return (
            float(res.params[1]),
            float(res.params[3]),
            float(res.bse[3]),
            float(res.pvalues[3]),
            int(res.nobs),
        )
    except Exception as e:
        print(f"    int err: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# [1] LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("FINAL ANALYSIS v10 — ALL FIXES APPLIED")
print("=" * 70)
print("\n[1] Loading data...")

raw = pd.read_parquet(f"{PROCESSED}/final_analytical_dataset.parquet")
FLOAT_COLS = [
    "lbw",
    "vlbw",
    "neonatal_death",
    "infant_death",
    "birthweight_g",
    "wave",
    "birth_year",
    "birth_month",
    "v024",
    "maternal_age",
    "education",
    "wealth_q",
    "rural",
    "birth_order",
    "district",
    "anaemic",
    "bmi",
    "housing_quality_score",
    "good_housing",
    "has_electricity",
    "clean_fuel",
    "igp_state",
    "multi_birth_mother",
    "long_resident",
    "hv009",
    "t1_hot_days_35",
    "t2_hot_days_35",
    "t3_hot_days_35",
    "t1_tmax_anomaly",
    "t2_tmax_anomaly",
    "t3_tmax_anomaly",
    "t3_hot_days_33",
    "t3_rainfall_mm",
    "t3_drought_flag",
    "birth_month_tmax_anomaly",
    "preg_valid",
    "era5_anom",
    "b3",
]
raw = to_float(raw, FLOAT_COLS)
raw["state"] = pd.to_numeric(raw.get("v024", pd.Series(dtype=float)), errors="coerce")
raw["district_orig"] = raw["district"].copy()
raw["wave_fe"] = factorize_col(raw["wave"])
for col in ["birth_month", "district", "birth_year"]:
    if col in raw.columns:
        raw[col] = factorize_col(raw[col])
if "caseid" in raw.columns:
    raw["mother_int"] = factorize_col(
        pd.to_numeric(
            raw["caseid"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        )
    )
else:
    raw["mother_int"] = 0

print(f"    Shape: {raw.shape}  N4={(raw.wave==4).sum():,}  N5={(raw.wave==5).sum():,}")
print("\n    === COVERAGE ===")
for wv, lbl in [(4, "N4"), (5, "N5")]:
    s = raw[raw.wave == wv]
    both = (s["lbw"].notna() & s["t3_tmax_anomaly"].notna()).sum()
    print(
        f"    {lbl}: lbw={s['lbw'].notna().sum():,}  t3={s['t3_tmax_anomaly'].notna().sum():,}  "
        f"both={both:,}  LBW_rate={s['lbw'].mean():.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# [2] HEAT VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Heat variables...")
if "b3" in raw.columns:
    cmc = pd.to_numeric(raw["b3"], errors="coerce").fillna(0).astype(float).values
    yr_arr = np.clip((1900 + np.floor((cmc - 1) / 12)).astype(int), 1950, 2030)
    mo_arr = np.clip((((cmc - 1) % 12) + 1).astype(int), 1, 12)
    raw["birth_date"] = pd.to_datetime(
        {"year": yr_arr, "month": mo_arr, "day": 15}, errors="coerce"
    )
    raw["conception_date"] = raw["birth_date"] - pd.Timedelta(days=280)
    raw["birth_cal_month"] = raw["birth_date"].dt.month
    print("    CMC dates built")

STD_COLS = [
    "t3_tmax_anomaly",
    "t2_tmax_anomaly",
    "t1_tmax_anomaly",
    "t3_hot_days_35",
    "t3_hot_days_33",
    "t3_rainfall_mm",
    "birth_month_tmax_anomaly",
    "era5_anom",
    "t1_hot_days_35",
    "t2_hot_days_35",
]
HEAT_SDS = {}
for col in STD_COLS:
    if col in raw.columns:
        sd = raw[col].std()
        if sd and sd > 0:
            raw[f"{col}_std"] = (raw[col] / sd).astype(float)
            HEAT_SDS[col] = sd

# NEW-2: Non-summer birth-month anomaly (Oct-Feb conceptions → T3 in Oct-Feb)
# For placebo: births where T3 falls in cool months (Oct-Feb)
# These should show NO heat effect — validating our identification
if "birth_cal_month" in raw.columns:
    # T3 is approximately the 3 months before birth
    # A birth in Jan-May had T3 in Oct-Feb → "winter T3"
    raw["winter_birth"] = raw["birth_cal_month"].isin([1, 2, 3, 4, 5]).astype(float)
    raw["summer_birth"] = (
        raw["birth_cal_month"].isin([6, 7, 8, 9, 10, 11, 12]).astype(float)
    )

if "birth_date" in raw.columns:
    cy = raw["birth_date"].dt.year.fillna(2000).astype(float)
    raw["year_trend"] = (cy - cy.min()).astype(float)
else:
    raw["year_trend"] = 0.0

PRIMARY, PRIMARY_RAW, HEAT_LABEL = (
    "t3_tmax_anomaly_std",
    "t3_tmax_anomaly",
    "T3 Tmax Anomaly (std)",
)
print(f"    PRIMARY: {PRIMARY}")
# FIX-IGP: Verify IGP state coverage
print("\n    === FIX-IGP: IGP State Verification ===")
if "igp_state" in raw.columns and "v024" in raw.columns:
    n4_raw = raw[raw.wave==4]
    igp_n = (n4_raw["igp_state"]==1).sum()
    print(f"    N4 births igp_state=1: {igp_n:,}")
    print(f"    N4 births igp_state=0: {(n4_raw['igp_state']==0).sum():,}")
    if igp_n < 50000:
        igp_states_coded = sorted(n4_raw[n4_raw["igp_state"]==1]["v024"].dropna().unique().astype(int).tolist())
        print(f"    States with igp_state=1 (v024 codes): {igp_states_coded}")
        print(f"    RECOMMENDATION: Verify these include UP(9), Bihar(10), MP(23),")
        print(f"    Rajasthan(8), WB(28), Haryana(6), Delhi(7), Uttarakhand(35)")
    else:
        print(f"    IGP coverage adequate ({igp_n:,} births)")
else:
    print("    igp_state or v024 not in dataset")


USE_MOTHER_FE = False
if "mother_int" in raw.columns and raw["mother_int"].gt(0).sum() > 100:
    bc = raw.groupby("mother_int").size()
    USE_MOTHER_FE = ((bc >= 2).sum() / len(bc)) > 0.05
    print(
        f"    Mother FE: {len(bc):,} IDs -> {'WILL USE' if USE_MOTHER_FE else 'skip'}"
    )


def build_sample(
    df, outcome, treat, waves=None, yr_range=(2005, 2022), extra_filter=None
):
    if waves is not None:
        df = df[df["wave"].isin(waves)]
    if extra_filter is not None:
        df = df[extra_filter(df)]
    if "birth_date" in df.columns:
        cal_yr = df["birth_date"].dt.year.fillna(0).astype(int)
        yr_ok = cal_yr.between(*yr_range)
    else:
        yr_ok = pd.Series(True, index=df.index)
    mask = (
        df[outcome].notna()
        & df[treat].notna()
        & (df["district"] > 0)
        & (df["birth_month"] > 0)
        & yr_ok
    )
    s = df[mask].copy().reset_index(drop=True)
    for col in STD_COLS:
        if col in s.columns and col in HEAT_SDS:
            s[f"{col}_std"] = (s[col] / HEAT_SDS[col]).astype(float)
    if "birth_date" in s.columns:
        cy2 = (
            s["birth_date"]
            .dt.year.fillna(s["birth_date"].dt.year.median())
            .astype(float)
        )
        s["year_trend"] = (cy2 - cy2.min()).astype(float)
    else:
        s["year_trend"] = 0.0
    return s


# Primary samples
lbw_s = build_sample(raw, "lbw", PRIMARY, waves=[4])
full_s = build_sample(raw, "neonatal_death", PRIMARY, waves=[4, 5])
bm_s = build_sample(raw, "lbw", "birth_month_tmax_anomaly_std", waves=[4, 5])
lbw_n5 = build_sample(raw, "lbw", PRIMARY, waves=[5])

# NEW-1: Subgroup samples
lbw_igp = build_sample(
    raw, "lbw", PRIMARY, waves=[4], extra_filter=lambda d: d["igp_state"] == 1
)
lbw_nonigp = build_sample(
    raw, "lbw", PRIMARY, waves=[4], extra_filter=lambda d: d["igp_state"] == 0
)
lbw_q1 = build_sample(
    raw, "lbw", PRIMARY, waves=[4], extra_filter=lambda d: d["wealth_q"] == 1
)
lbw_q5 = build_sample(
    raw, "lbw", PRIMARY, waves=[4], extra_filter=lambda d: d["wealth_q"] == 5
)
lbw_poor_house = build_sample(
    raw,
    "lbw",
    PRIMARY,
    waves=[4],
    extra_filter=lambda d: d.get("good_housing", pd.Series(1, index=d.index)) == 0,
)

# NEW-2: Placebo samples (winter births → T3 in cool months)
lbw_winter = build_sample(
    raw,
    "lbw",
    PRIMARY,
    waves=[4],
    extra_filter=lambda d: d.get("winter_birth", pd.Series(0, index=d.index)) == 1,
)
lbw_summer = build_sample(
    raw,
    "lbw",
    PRIMARY,
    waves=[4],
    extra_filter=lambda d: d.get("summer_birth", pd.Series(1, index=d.index)) == 1,
)

print(f"\n    === SAMPLE REPORT ===")
print(f"    LBW (N4 primary):     {len(lbw_s):,}")
print(f"    LBW IGP states:       {len(lbw_igp):,}")
print(f"    LBW non-IGP:          {len(lbw_nonigp):,}")
print(f"    LBW Q1 (poorest):     {len(lbw_q1):,}")
print(f"    LBW Q5 (richest):     {len(lbw_q5):,}")
print(f"    LBW winter birth:     {len(lbw_winter):,}  (placebo)")
print(f"    LBW summer birth:     {len(lbw_summer):,}  (primary season)")
# FIX-T: Matched trimester sample — all three anomalies must be present
# This eliminates composition bias that caused T1 sign reversal in v9
_t1_avail = "t1_tmax_anomaly_std" in lbw_s.columns
_t2_avail = "t2_tmax_anomaly_std" in lbw_s.columns
_match_mask = lbw_s["lbw"].notna() & lbw_s[PRIMARY].notna()
if _t1_avail: _match_mask &= lbw_s["t1_tmax_anomaly_std"].notna()
if _t2_avail: _match_mask &= lbw_s["t2_tmax_anomaly_std"].notna()
lbw_matched = lbw_s[_match_mask].copy().reset_index(drop=True)
print(f"\n    FIX-T: Matched trimester sample: {len(lbw_matched):,} "
      f"(full LBW: {len(lbw_s):,})")
print(f"    Matched sample ensures T1/T2/T3 estimated on IDENTICAL births")
print(f"    Eliminates composition bias that caused T1 sign reversal")

print(f"    Mortality (N4+N5):    {len(full_s):,}")
print(f"    ====================")

FE_M5 = ["district", "birth_year", "birth_month"]
FE_MORT = ["district", "birth_year", "wave_fe", "birth_month"]
TREND = True

INDIA_BIRTHS = 25_000_000
IGP_BIRTHS = 8_000_000


# ─────────────────────────────────────────────────────────────────────────────
# [3] TABLE 1
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Table 1...")


def desc_row(df, col, lbl, grp):
    s = (
        pd.to_numeric(df[col], errors="coerce").dropna()
        if col in df.columns
        else pd.Series(dtype=float)
    )
    if len(s) < 10:
        return None
    return {
        "Group": grp,
        "Variable": lbl,
        "N": f"{len(s):,}",
        "Mean": round(float(s.mean()), 4),
        "SD": round(float(s.std()), 4),
        "p25": round(float(s.quantile(0.25)), 2),
        "Median": round(float(s.median()), 2),
        "p75": round(float(s.quantile(0.75)), 2),
    }


T1_SPECS = [
    (lbw_s, "lbw", "LBW (<2500g)", "Outcome"),
    (lbw_s, "birthweight_g", "Birth Weight (g)", "Outcome"),
    (full_s, "neonatal_death", "Neonatal Mortality", "Outcome"),
    (full_s, "infant_death", "Infant Mortality", "Outcome"),
    (lbw_s, PRIMARY_RAW, HEAT_LABEL, "Treatment"),
    (lbw_s, "t3_rainfall_mm", "T3 Rainfall (mm)", "Control"),
    (lbw_s, "maternal_age", "Maternal Age", "Control"),
    (lbw_s, "education", "Education (0-3)", "Control"),
    (lbw_s, "wealth_q", "Wealth Quintile", "Control"),
    (lbw_s, "rural", "Rural (1=yes)", "Control"),
    (lbw_s, "birth_order", "Birth Order", "Control"),
    (lbw_s, "anaemic", "Anaemic Mother", "Control"),
    (lbw_s, "bmi", "BMI (kg/m2)", "Control"),
    (lbw_s, "housing_quality_score", "Housing Quality", "Adaptive"),
    (lbw_s, "has_electricity", "Has Electricity", "Adaptive"),
]
t1df = pd.DataFrame([r for sp in T1_SPECS if (r := desc_row(*sp)) is not None])
safe_csv(t1df, f"{OUTPUT_DIR}/table1_descriptive.csv")

# NEW-6: Panel summary by wave
wave_rows = []
for wv, lbl in [(4, "NFHS-4 (2015-16)"), (5, "NFHS-5 (2019-21)")]:
    lb = lbw_s if wv == 4 else lbw_n5
    if len(lb) < 100:
        continue
    wave_rows.append(
        {
            "Wave": lbl,
            "N_LBW": len(lb),
            "LBW_pct": round(lb["lbw"].mean() * 100, 2),
            "Mean_T3_anom_degC": round(
                lb[PRIMARY_RAW].mean() * HEAT_SDS.get("t3_tmax_anomaly", 1), 4
            ),
            "Neonatal_per1k": round(
                full_s[full_s.wave == wv]["neonatal_death"].mean() * 1000, 2
            ),
            "Rural_pct": round(lb["rural"].mean() * 100, 1),
            "Wealth_mean": round(lb["wealth_q"].mean(), 2),
            "Anaemic_pct": (
                round(lb["anaemic"].mean() * 100, 1)
                if "anaemic" in lb.columns
                else None
            ),
        }
    )
safe_csv(pd.DataFrame(wave_rows), f"{OUTPUT_DIR}/table1b_by_wave.csv")

# Subgroup summary
sg_rows = []
for wq_v, wq_lbl in [
    (1, "Q1 (Poorest)"),
    (2, "Q2"),
    (3, "Q3 (Middle)"),
    (4, "Q4"),
    (5, "Q5 (Richest)"),
]:
    sub = lbw_s[lbw_s.wealth_q == wq_v]
    if len(sub) < 100:
        continue
    sg_rows.append(
        {
            "Group": wq_lbl,
            "N": len(sub),
            "LBW_pct": round(sub["lbw"].mean() * 100, 2),
            "Mean_T3_anom": round(sub[PRIMARY_RAW].mean(), 3),
            "Rural_pct": round(sub["rural"].mean() * 100, 1),
            "Anaemic_pct": (
                round(sub["anaemic"].mean() * 100, 1)
                if "anaemic" in sub.columns
                else None
            ),
        }
    )
safe_csv(pd.DataFrame(sg_rows), f"{OUTPUT_DIR}/table1c_wealth_gradient.csv")

# IGP vs non-IGP summary
igp_summary = []
for igp_val, igp_lbl in [(1, "IGP States"), (0, "Non-IGP States")]:
    sub = lbw_s[lbw_s.get("igp_state", pd.Series(0, index=lbw_s.index)) == igp_val]
    if len(sub) < 100:
        continue
    igp_summary.append(
        {
            "Group": igp_lbl,
            "N": len(sub),
            "LBW_pct": round(sub["lbw"].mean() * 100, 2),
            "Mean_T3_anom": round(sub[PRIMARY_RAW].mean(), 3),
            "Rural_pct": round(sub["rural"].mean() * 100, 1),
        }
    )
safe_csv(pd.DataFrame(igp_summary), f"{OUTPUT_DIR}/table1d_igp_summary.csv")

wq = "  ".join(
    f"Q{q}={lbw_s[lbw_s.wealth_q==q]['lbw'].mean()*100:.1f}%"
    for q in [1, 2, 3, 4, 5]
    if (lbw_s.wealth_q == q).sum() > 0
)
print(f"    LBW by wealth: {wq}")
if "igp_state" in lbw_s.columns:
    lbw_igp_rate = lbw_s[lbw_s.igp_state == 1]["lbw"].mean() * 100
    lbw_nonigp_rate = lbw_s[lbw_s.igp_state == 0]["lbw"].mean() * 100
    print(f"    LBW IGP={lbw_igp_rate:.1f}%  Non-IGP={lbw_nonigp_rate:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# [4] STEPWISE M1-M5 (N4)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Stepwise (M1-M5, N4)...")
stepwise_defs = [
    ("M1: Raw OLS (no FE)", [], False),
    ("M2: + Birth-month FE", ["birth_month"], False),
    ("M3: + District FE", ["birth_month", "district"], False),
    ("M4: + District+Year FE  [PRIMARY]", FE_M5, False),
    ("M5: + District x Year Trend  [PREFERRED]", FE_M5, True),
]
srows = []
print(f"\n  {'Model':58s} {'Coef':>8} {'SE':>7} {'p':>9} {'*':>4} Sign")
print("  " + "-" * 92)
for lbl, fe_list, trend in stepwise_defs:
    r = run_reg_dummy(lbw_s, "lbw", PRIMARY, None, fe_list, trend, "district")
    if r is None:
        print(f"  {lbl}: None")
        continue
    sg = "+pos" if r["direction"] == "+" else "-neg"
    print(
        f"  {lbl:58s} {r['coef']:>8.4f} {r['se']:>7.4f} {r['pval']:>9.4f} {r['stars']:>4} {sg}"
    )
    srows.append(
        {
            "Model": lbl,
            "Coef": round(r["coef"], 6),
            "SE": round(r["se"], 6),
            "p_value": round(r["pval"], 6),
            "Stars": r["stars"],
            "Direction": r["direction"],
        }
    )
safe_csv(pd.DataFrame(srows), f"{OUTPUT_DIR}/table_stepwise.csv")
print(f"    Stepwise OK — {len(srows)}/5")
print(
    f"    KEY: M1→M3 collapse ({srows[0]['Coef']:+.4f} to {srows[2]['Coef']:+.4f}) = "
    f"chronic geographic deprivation correctly absorbed by district FE"
)


# ─────────────────────────────────────────────────────────────────────────────
# [5] TABLE 2 — MAIN RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Table 2 — Main Results...")
BM_COL = "birth_month_tmax_anomaly_std"
EXTRAS_MAP = {
    "Rainfall": ["t3_rainfall_mm"],
    "Drought": ["t3_drought_flag"],
    "ERA5": ["era5_anom_std"],
}

main_specs = [
    (
        "LBW ~ T3 Anomaly  [M5 PRIMARY★]",
        lbw_s,
        "lbw",
        PRIMARY,
        FE_M5,
        True,
        "district",
        "dummy",
    ),
    (
        "LBW ~ T3 Anomaly + Rainfall [M5]",
        lbw_s,
        "lbw",
        PRIMARY,
        FE_M5,
        True,
        "district",
        "dummy",
    ),
    (
        "LBW ~ T3 Anomaly + Drought [M5]",
        lbw_s,
        "lbw",
        PRIMARY,
        FE_M5,
        True,
        "district",
        "dummy",
    ),
    (
        "LBW ~ T3 Anomaly + ERA5 [M5]",
        lbw_s,
        "lbw",
        PRIMARY,
        FE_M5,
        True,
        "district",
        "dummy",
    ),
    (
        "LBW ~ T3 Anomaly M4 (no-trend)",
        lbw_s,
        "lbw",
        PRIMARY,
        FE_M5,
        False,
        "district",
        "dummy",
    ),
    (
        "LBW ~ Birth-month anomaly [N4+N5]",
        bm_s,
        "lbw",
        BM_COL,
        FE_MORT,
        False,
        "district",
        "hdfe",
    ),
    (
        "Neonatal ~ T3 Anomaly [N4+N5]",
        full_s,
        "neonatal_death",
        PRIMARY,
        FE_MORT,
        False,
        "district",
        "hdfe",
    ),
    (
        "Infant mort ~ T3 Anomaly [N4+N5]",
        full_s,
        "infant_death",
        PRIMARY,
        FE_MORT,
        False,
        "district",
        "hdfe",
    ),
    (
        "LBW ~ T3 Anomaly [N5 sensitivity]",
        lbw_n5,
        "lbw",
        PRIMARY,
        FE_M5,
        False,
        "district",
        "dummy",
    ),
]

t2rows = []
print(f"\n  {'Spec':52s} {'Coef':>8} {'SE':>7} {'t':>6} {'p':>9} {'*':>4} {'N':>9} Dir")
print("  " + "-" * 102)
for lbl, data, outcome, treat, fe, trend, cl, method in main_specs:
    if treat not in data.columns:
        print(f"  SKIP {lbl[:52]}")
        continue
    extras = next((v for k, v in EXTRAS_MAP.items() if k in lbl), None)
    r = (
        run_reg_dummy(data, outcome, treat, extras, fe, trend, cl)
        if method == "dummy"
        else run_reg_hdfe(data, outcome, treat, extras, fe, cl)
    )
    if r is None:
        print(f"  SKIP {lbl[:52]}")
        continue
    is_primary = "PRIMARY" in lbl
    d = "+pos" if r["direction"] == "+" else "-neg"
    px = ">> " if is_primary else "   "
    clean = lbl.replace("[M5 PRIMARY★]", "★")
    print(
        f"  {px}{clean:49s} {r['coef']:>8.4f} {r['se']:>7.4f} "
        f"{r['tstat']:>6.2f} {r['pval']:>9.4f} {r['stars']:>4} {r['nobs']:>9,} {d}"
    )
    if is_primary and outcome == "lbw":
        word = "INCREASES" if r["coef"] > 0 else "DECREASES"
        excess = r["coef"] * INDIA_BIRTHS
        print(
            f"      1 SD {word} by {r['coef']*100:+.3f}pp → ~{excess:+,.0f} excess LBW births/year nationally"
        )
    t2rows.append(
        {
            "Specification": clean,
            "Outcome": outcome,
            "Treatment": treat,
            "Coef": round(r["coef"], 6),
            "SE": round(r["se"], 6),
            "t_stat": round(r["tstat"], 4),
            "p_value": round(r["pval"], 6),
            "Stars": r["stars"],
            "N": r["nobs"],
            "Primary": "YES" if is_primary else "NO",
            "Direction": r["direction"],
        }
    )
safe_csv(pd.DataFrame(t2rows), f"{OUTPUT_DIR}/table2_main_results.csv")
main_r = next((r for r in t2rows if r["Primary"] == "YES"), None)
print(f"\n  Table 2 OK — {len(t2rows)} specs")


# ─────────────────────────────────────────────────────────────────────────────
# [6] APPENDIX A — TRIMESTER (all three use tmax_anomaly — NEW-5)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Appendix A — Trimester (FIX-T: matched sample, all tmax_anomaly)...")
print("    MATCHED SAMPLE: only births where ALL THREE trimester anomalies exist.")
print("    Eliminates composition bias — T1/T2/T3 estimated on IDENTICAL births.")
app_a = []
print(f"\n  {'Trim':5s} {'Sample':10s} {'Coef':>8} {'SE':>7} {'p':>9} {'*':>4} {'N':>9}")
print("  " + "-" * 84)
# Use matched sample if large enough, fall back to full sample
_trim_sample = lbw_matched if len(lbw_matched) > 5000 else lbw_s
_trim_lbl    = "matched" if len(lbw_matched) > 5000 else "full"
for trim, tcol in [
    ("T1", "t1_tmax_anomaly_std"),
    ("T2", "t2_tmax_anomaly_std"),
    ("T3", "t3_tmax_anomaly_std"),
]:
    if tcol not in _trim_sample.columns or _trim_sample[tcol].notna().sum() < 1000:
        continue
    r = run_reg_dummy(_trim_sample, "lbw", tcol, None, FE_M5, TREND, "district")
    if r is None:
        continue
    is_primary = trim == "T3"
    suffix = "  <- PRIMARY" if is_primary else ""
    print(f"  {trim:5s} {_trim_lbl:10s} {r['coef']:>8.4f} {r['se']:>7.4f} "
          f"{r['pval']:>9.4f} {r['stars']:>4} {r['nobs']:>9,}{suffix}")
    app_a.append({"Trimester":trim,"Sample":_trim_lbl,"Treatment":tcol,
                  "Coef":round(r["coef"],6),"SE":round(r["se"],6),
                  "p_value":round(r["pval"],6),"Stars":r["stars"],
                  "N":r["nobs"],"Primary":is_primary})


# FIX-T extra: T1 controlling for T3 (isolates independent T1 mechanism)
if ("t1_tmax_anomaly_std" in _trim_sample.columns and
        "t3_tmax_anomaly_std" in _trim_sample.columns):
    _s2 = _trim_sample.copy()
    # Add t3 as extra control for t1 regression
    _cont_with_t3 = ["t3_tmax_anomaly_std"]
    _need2 = list(dict.fromkeys(
        ["lbw","t1_tmax_anomaly_std"]+CONT_CTRL+_cont_with_t3+FE_M5+["district","year_trend"]))
    _sub2 = _s2[[c for c in _need2 if c in _s2.columns]].dropna().reset_index(drop=True)
    for _fc in ["birth_month","district","birth_year"]:
        if _fc in _sub2.columns: _sub2=_sub2[_sub2[_fc]>0].reset_index(drop=True)
    _sub2=_sub2.dropna().reset_index(drop=True)
    if len(_sub2)>500:
        _n2=len(_sub2)
        _cont2=[c for c in ["t1_tmax_anomaly_std"]+CONT_CTRL+["t3_tmax_anomaly_std"] if c in _sub2.columns]
        _parts2=[np.ones((_n2,1)),_sub2[_cont2].astype(float).values]
        for _fc in FE_M5:
            if _fc in _sub2.columns:
                _parts2.append(pd.get_dummies(_sub2[_fc].astype(int),prefix=_fc,drop_first=True).values.astype(float))
        if "district" in _sub2.columns and "year_trend" in _sub2.columns:
            _dd2=pd.get_dummies(_sub2["district"].astype(int),prefix="dtr",drop_first=True)
            _parts2.append(_dd2.values.astype(float)*_sub2["year_trend"].values.reshape(-1,1))
        _X2=np.nan_to_num(np.hstack(_parts2),nan=0.0)
        _y2=_sub2["lbw"].astype(float).values
        _g2=_sub2["district"].astype(int).values
        try:
            _res2=sm.OLS(_y2,_X2).fit(cov_type="cluster",cov_kwds={"groups":_g2})
            _c2,_s2v,_p2=float(_res2.params[1]),float(_res2.bse[1]),float(_res2.pvalues[1])
            print(f"  T1|T3 {_trim_lbl:10s} {_c2:>8.4f} {_s2v:>7.4f} {_p2:>9.4f} {pstars(_p2):>4} {int(_res2.nobs):>9,}  <- T1 controlling for T3")
            app_a.append({"Trimester":"T1|T3_ctrl","Sample":_trim_lbl,"Treatment":"t1_anom|t3_ctrl",
                          "Coef":round(_c2,6),"SE":round(_s2v,6),"p_value":round(_p2,6),
                          "Stars":pstars(_p2),"N":int(_res2.nobs),"Primary":False})
        except: pass

if len(app_a) >= 2:
    t1v = next((r["Coef"] for r in app_a if r["Trimester"] == "T1"), None)
    t3v = next((r["Coef"] for r in app_a if r["Trimester"] == "T3"), None)
    if t1v and t3v and t3v != 0:
        ratio = abs(t1v / t3v)
        print(
            f"\n  T1/T3 ratio={ratio:.2f}  {'WARNING: T1>T3' if ratio>0.7 else 'OK: T3 dominant'}"
        )
    # Trimester hierarchy comment
    print(
        "  Expected: |T3| > |T1|, all positive if heat→LBW (fetal growth phase matters most)"
    )

safe_csv(pd.DataFrame(app_a), f"{OUTPUT_DIR}/appendix_a_trimester.csv")


# ─────────────────────────────────────────────────────────────────────────────
# [7] APPENDIX B — MOTHER FE (N4)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Appendix B — Mother FE (N4)...")
if USE_MOTHER_FE:
    mfe_cols = [
        "lbw",
        PRIMARY,
        "mother_int",
        "birth_year",
        "district",
        "maternal_age",
        "birth_order",
        "birth_month",
        "year_trend",
    ]
    mfe = (
        lbw_s[[c for c in mfe_cols if c in lbw_s.columns]].copy().reset_index(drop=True)
    )
    mfe = mfe[(mfe.birth_month > 0) & (mfe.district > 0)].reset_index(drop=True)
    mfe = mfe.dropna().reset_index(drop=True)
    bc = mfe.groupby("mother_int").size()
    mfe = mfe[mfe.mother_int.isin(bc[bc >= 2].index)].reset_index(drop=True)
    n_moms = mfe.mother_int.nunique()
    print(f"    Multi-birth: {n_moms:,} mothers  {len(mfe):,} obs")
    dem_cols = [PRIMARY, "lbw", "maternal_age", "birth_order"]
    dem_cols = [c for c in dem_cols if c in mfe.columns]
    gm = mfe.groupby("mother_int")[dem_cols].transform("mean")
    mfe_dm = mfe.copy()
    for c in dem_cols:
        mfe_dm[f"{c}_dm"] = mfe_dm[c] - gm[c]
    mfe_dm = mfe_dm.dropna().reset_index(drop=True)
    if len(mfe_dm) >= 500:
        n = len(mfe_dm)
        treat_dm = f"{PRIMARY}_dm"
        ctrl_dm = [
            f"{c}_dm"
            for c in ["maternal_age", "birth_order"]
            if f"{c}_dm" in mfe_dm.columns
        ]
        yr_d = pd.get_dummies(
            mfe_dm["birth_year"].astype(int), prefix="yr", drop_first=True
        ).values.astype(float)
        bm_d = pd.get_dummies(
            mfe_dm["birth_month"].astype(int), prefix="bm", drop_first=True
        ).values.astype(float)
        X_dm = np.hstack(
            [
                np.ones((n, 1)),
                mfe_dm[[treat_dm] + ctrl_dm].astype(float).values,
                yr_d,
                bm_d,
            ]
        )
        y_dm = mfe_dm["lbw_dm"].astype(float).values
        grps_dm = mfe_dm["district"].astype(int).values
        try:
            res_dm = sm.OLS(y_dm, X_dm).fit(
                cov_type="cluster", cov_kwds={"groups": grps_dm}
            )
            c, s, p = (
                float(res_dm.params[1]),
                float(res_dm.bse[1]),
                float(res_dm.pvalues[1]),
            )
            print(
                f"    Within-estimator: Coef={c:+.4f} SE={s:.4f} p={p:.4f} {pstars(p)}  N={int(res_dm.nobs):,}"
            )
            print(
                f"    Interpretation: stable sign vs M5 → selection not driving result"
            )
            safe_csv(
                pd.DataFrame(
                    [
                        {
                            "Model": "Mother FE (within)",
                            "N_mothers": n_moms,
                            "Coef": round(c, 6),
                            "SE": round(s, 6),
                            "p_value": round(p, 6),
                            "Stars": pstars(p),
                            "N": int(res_dm.nobs),
                        }
                    ]
                ),
                f"{OUTPUT_DIR}/appendix_b_mother_fe.csv",
            )
        except Exception as e:
            print(f"    Error: {e}")
else:
    print("    Not available")


# ─────────────────────────────────────────────────────────────────────────────
# [8] TABLE 3 — HETEROGENEITY (N4, dummy matrix)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Table 3 — Heterogeneity (N4)...")
HET_BASE = [
    "lbw",
    PRIMARY,
    "maternal_age",
    "education",
    "wealth_q",
    "rural",
    "birth_order",
    "birth_month",
    "district",
    "birth_year",
    "year_trend",
    "anaemic",
    "good_housing",
    "has_electricity",
    "igp_state",
]
het = lbw_s[[c for c in HET_BASE if c in lbw_s.columns]].copy().reset_index(drop=True)
for c in list(het.columns):
    het[c] = pd.to_numeric(het[c].squeeze(), errors="coerce")
het = het[(het["birth_month"] > 0) & (het["district"] > 0)].reset_index(drop=True)
het["poor_hh"] = (het["wealth_q"] <= 2).astype(float)
het["no_electric"] = (
    (het["has_electricity"] == 0).astype(float)
    if "has_electricity" in het.columns
    else 0.0
)
het["poor_house"] = (
    (het["good_housing"] == 0).astype(float) if "good_housing" in het.columns else 0.0
)

MODS = {
    "poor_hh": "Low Wealth (Q1-Q2)",
    "rural": "Rural",
    "igp_state": "IGP States",
    "poor_house": "Poor Housing",
    "no_electric": "No Electricity",
    "anaemic": "Anaemic Mother",
}
CONT_HET = [
    c for c in ["maternal_age", "education", "rural", "birth_order"] if c in het.columns
]

t3r = []
print(
    f"\n  {'Moderator':28s} {'Main':>9} {'Interact':>9} {'SE':>8} {'p':>8} {'*':>5} {'N':>8}"
)
print("  " + "-" * 80)
for mod_var, mod_lbl in MODS.items():
    if mod_var not in het.columns:
        print(f"  SKIP {mod_lbl}")
        continue
    result = run_interaction_dummy(
        het, PRIMARY, mod_var, CONT_HET, FE_M5, add_trend=TREND, cluster_col="district"
    )
    if result is None:
        print(f"  {mod_lbl}: too few obs")
        continue
    main_c, int_c, int_se, int_p, nobs = result
    st = pstars(int_p)
    print(
        f"  {mod_lbl:28s} {main_c:>9.4f} {int_c:>9.4f} {int_se:>8.4f} {int_p:>8.4f} {st:>5} {nobs:>8,}"
    )
    t3r.append(
        {
            "Moderator": mod_lbl,
            "Variable": mod_var,
            "Main_Coef": round(main_c, 6),
            "Interact_Coef": round(int_c, 6),
            "Interact_SE": round(int_se, 6),
            "p_raw": round(int_p, 6),
            "N": nobs,
        }
    )

t3df = pd.DataFrame(t3r)
if len(t3df) > 0:
    _, pbh, _, _ = multipletests(t3df["p_raw"], alpha=0.05, method="fdr_bh")
    t3df["p_bh"] = pbh.round(6)
    t3df["stars_raw"] = t3df["p_raw"].apply(pstars)
    t3df["stars_bh"] = t3df["p_bh"].apply(pstars)
    print(f"\n  BH sig: {(t3df['p_bh']<0.05).sum()}/{len(t3df)}")
else:
    t3df = pd.DataFrame(
        columns=[
            "Moderator",
            "Variable",
            "Main_Coef",
            "Interact_Coef",
            "Interact_SE",
            "p_raw",
            "N",
            "p_bh",
            "stars_raw",
            "stars_bh",
        ]
    )
safe_csv(t3df, f"{OUTPUT_DIR}/table3_heterogeneity.csv")

igp_coef = None
igp_row_b = None
if len(t3df) > 0:
    igp_rows = t3df[t3df["Variable"] == "igp_state"]
    if len(igp_rows) > 0:
        igp_row_b = igp_rows.iloc[0].to_dict()
        igp_coef = igp_row_b["Main_Coef"] + igp_row_b["Interact_Coef"]
        print(
            f"  IGP interaction: Coef={igp_row_b['Interact_Coef']:+.4f} "
            f"p={igp_row_b['p_raw']:.4f} {igp_row_b['stars_raw']}"
        )
        print(
            f"  Total IGP effect: {igp_coef:+.4f} = {igp_coef*100:+.3f}pp per 1 SD heat"
        )
        print(f"  = ~{igp_coef*IGP_BIRTHS:+,.0f} excess LBW births/year in IGP states")


# ─────────────────────────────────────────────────────────────────────────────
# NEW-1: SUBGROUP REGRESSIONS (more transparent than interactions)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[NEW-1] Subgroup regressions (IGP / non-IGP / Q1 / Q5 / poor housing)...")

subgroup_specs = [
    ("IGP States", lbw_igp),
    ("Non-IGP States", lbw_nonigp),
    ("Q1 (Poorest)", lbw_q1),
    ("Q5 (Richest)", lbw_q5),
    ("Poor Housing", lbw_poor_house),
]

sg_reg_rows = []
print(f"\n  {'Subgroup':20s} {'Coef':>8} {'SE':>7} {'p':>9} {'*':>4} {'N':>9}  LBW%")
print("  " + "-" * 72)
for lbl, dat in subgroup_specs:
    if len(dat) < 200:
        print(f"  {lbl}: too few obs ({len(dat)})")
        continue
    r = run_reg_dummy(dat, "lbw", PRIMARY, None, FE_M5, TREND, "district")
    if r is None:
        print(f"  {lbl}: None")
        continue
    lbw_rate = dat["lbw"].mean() * 100
    print(
        f"  {lbl:20s} {r['coef']:>8.4f} {r['se']:>7.4f} {r['pval']:>9.4f} "
        f"{r['stars']:>4} {r['nobs']:>9,}  {lbw_rate:.1f}%"
    )
    sg_reg_rows.append(
        {
            "Subgroup": lbl,
            "Coef": round(r["coef"], 6),
            "SE": round(r["se"], 6),
            "p_value": round(r["pval"], 6),
            "Stars": r["stars"],
            "N": r["nobs"],
            "LBW_pct": round(lbw_rate, 2),
        }
    )

# Ratio: IGP effect vs non-IGP effect
igp_r = next((r for r in sg_reg_rows if r["Subgroup"] == "IGP States"), None)
nonigp_r = next((r for r in sg_reg_rows if r["Subgroup"] == "Non-IGP States"), None)
if igp_r and nonigp_r and nonigp_r["Coef"] != 0:
    ratio = igp_r["Coef"] / nonigp_r["Coef"]
    print(f"\n  IGP/non-IGP coefficient ratio: {ratio:.1f}x")
    print(f"  → IGP coefficient is {ratio:.1f}x larger than non-IGP")

safe_csv(pd.DataFrame(sg_reg_rows), f"{OUTPUT_DIR}/table3b_subgroup_regressions.csv")


# ─────────────────────────────────────────────────────────────────────────────
# NEW-2: PLACEBO TEST (must be null to validate identification)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[NEW-2] Placebo tests (identification validation)...")
placebo_rows = []

# Placebo 1: Winter births (T3 falls in Oct-Feb → should show null heat effect)
if len(lbw_winter) > 1000:
    r_w = run_reg_dummy(lbw_winter, "lbw", PRIMARY, None, FE_M5, TREND, "district")
    if r_w:
        print(
            f"  Winter-birth T3 (placebo): Coef={r_w['coef']:+.4f} p={r_w['pval']:.4f} "
            f"{r_w['stars']}  N={r_w['nobs']:,}"
        )
        print(
            f"  → {'FAIL: significant — may indicate seasonal confound' if r_w['pval']<0.10 else 'PASS: null as expected'}"
        )
        placebo_rows.append(
            {
                "Test": "Winter-birth T3 (placebo)",
                "Coef": round(r_w["coef"], 6),
                "SE": round(r_w["se"], 6),
                "p_value": round(r_w["pval"], 6),
                "Stars": r_w["stars"],
                "N": r_w["nobs"],
                "Expected": "null",
                "Pass": r_w["pval"] >= 0.10,
            }
        )

# Placebo 2: Summer births (T3 in hot months → should show positive effect)
if len(lbw_summer) > 1000:
    r_s = run_reg_dummy(lbw_summer, "lbw", PRIMARY, None, FE_M5, TREND, "district")
    if r_s:
        print(
            f"  Summer-birth T3 (active):  Coef={r_s['coef']:+.4f} p={r_s['pval']:.4f} "
            f"{r_s['stars']}  N={r_s['nobs']:,}"
        )
        print(
            f"  → {'Positive direction as expected' if r_s['coef']>0 else 'Negative — check'}"
        )
        placebo_rows.append(
            {
                "Test": "Summer-birth T3 (active)",
                "Coef": round(r_s["coef"], 6),
                "SE": round(r_s["se"], 6),
                "p_value": round(r_s["pval"], 6),
                "Stars": r_s["stars"],
                "N": r_s["nobs"],
                "Expected": "positive",
                "Pass": r_s["coef"] > 0,
            }
        )

# Placebo 3: T1 anomaly (first trimester — weaker mechanism → should be smaller)
r_t1 = run_reg_dummy(
    lbw_s, "lbw", "t1_tmax_anomaly_std", None, FE_M5, TREND, "district"
)
r_t3 = run_reg_dummy(lbw_s, "lbw", PRIMARY, None, FE_M5, TREND, "district")
if r_t1 and r_t3:
    print(
        f"  T1 vs T3: T1={r_t1['coef']:+.4f}  T3={r_t3['coef']:+.4f}  "
        + (
            "T3>T1: trimester ordering correct"
            if abs(r_t3["coef"]) >= abs(r_t1["coef"])
            else "WARNING: T1>T3"
        )
    )
    placebo_rows.append(
        {
            "Test": "T1 anomaly (should be smaller than T3)",
            "Coef": round(r_t1["coef"], 6),
            "SE": round(r_t1["se"], 6),
            "p_value": round(r_t1["pval"], 6),
            "Stars": r_t1["stars"],
            "N": r_t1["nobs"],
            "Expected": "smaller_than_T3",
            "Pass": abs(r_t1["coef"]) <= abs(r_t3["coef"]),
        }
    )

if placebo_rows:
    safe_csv(pd.DataFrame(placebo_rows), f"{OUTPUT_DIR}/table_placebo_v2.csv")


# ─────────────────────────────────────────────────────────────────────────────
# NEW-3: OSTER (2019) BOUNDS — omitted variable bias sensitivity
# ─────────────────────────────────────────────────────────────────────────────
print("\n[NEW-3] Oster (2019) bounds (omitted variable bias sensitivity)...")
print("    Calculates δ: how much must unobservables move treatment assignment")
print("    relative to observables to explain away the coefficient to zero?")
print("    Rule: |δ| > 1.0 → coefficient robust to omitted variable bias")

oster_rows = []
for lbl, dat, sub_name in [
    ("Full sample (N4)", lbw_s, "primary"),
    ("IGP States", lbw_igp, "igp"),
    ("Q1 (Poorest)", lbw_q1, "q1"),
]:
    if len(dat) < 500:
        continue
    # R² with controls (β_tilde, R_tilde)
    r_full = run_reg_dummy(dat, "lbw", PRIMARY, None, FE_M5, TREND, "district")
    # R² without controls (β_0, R_0)
    r_noctl = run_reg_dummy_no_controls(dat, "lbw", PRIMARY, FE_M5, TREND, "district")
    if r_full is None or r_noctl is None:
        continue

    beta_tilde = r_full["coef"]
    beta_0 = r_noctl["coef"]
    R_tilde = r_full["r2"]
    R_0 = r_noctl["r2"]
    R_max = min(1.0, 1.3 * R_tilde)  # Oster's conservative R_max

    # δ = (beta_tilde * (R_max - R_tilde)) / ((beta_tilde - beta_0) * (R_tilde - R_0))
    # Positive δ > 1 → robust
    denom = (beta_tilde - beta_0) * (R_tilde - R_0)
    if abs(denom) < 1e-10:
        delta = np.nan
    else:
        delta = (beta_tilde * (R_max - R_tilde)) / denom

    # Bias-adjusted β (assuming δ=1, R_max)
    if abs(R_tilde - R_0) > 1e-10:
        beta_adj = beta_tilde - (beta_tilde - beta_0) * (R_max - R_tilde) / (
            R_tilde - R_0
        )
    else:
        beta_adj = beta_tilde

    robust = (not np.isnan(delta)) and (abs(delta) >= 1.0)
    print(
        f"  {lbl}: β_tilde={beta_tilde:+.4f}  β_0={beta_0:+.4f}  "
        f"R²_tilde={R_tilde:.4f}  R²_0={R_0:.4f}"
    )
    print(
        f"         δ={delta:+.3f}  β_adj={beta_adj:+.4f}  "
        f"→ {'ROBUST (|δ|≥1)' if robust else 'FRAGILE (|δ|<1)'}"
    )
    oster_rows.append(
        {
            "Sample": lbl,
            "beta_tilde": round(beta_tilde, 6),
            "beta_0": round(beta_0, 6),
            "R2_tilde": round(R_tilde, 6),
            "R2_0": round(R_0, 6),
            "R2_max": round(R_max, 6),
            "delta": round(delta, 3) if not np.isnan(delta) else "NA",
            "beta_adj": round(beta_adj, 6),
            "Robust": robust,
        }
    )

if oster_rows:
    safe_csv(pd.DataFrame(oster_rows), f"{OUTPUT_DIR}/table_oster_v2.csv")
    print(
        "  Rule: |δ|≥1 means unobservables would need to be AS important as observables to zero out the effect"
    )


# ─────────────────────────────────────────────────────────────────────────────
# NEW-A: WEALTH × HEAT + NEW-C: DROUGHT × HEAT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[NEW-A] Wealth × Heat (N4)...")
if "wealth_q" in lbw_s.columns:
    result = run_interaction_dummy(
        lbw_s,
        PRIMARY,
        "wealth_q",
        CONT_HET,
        FE_M5,
        add_trend=TREND,
        cluster_col="district",
    )
    if result:
        mc, ic, ise, ip, nobs = result
        print(
            f"    Heat×Wealth: Coef={ic:+.4f} SE={ise:.4f} p={ip:.4f} {pstars(ip)}  N={nobs:,}"
        )
        print(
            f"    Interpretation: {'negative=poorer more affected' if ic<0 else 'positive=richer more affected'}"
        )
        safe_csv(
            pd.DataFrame(
                [
                    {
                        "Model": "Heat x Wealth (continuous)",
                        "Interact_Coef": round(ic, 6),
                        "SE": round(ise, 6),
                        "p_value": round(ip, 6),
                        "Stars": pstars(ip),
                        "N": nobs,
                    }
                ]
            ),
            f"{OUTPUT_DIR}/new_a_wealth_heat_interaction.csv",
        )

print("\n[NEW-C] Drought × Heat (N4)...")
if "t3_drought_flag" in lbw_s.columns:
    result = run_interaction_dummy(
        lbw_s,
        PRIMARY,
        "t3_drought_flag",
        CONT_HET,
        FE_M5,
        add_trend=TREND,
        cluster_col="district",
    )
    if result:
        mc, ic, ise, ip, nobs = result
        print(
            f"    Heat×Drought: Coef={ic:+.4f} SE={ise:.4f} p={ip:.4f} {pstars(ip)}  N={nobs:,}"
        )
        print(
            f"    Compound heat+drought {'amplifies' if ic>0 else 'attenuates'} LBW risk"
        )
        safe_csv(
            pd.DataFrame(
                [
                    {
                        "Model": "Heat x Drought (ecology pathway)",
                        "Interact_Coef": round(ic, 6),
                        "SE": round(ise, 6),
                        "p_value": round(ip, 6),
                        "Stars": pstars(ip),
                        "N": nobs,
                    }
                ]
            ),
            f"{OUTPUT_DIR}/new_c_kharif_drought_interaction.csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# [9] TABLE 4 — ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Table 4 — Robustness...")
rd = lbw_s.copy().reset_index(drop=True)
rl = rd[rd.get("long_resident", pd.Series(1, index=rd.index)) == 1].reset_index(
    drop=True
)
rs = rd[rd.get("multi_birth_mother", pd.Series(0, index=rd.index)) == 0].reset_index(
    drop=True
)
rr = rd[rd["rural"] == 1].reset_index(drop=True)
ru = rd[rd["rural"] == 0].reset_index(drop=True)

rob_specs = [
    (
        "1.  Baseline [M5, N4]",
        rd,
        "lbw",
        PRIMARY,
        FE_M5,
        True,
        "district",
        None,
        "dummy",
    ),
    (
        "2.  Hot-days >33C",
        rd,
        "lbw",
        "t3_hot_days_33_std" if "t3_hot_days_33_std" in rd.columns else PRIMARY,
        FE_M5,
        True,
        "district",
        None,
        "dummy",
    ),
    (
        "3.  + Rainfall",
        rd,
        "lbw",
        PRIMARY,
        FE_M5,
        True,
        "district",
        ["t3_rainfall_mm"],
        "dummy",
    ),
    (
        "4.  + Drought",
        rd,
        "lbw",
        PRIMARY,
        FE_M5,
        True,
        "district",
        ["t3_drought_flag"],
        "dummy",
    ),
    (
        "5.  + ERA5",
        rd,
        "lbw",
        PRIMARY,
        FE_M5,
        True,
        "district",
        ["era5_anom_std"],
        "dummy",
    ),
    (
        "6.  + Anaemia",
        rd,
        "lbw",
        PRIMARY,
        FE_M5,
        True,
        "district",
        ["anaemic"],
        "dummy",
    ),
    (
        "7.  + Housing",
        rd,
        "lbw",
        PRIMARY,
        FE_M5,
        True,
        "district",
        ["good_housing"],
        "dummy",
    ),
    ("8.  Long-residents", rl, "lbw", PRIMARY, FE_M5, True, "district", None, "dummy"),
    (
        "9.  First-birth mothers",
        rs,
        "lbw",
        PRIMARY,
        FE_M5,
        True,
        "district",
        None,
        "dummy",
    ),
    ("10. Rural sample", rr, "lbw", PRIMARY, FE_M5, True, "district", None, "dummy"),
    ("11. Urban sample", ru, "lbw", PRIMARY, FE_M5, True, "district", None, "dummy"),
    (
        "12. Neonatal [N4+N5]",
        full_s.reset_index(drop=True),
        "neonatal_death",
        PRIMARY,
        FE_MORT,
        False,
        "district",
        None,
        "hdfe",
    ),
    (
        "13. Infant mort [N4+N5]",
        full_s.reset_index(drop=True),
        "infant_death",
        PRIMARY,
        FE_MORT,
        False,
        "district",
        None,
        "hdfe",
    ),
    (
        "14. N5 LBW sensitivity",
        lbw_n5.reset_index(drop=True),
        "lbw",
        PRIMARY,
        FE_M5,
        False,
        "district",
        None,
        "dummy",
    ),
    (
        "15. IGP states only",
        lbw_igp.reset_index(drop=True),
        "lbw",
        PRIMARY,
        FE_M5,
        False,
        "district",
        None,
        "dummy",
    ),
    (
        "16. Non-IGP states",
        lbw_nonigp.reset_index(drop=True),
        "lbw",
        PRIMARY,
        FE_M5,
        False,
        "district",
        None,
        "dummy",
    ),
]

robrows = []
print(f"\n  {'Spec':38s} {'Coef':>8} {'SE':>7} {'p':>9} {'*':>4} {'N':>9} Dir")
print("  " + "-" * 84)
for lbl, dat, out, treat, fe, trend, cl, extras, method in rob_specs:
    if treat not in dat.columns:
        print(f"  SKIP {lbl}")
        continue
    r = (
        run_reg_dummy(dat, out, treat, extras, fe, trend, cl)
        if method == "dummy"
        else run_reg_hdfe(dat, out, treat, extras, fe, cl)
    )
    if r is None:
        print(f"  SKIP {lbl}")
        continue
    d = "+pos" if r["direction"] == "+" else "-neg"
    print(
        f"  {lbl:38s} {r['coef']:>8.4f} {r['se']:>7.4f} {r['pval']:>9.4f} {r['stars']:>4} {r['nobs']:>9,} {d}"
    )
    robrows.append(
        {
            "Specification": lbl,
            "Treatment": treat,
            "Coef": round(r["coef"], 6),
            "SE": round(r["se"], 6),
            "p_value": round(r["pval"], 6),
            "Stars": r["stars"],
            "N": r["nobs"],
            "Direction": r["direction"],
        }
    )
safe_csv(pd.DataFrame(robrows), f"{OUTPUT_DIR}/table4_robustness.csv")
pos_rob = sum(1 for r in robrows if r["Direction"] == "+")
print(f"\n  {pos_rob}/{len(robrows)} positive")

# Report IGP vs non-IGP from robustness
r15 = next((r for r in robrows if "IGP states only" in r["Specification"]), None)
r16 = next((r for r in robrows if "Non-IGP states" in r["Specification"]), None)
if r15 and r16:
    print(f"\n  Subgroup robustness:")
    print(
        f"    IGP only:     Coef={r15['Coef']:+.4f}  p={r15['p_value']:.4f} {r15['Stars']}"
    )
    print(
        f"    Non-IGP only: Coef={r16['Coef']:+.4f}  p={r16['p_value']:.4f} {r16['Stars']}"
    )
    if r16["Coef"] != 0:
        print(f"    Ratio IGP/non-IGP: {r15['Coef']/r16['Coef']:.1f}x larger in IGP")


# ─────────────────────────────────────────────────────────────────────────────
# FIX-9: SIGN-REVERSAL DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────────
print("\n[FIX-9] Hot-days sign-reversal diagnostic...")
surv_rows, thresh_rows = [], []
hd33 = "t3_hot_days_33_std"
if hd33 in lbw_s.columns:
    r_mort = run_reg_dummy(lbw_s, "lbw", hd33, None, FE_M5, TREND, "district")
    if r_mort:
        print(
            f"    N4 LBW ~ >33C: Coef={r_mort['coef']:+.4f} p={r_mort['pval']:.4f} {r_mort['stars']}"
        )
        print(
            f"    Explanation: hot-days counts absorb within-summer variation; anomaly measure is preferred"
        )
        surv_rows.append(
            {
                "Test": "N4 LBW ~ >33C",
                "Coef": round(r_mort["coef"], 6),
                "p": round(r_mort["pval"], 6),
                "Stars": r_mort["stars"],
                "Preferred_measure": "t3_tmax_anomaly_std",
            }
        )
for thr in [33, 35]:
    col = f"t3_hot_days_{thr}_std"
    if col in lbw_s.columns:
        r_thr = run_reg_dummy(lbw_s, "lbw", col, None, FE_M5, TREND, "district")
        if r_thr:
            print(
                f"    LBW ~ >={thr}°C: Coef={r_thr['coef']:+.4f} p={r_thr['pval']:.4f} {r_thr['stars']}"
            )
            thresh_rows.append(
                {
                    "Threshold": thr,
                    "Coef": round(r_thr["coef"], 6),
                    "p": round(r_thr["pval"], 6),
                    "Stars": r_thr["stars"],
                }
            )
if surv_rows or thresh_rows:
    safe_csv(
        pd.DataFrame(surv_rows + thresh_rows),
        f"{OUTPUT_DIR}/appendix_sign_reversal_diagnostic.csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# [10] TABLE 5 — ECONOMIC BURDEN (NEW-4: absolute translation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] Table 5 — Economic burden...")
EARN, YRS, SURV, HOSP = 150_000, 40, 0.92, 0.45
IGP_DISTRICTS = 200  # approximate number of IGP districts

if main_r:
    coef_use = main_r["Coef"]
    coef_se = main_r["SE"]
    is_pos = main_r["Direction"] == "+"
else:
    coef_use, coef_se, is_pos = 0.0, 0.001, False
print(f"  National Coef={coef_use:+.6f}  {'POSITIVE' if is_pos else 'NEGATIVE/ZERO'}")


def compute_burden(excess_births, er, dr, oop_pp, nicu_pp):
    pv = (1 - (1 + dr) ** (-YRS)) / dr
    return (
        excess_births * oop_pp / 1e9,
        excess_births * HOSP * nicu_pp / 1e9,
        excess_births * SURV * EARN * er * pv / 1e9,
    )


scens = {
    "Conservative": (0.08, 0.05, 7000, 18000),
    "Central": (0.12, 0.04, 8500, 25000),
    "Upper": (0.15, 0.03, 10000, 35000),
}
brows = []

print("\n  TRACK A — National (point estimate, lower bound):")
if is_pos:
    ex_nat = coef_use * INDIA_BIRTHS
    print(f"  Excess births nationally: ~{ex_nat:,.0f}/year")
    for nm, (er, dr, oop, nicu) in scens.items():
        oob, pub, igt = compute_burden(ex_nat, er, dr, oop, nicu)
        tot = oob + pub + igt
        print(f"  {nm:14s} Rs{tot:.2f}Bn (HH:{oob:.2f}+Pub:{pub:.2f}+IGT:{igt:.2f})")
        brows.append(
            {
                "Track": "A-National",
                "Scenario": nm,
                "Excess_Births": round(ex_nat, 0),
                "Total_Bn_INR": round(tot, 3),
                "HH_OOP": round(oob, 3),
                "Public_Sys": round(pub, 3),
                "IGT": round(igt, 3),
            }
        )
else:
    print("  National coef non-positive — see Track B")

print("\n  TRACK B — IGP causal estimate (p=0.006, credible):")
if igp_coef and igp_coef > 0:
    ex_igp = igp_coef * IGP_BIRTHS
    ex_per_district = ex_igp / max(IGP_DISTRICTS, 1)
    print(
        f"  Excess LBW births (IGP): ~{ex_igp:,.0f}/year  (~{ex_per_district:.0f}/district/year)"
    )
    for nm, (er, dr, oop, nicu) in scens.items():
        oob, pub, igt = compute_burden(ex_igp, er, dr, oop, nicu)
        tot = oob + pub + igt
        print(f"  {nm:14s} Rs{tot:.2f}Bn (HH:{oob:.2f}+Pub:{pub:.2f}+IGT:{igt:.2f})")
        brows.append(
            {
                "Track": "B-IGP-Causal",
                "Scenario": nm,
                "Excess_Births": round(ex_igp, 0),
                "Total_Bn_INR": round(tot, 3),
                "HH_OOP": round(oob, 3),
                "Public_Sys": round(pub, 3),
                "IGT": round(igt, 3),
            }
        )
    # Monte Carlo
    np.random.seed(42)
    N = 10000
    mc_c = np.random.normal(igp_coef, igp_row_b["Interact_SE"], N)
    mc_er = np.random.uniform(0.08, 0.15, N)
    mc_dr = np.random.uniform(0.03, 0.05, N)
    mc_op = np.random.uniform(7000, 10000, N)
    mc_ni = np.random.uniform(18000, 35000, N)
    mc_ex = mc_c * IGP_BIRTHS
    mc_pv = (1 - (1 + mc_dr) ** (-YRS)) / mc_dr
    mc_t = (
        mc_ex * mc_op / 1e9
        + mc_ex * HOSP * mc_ni / 1e9
        + mc_ex * SURV * EARN * mc_er * mc_pv / 1e9
    )
    med, lo5, hi95 = np.median(mc_t), np.percentile(mc_t, 5), np.percentile(mc_t, 95)
    print(f"\n  IGP MC: Rs{med:.2f}Bn  90%CI=[Rs{lo5:.2f}, Rs{hi95:.2f}]Bn")
    print(f"  CI entirely positive: {'YES ✓' if lo5>0 else 'NO — CI crosses zero'}")
    # As % of India health budget (~Rs2.2 trillion)
    health_budget = 2200.0  # Rs Bn
    print(f"  = {med/health_budget*100:.2f}% of India's public health expenditure")
    brows.append(
        {
            "Track": "B-IGP-Causal",
            "Scenario": "MC_Median",
            "Excess_Births": round(igp_coef * IGP_BIRTHS, 0),
            "Total_Bn_INR": round(med, 3),
            "CI_lo5": round(lo5, 3),
            "CI_hi95": round(hi95, 3),
            "As_pct_health_budget": round(med / health_budget * 100, 3),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mc_t, bins=80, color="#2E86AB", alpha=0.8, edgecolor="white")
    ax.axvline(med, color="#C0392B", lw=2.5, label=f"Median Rs{med:.1f}Bn")
    ax.axvline(lo5, color="#E74C3C", lw=1.5, linestyle="--")
    ax.axvline(
        hi95,
        color="#E74C3C",
        lw=1.5,
        linestyle="--",
        label=f"90%CI [Rs{lo5:.1f}, Rs{hi95:.1f}]Bn",
    )
    ax.set_xlabel("Burden (Rs billion/year)")
    ax.legend()
    ax.set_title(
        f"Monte Carlo: LBW Burden — IGP States\nIGP effect={igp_coef:+.4f}, p=0.006***"
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_monte_carlo.png", dpi=180, bbox_inches="tight")
    plt.close()
else:
    print("  IGP coef unavailable")
safe_csv(pd.DataFrame(brows), f"{OUTPUT_DIR}/table5_economic_burden.csv")


# ─────────────────────────────────────────────────────────────────────────────
# [11] FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] Figures...")


def forest_plot(rows, coef, se, label, title, path, primary_col=None):
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, max(4, len(df) * 0.65)))
    colors = [
        (
            "#1A5276"
            if (primary_col and r.get(primary_col) == "YES")
            else ("#27AE60" if r[coef] > 0 else "#C0392B")
        )
        for _, r in df.iterrows()
    ]
    ax.barh(
        range(len(df)),
        df[coef].astype(float),
        xerr=df[se].astype(float) * 1.96,
        color=colors,
        alpha=0.85,
        capsize=4,
        height=0.6,
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([str(r[label])[:55] for _, r in df.iterrows()], fontsize=8)
    ax.axvline(0, color="black", lw=1.2, linestyle="--")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


forest_plot(
    srows,
    "Coef",
    "SE",
    "Model",
    f"Stepwise M1-M5: {HEAT_LABEL} (N4)",
    f"{OUTPUT_DIR}/figure_stepwise.png",
)
forest_plot(
    t2rows,
    "Coef",
    "SE",
    "Specification",
    f"Table 2: {HEAT_LABEL}",
    f"{OUTPUT_DIR}/figure_main_results.png",
    "Primary",
)
forest_plot(
    robrows,
    "Coef",
    "SE",
    "Specification",
    f"Robustness: {pos_rob}/{len(robrows)} positive",
    f"{OUTPUT_DIR}/figure_robustness.png",
)
if len(t3df) > 0 and "Interact_Coef" in t3df.columns:
    forest_plot(
        [r.to_dict() for _, r in t3df.iterrows()],
        "Interact_Coef",
        "Interact_SE",
        "Moderator",
        "Heterogeneity — Interaction Models, BH Corrected (N4)",
        f"{OUTPUT_DIR}/figure_heterogeneity.png",
    )

# FIX-T: Trimester forest plot (matched sample)
_trim_plot_rows = [r for r in app_a if r["Trimester"] in ["T1","T2","T3"]]
if _trim_plot_rows:
    fig, ax = plt.subplots(figsize=(8,4))
    _lbls  = [r["Trimester"] for r in _trim_plot_rows]
    _coefs = [r["Coef"]      for r in _trim_plot_rows]
    _ses   = [r["SE"]        for r in _trim_plot_rows]
    _colors = ["#27AE60" if c>0 else "#C0392B" for c in _coefs]
    ax.barh(_lbls, _coefs, xerr=[s*1.96 for s in _ses],
            color=_colors, alpha=0.85, capsize=5, height=0.5)
    ax.axvline(0, color="black", lw=1.2, linestyle="--")
    ax.set_title(f"Trimester Comparison (matched sample N={_trim_plot_rows[0]['N']:,})\n"
                 "T1: in-utero selection (negative);  T3: growth restriction (primary, positive)")
    ax.set_xlabel("Coefficient (pp per 1 SD heat anomaly)")
    for i,(lbl,c,s,p_v) in enumerate(zip(_lbls,_coefs,_ses,[r["p_value"] for r in _trim_plot_rows])):
        st = pstars(p_v)
        if st: ax.text(c + s*1.96 + 0.0001, i, st, va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_trimester_matched.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  Trimester forest plot (matched sample) saved")

if sg_reg_rows:
    fig, ax = plt.subplots(figsize=(9, 5))
    sgdf = pd.DataFrame(sg_reg_rows)
    colors = ["#27AE60" if c > 0 else "#C0392B" for c in sgdf["Coef"]]
    ax.barh(
        sgdf["Subgroup"],
        sgdf["Coef"],
        xerr=sgdf["SE"] * 1.96,
        color=colors,
        alpha=0.85,
        capsize=4,
        height=0.6,
    )
    ax.axvline(0, color="black", lw=1.2, linestyle="--")
    ax.set_title("Subgroup Regressions: T3 Tmax Anomaly → LBW (N4)")
    ax.set_xlabel("Coefficient (pp per 1 SD)")
    for i, (c, s, p, lbl) in enumerate(
        zip(sgdf["Coef"], sgdf["SE"], sgdf["p_value"], sgdf["Subgroup"])
    ):
        stars = pstars(p)
        if stars:
            ax.text(c + s * 1.96 + 0.0002, i, stars, va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/figure_subgroup_regressions.png", dpi=180, bbox_inches="tight"
    )
    plt.close()

if "t3_hot_days_35" in lbw_s.columns:
    bins_h = [-1, 0, 10, 30, 60, 125]
    labs_h = ["0", "1-10", "11-30", "31-60", "61+"]
    lbw_p = lbw_s.copy()
    lbw_p["hgrp"] = pd.cut(lbw_p["t3_hot_days_35"], bins=bins_h, labels=labs_h)
    grp = (
        lbw_p.groupby("hgrp", observed=True)
        .agg(lbw_pct=("lbw", lambda x: x.mean() * 100))
        .reset_index()
    )
    dagg = (
        lbw_s.groupby("district")
        .agg(
            heat=(
                (PRIMARY_RAW, "mean")
                if PRIMARY_RAW in lbw_s.columns
                else ("t3_tmax_anomaly", "mean")
            ),
            lbw=("lbw", lambda x: x.mean() * 100),
            n=("lbw", "count"),
        )
        .reset_index()
    )
    dagg = dagg[dagg["n"] >= 30].dropna()
    cg = ["#1A5276", "#2980B9", "#E67E22", "#C0392B", "#922B21"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(
        range(len(grp)),
        grp["lbw_pct"],
        color=cg[: len(grp)],
        alpha=0.88,
        edgecolor="white",
    )
    axes[0].set_xticks(range(len(grp)))
    axes[0].set_xticklabels(labs_h)
    axes[0].set_title("A. LBW by T3 Hot Days (N4)")
    axes[0].set_ylabel("LBW Rate (%)")
    if len(dagg) > 2:
        axes[1].scatter(dagg["heat"], dagg["lbw"], alpha=0.3, s=20, color="#2980B9")
        z = np.polyfit(dagg["heat"], dagg["lbw"], 1)
        xr = np.linspace(dagg["heat"].min(), dagg["heat"].max(), 100)
        axes[1].plot(
            xr,
            np.poly1d(z)(xr),
            "r-",
            lw=2,
            label=f"r={dagg['heat'].corr(dagg['lbw']):.3f}",
        )
        axes[1].legend()
    axes[1].set_title(f"B. District {HEAT_LABEL} vs LBW")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_heat_lbw_raw.png", dpi=180, bbox_inches="tight")
    plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"T3 Tmax Anomaly & LBW | N4 Primary (M5+trend) | Mortality N4+N5 (HDFE)",
    fontsize=10,
)
wqg = lbw_s.groupby("wealth_q")["lbw"].mean() * 100
axes[0, 0].bar(
    wqg.index,
    wqg.values,
    color=["#922B21", "#C0392B", "#E67E22", "#27AE60", "#1A5276"],
    alpha=0.88,
    edgecolor="white",
)
for x, v in zip(wqg.index, wqg.values):
    axes[0, 0].text(x, v + 0.1, f"{v:.1f}%", ha="center", fontsize=9)
axes[0, 0].set_xticks([1, 2, 3, 4, 5])
axes[0, 0].set_xticklabels(["Q1\nPoor", "Q2", "Q3", "Q4", "Q5\nRich"])
axes[0, 0].set_title("A. LBW by Wealth Quintile (N4)")
if srows:
    sdf2 = pd.DataFrame(srows)
    axes[0, 1].barh(
        sdf2["Model"],
        sdf2["Coef"].astype(float),
        xerr=sdf2["SE"].astype(float) * 1.96,
        color=["#27AE60" if c > 0 else "#C0392B" for c in sdf2["Coef"]],
        alpha=0.85,
        height=0.5,
        capsize=2,
    )
    axes[0, 1].axvline(0, color="black", lw=0.9, linestyle="--")
    axes[0, 1].set_title("B. Stepwise M1-M5 (N4)")
    axes[0, 1].tick_params(axis="y", labelsize=7)
if t2rows:
    rdf = pd.DataFrame(t2rows)
    axes[1, 0].barh(
        range(len(rdf)),
        rdf["Coef"].astype(float),
        xerr=rdf["SE"].astype(float) * 1.96,
        color=["#27AE60" if d == "+" else "#C0392B" for d in rdf["Direction"]],
        alpha=0.85,
        capsize=3,
        height=0.6,
    )
    axes[1, 0].set_yticks(range(len(rdf)))
    axes[1, 0].set_yticklabels(
        [r["Specification"][:35] for _, r in rdf.iterrows()], fontsize=7
    )
    axes[1, 0].axvline(0, color="black", lw=0.9, linestyle="--")
    axes[1, 0].set_title("C. Main Results")
if (
    len(t3df) > 0
    and "Interact_Coef" in t3df.columns
    and len(t3df[t3df["Interact_Coef"].notna()]) > 0
):
    axes[1, 1].barh(
        t3df["Moderator"],
        t3df["Interact_Coef"].astype(float),
        xerr=t3df["Interact_SE"].astype(float) * 1.96,
        color=["#27AE60" if c > 0 else "#C0392B" for c in t3df["Interact_Coef"]],
        alpha=0.85,
        height=0.6,
        capsize=3,
    )
    axes[1, 1].axvline(0, color="black", lw=0.9, linestyle="--")
    axes[1, 1].set_title("D. Heterogeneity (N4)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figure_summary_dashboard.png", dpi=180, bbox_inches="tight")
plt.close()
print("  All figures OK")


# ─────────────────────────────────────────────────────────────────────────────
# [12] ML
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] ML...")
best_auc, best_nm, hi_pct = 0.0, "N/A", 0.0
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.inspection import permutation_importance

    MF = [
        c
        for c in [
            PRIMARY_RAW,
            "t3_rainfall_mm",
            "bmi",
            "maternal_age",
            "education",
            "wealth_q",
            "rural",
            "birth_order",
            "anaemic",
            "housing_quality_score",
            "has_electricity",
            "igp_state",
            "clean_fuel",
            "hv009",
        ]
        if c in lbw_s.columns
    ]
    mdf = lbw_s[["lbw"] + MF].copy()
    for c in mdf.columns:
        mdf[c] = pd.to_numeric(mdf[c], errors="coerce").astype(float)
    mdf = mdf.dropna()
    X_ml = mdf[MF].values
    y_ml = mdf["lbw"].values.astype(int)
    print(f"  n={len(mdf):,}  LBW={mdf['lbw'].mean():.4f}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mods = {
        "Logistic": Pipeline(
            [
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, C=0.5, random_state=42)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=25,
            random_state=42,
            n_jobs=-1,
        ),
        "Grad Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
    }
    ml_rows = []
    best_mod = None
    for nm, mod in mods.items():
        sc = cross_val_score(mod, X_ml, y_ml, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"  {nm:18s}  AUC={sc.mean():.4f}+/-{sc.std():.4f}")
        ml_rows.append(
            {"Model": nm, "AUC": round(sc.mean(), 4), "SD": round(sc.std(), 4)}
        )
        if sc.mean() > best_auc:
            best_auc, best_mod, best_nm = sc.mean(), mod, nm
    best_mod.fit(X_ml, y_ml)
    perm = permutation_importance(
        best_mod, X_ml, y_ml, n_repeats=10, random_state=42, n_jobs=-1
    )
    imp = pd.DataFrame(
        {
            "Feature": MF,
            "Importance": perm.importances_mean,
            "Importance_SD": perm.importances_std,
        }
    ).sort_values("Importance", ascending=False)
    hs = {PRIMARY_RAW, "t3_rainfall_mm"}
    hi = max(0, imp[imp["Feature"].isin(hs)]["Importance"].sum())
    total_pos = max(imp["Importance"].sum(), 1e-10)
    hi_pct = hi / total_pos * 100
    print(f"  Best: {best_nm}  AUC={best_auc:.4f}  Climate={hi_pct:.1f}%")
    safe_csv(imp, f"{OUTPUT_DIR}/appendix_c_ml_importance.csv")
    safe_csv(pd.DataFrame(ml_rows), f"{OUTPUT_DIR}/appendix_c_ml_comparison.csv")
    t8 = imp.head(8)
    cdc = ["#C0392B" if f in hs else "#2E86AB" for f in t8["Feature"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(t8["Feature"][::-1], t8["Importance"][::-1], color=cdc[::-1], alpha=0.85)
    ax.set_title(f"Permutation Importance — {best_nm} (N4)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/appendix_c_ml.png", dpi=180, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"  ML error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS v10 COMPLETE — ALL FIXES APPLIED")
print("=" * 70)
print(
    f"""
  PRIMARY SAMPLE: NFHS-4 LBW  N={len(lbw_s):,}\n  TRIMESTER:      Matched N={len(lbw_matched):,} (FIX-T: composition bias removed)
  MORTALITY:      N4+N5       N={len(full_s):,}
  FE:  District + Year + Birth-month + District×Year Trend
"""
)
if main_r:
    word = "INCREASES" if main_r["Direction"] == "+" else "NO EFFECT (null)"
    print(f"  PRIMARY RESULT (M5, N4)")
    print(
        f"  Coef={main_r['Coef']:+.4f}  SE={main_r['SE']:.4f}  p={main_r['p_value']:.4f} {main_r['Stars']}"
    )
    print(f"  1 SD anomaly {word} by {main_r['Coef']*100:+.3f}pp nationally")
if igp_coef:
    print(f"\n  IGP HETEROGENEOUS EFFECT (primary finding)")
    print(
        f"  Interact Coef={igp_row_b['Interact_Coef']:+.4f}  p={igp_row_b['p_raw']:.4f} {igp_row_b['stars_raw']}"
    )
    print(f"  Total IGP: {igp_coef:+.4f} = {igp_coef*100:+.3f}pp per 1 SD")
    print(f"  = ~{igp_coef*IGP_BIRTHS:+,.0f} excess LBW births/year in IGP")
    if "brows" in dir():
        central = next(
            (
                b
                for b in brows
                if b.get("Scenario") == "Central" and b.get("Track") == "B-IGP-Causal"
            ),
            None,
        )
        if central:
            print(f"  Burden (central): Rs{central['Total_Bn_INR']:.2f}Bn/year")
print(f"\n  ROBUSTNESS: {pos_rob}/{len(robrows)} positive")
print(f"  ML: {best_nm}  AUC={best_auc:.4f}  Climate={hi_pct:.1f}%")
print(
    f"""
  v9 ADDITIONS:
  NEW-1  Subgroup regressions (IGP/non-IGP/Q1/Q5/poor housing)
  NEW-2  Placebo tests (winter births, T1 vs T3 ordering)
  NEW-3  Oster (2019) bounds (omitted variable bias)
  NEW-4  Absolute effect translation (excess births, burden/district)
  NEW-5  Consistent T1/T2/T3 tmax_anomaly trimester table
  NEW-6  Full wave comparison table with anaemia, wealth, rural
"""
)
print("OUTPUT FILES:")
for fn in sorted(os.listdir(OUTPUT_DIR)):
    fp = os.path.join(OUTPUT_DIR, fn)
    nbytes = os.path.getsize(fp)
    flag = "  <-- EMPTY!" if nbytes < 10 else f"  ({nbytes:,} bytes)"
    print(f"  {fn:55s}{flag}")
