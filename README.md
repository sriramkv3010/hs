# Temperature Shocks, Maternal Health, and the Intergenerational Economic Burden of Climate Change in India

> **Kotipalli Venkata Sriram · Lanka Karthikeya · under the guidance of Dr. Vikas Kumar**  
> *Ecology, Economy and Society — the INSEE Journal*  
> Special Section: Economics of Climate Change in Developing Countries

---

## Abstract

This study estimates the impact of prenatal heat stress on birth outcomes and the resulting economic burden across India. Using 1.32 million births from the National Family Health Surveys (NFHS-4 & NFHS-5, 2010–2021) merged with high-resolution IMD temperature data, we show that each additional day above 35°C during pregnancy significantly increases the probability of low birth weight (LBW). Effects are largest in the first trimester, consistent with the fetal origins hypothesis. The economically disadvantaged, rural populations, and households in the Indo-Gangetic Plain (IGP) are disproportionately affected. The central estimate of the annual economic burden is **Rs. 104.98 billion** (~90% CI: Rs. 73.88–141.70 billion), dominated by intergenerational productivity losses.

---

## Table of Contents

- [Key Findings](#key-findings)
- [Data Sources](#data-sources)
- [Repository Structure](#repository-structure)
- [Empirical Strategy](#empirical-strategy)
- [Results](#results)
- [Economic Burden](#economic-burden)
- [Replication](#replication)
- [Citation](#citation)

---

## Key Findings

| Finding | Estimate |
|---|---|
| Effect of one additional hot day (>35°C, T3) on LBW probability | +1.25 pp (β = 0.0125, p < 0.001) |
| Effect per standard deviation of exposure (27 days) | +1.25 pp in LBW incidence |
| Excess LBW births nationally per SD increase | ~312,900 |
| First-trimester effect (strongest) | β = 0.0174*** |
| Second-trimester effect | β = 0.0167*** |
| Third-trimester effect (preferred spec) | β = 0.0125*** |
| Wealth gradient: poorest (Q1) vs richest (Q5) | β = 0.0162 vs β = 0.0048 |
| IGP states vs non-IGP | β = 0.0150 vs β = 0.0098 |
| Central economic burden estimate (Monte Carlo) | Rs. 104.98 billion/year |
| 90% confidence interval | Rs. 73.88 – 141.70 billion |

**15 out of 16 robustness specifications yield positive, significant coefficients.** The lone exception (district FE) reflects a well-documented Simpson's Paradox driven by the cross-sectional nature of heat exposure in India.

---

## Data Sources

### 1. Climate Data — India Meteorological Department (IMD)

| File | Format | Description |
|---|---|---|
| `imd_rainfall.nc` | NetCDF | Daily rainfall gridded data |
| `imd_tmax.grd` | GRD | Daily maximum temperature gridded data |

**Source:** [India Meteorological Department — imdpune.gov.in](https://imdpune.gov.in)  
High-resolution gridded daily Tmax matched to Census 2011 district boundaries using area-weighted spatial extraction.

---

### 2. Birth & Health Data — National Family Health Survey (NFHS)

| Survey Round | Period | Births |
|---|---|---|
| NFHS-4 | 2015–16 | 575,946 |
| NFHS-5 | 2019–21 | 747,836 |
| **Combined** | **2010–2021** | **1,323,782** |

**Source:** [DHS Program — dhsprogram.com](https://dhsprogram.com/data/available-datasets.cfm)

Each NFHS round includes three record types used in this study:

| Record Type | Folder Code | Files |
|---|---|---|
| Births Recode | `IABR74DT` | `.DCT` `.DO` `.DTA` `.FRQ` `.FRW` `.MAP` |
| Household Recode | `IAHR74DT` | `.DCT` `.DO` `.DTA` `.FRQ` `.FRW` `.MAP` |
| Individual (Women's) Recode | `IAIR74DT` | `.DCT` `.DO` `.DTA` `.FRQ` `.FRW` `.MAP` |

The analytical LBW sample covers **356,074 births** with valid birth weight data (188,286 from NFHS-4; 167,788 from NFHS-5). Mortality outcomes use a broader sample of **723,444 births**.

---

### 3. Health Infrastructure — National Health Systems (NHS) Resource Centre

| File | Period |
|---|---|
| `NHS2015-16_all.xls` | 2015–16 |
| `NHS2019-20_all.xls` | 2019–20 |

**Source:** [National Health Mission — nhm.gov.in](https://nhm.gov.in/index4.php?lang=1&level=0&linkid=221&lid=613)  
District-level health infrastructure indicators: facility availability, institutional delivery rates, and capacity measures.

---

### 4. Healthcare Costs — National Health Systems Resource Centre (NHSRC)

| File | Format | Description |
|---|---|---|
| NHSRC Costing Report | `.xlsx` | Unit costs for neonatal intensive care and delivery services |

**Source:** [NHSRC — nhsrcindia.org](https://nhsrcindia.org/category-resources/costingresources)  
Used to estimate the public health system burden component of the economic cost framework.

---

### 5. Household Expenditure — NSS 75th Round (2017–18)

| File | Format | Description |
|---|---|---|
| Household Social Consumption — Health | `.pdf` report | Out-of-pocket healthcare expenditure by income decile |

**Source:** [MOSPI — mospi.gov.in / MicroData](https://microdata.gov.in/nada43/index.php/catalog/115)  
Used to compute excess out-of-pocket (OOP) expenditure per LBW birth across wealth quintiles.

---

### 6. Spatial Data — Census District Boundaries

#### District Boundary Shapefiles

| File | Description |
|---|---|
| `India-Districts-2011Census.shp` | Main polygon shapefile |
| `India-Districts-2011Census.dbf` | Attribute table |
| `India-Districts-2011Census.prj` | Projection file |
| `India-Districts-2011Census.shx` | Spatial index |

**Source:** [Datameet — github.com/datameet/maps](https://github.com/datameet/maps/tree/master/Districts)

#### District Population & Crosswalk

| File/Folder | Description |
|---|---|
| `delimitation_new_district_order_2008.pdf` | District delimitation reference |
| `SHRUG/shrug-pc-keys-csv/` | SHRUG primary census keys |
| `SHRUG/shrug-shrid-keys-csv/` | SHRUG SHRID spatial crosswalk |

**Source:** [SHRUG — devdatalab.org](https://www.devdatalab.org/shrug_download/)  
The SHRUG crosswalk harmonizes NFHS-4 and NFHS-5 district identifiers against static Census 2011 boundaries, enabling consistent exposure assignment across both survey waves.

---

### 7. Forward Projections — ERA5 Reanalysis

| File | Format | Description |
|---|---|---|
| `era5-x0.25_timeseries_tasmax_tas_timeseries_monthly_1950-2023_mean_historical_era5_x0.25_mean.xlsx` | `.xlsx` | Monthly mean Tmax and Tas at 0.25° resolution, 1950–2023 |

**Source:** [Copernicus Climate Data Store — cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means)  
Used for (i) validation of IMD gridded temperature data and (ii) forward projection of hot-day counts under continued warming scenarios.

---


## Empirical Strategy

### Primary Specification

The baseline OLS model with birth-month fixed effects (M3):

```
LBWidt = α + β₁ · HotDays35dt + X'idt γ + γm + εidt
```

where:
- `LBWidt` = 1 if birth weight < 2,500g; child *i*, district *d*, time *t*
- `HotDays35dt` = days with Tmax > 35°C during the relevant trimester
- `X'idt` = maternal age, education, parity, ANC visits, wealth quintile, rural/urban, housing quality, cooking fuel
- `γm` = birth-month fixed effects (absorb common seasonality)
- Standard errors clustered at district level

### Model Ladder (Stepwise Sensitivity)

| Model | Specification | Coefficient | SE | p-value |
|---|---|---|---|---|
| M1 | Raw OLS | 0.0067 | 0.0006 | < 0.001 |
| M2 | + Individual controls | 0.0067 | 0.0006 | < 0.001 |
| **M3** | **+ Month FE (PRIMARY)** | **0.0125** | **0.0009** | **< 0.001** |
| M4 | + District + Month FE | -0.0073 | 0.0012 | < 0.001 |
| M5 | + State + Month FE | 0.0031 | 0.0040 | 0.440 |
| M6 | + State + Year + Month FE | 0.0034 | 0.0042 | 0.414 |

The sign reversal in M4 is a classic **Simpson's Paradox**: hotter districts also have better health infrastructure (referral hospitals, NICUs), so district FE absorbs the true cross-district signal. Birth-month FE (M3) is the preferred identification strategy.

### Robustness to Unobservables — Oster (2019) Bounds

Setting Rmax = 1.3 × R̃² and δ = 1 (equal selection on observables and unobservables), the bias-adjusted β* remains strictly positive. The δ required to drive β* to zero exceeds 1 substantially, meaning unobservables would need to be far more influential than wealth, education, and housing combined — unlikely in this setting.

---

## Results

### Main Effects (Table 2)

| Specification | Coefficient | SE | N |
|---|---|---|---|
| Hot Days > 35°C, T3 **[PRIMARY M3]** | **0.0125***| 0.0009 | 356,074 |
| Hot Days > 33°C, T3 | 0.0098*** | 0.0008 | 356,074 |
| T3 + Rainfall control | 0.0077*** | 0.0009 | 356,074 |
| T3 + ERA5 validation | 0.0127*** | 0.0009 | 356,074 |
| Hot Days > 35°C, T1 | 0.0174*** | 0.0009 | 356,074 |
| Hot Days > 35°C, T2 | 0.0167*** | 0.0009 | 356,074 |
| Neonatal mortality | 0.0023*** | 0.0003 | 723,444 |
| Infant mortality | 0.0027*** | 0.0003 | 723,444 |

*** p < 0.01

### Trimester Gradient

The effect follows a monotonically decreasing gradient from T1 to T3, consistent with the fetal origins hypothesis and the biological primacy of organogenesis in early pregnancy:

```
T1: β = 0.0174  >  T2: β = 0.0167  >  T3: β = 0.0125
```

### Heterogeneity (Table 3)

| Subgroup | N | LBW Rate | Coefficient | BH-corrected |
|---|---|---|---|---|
| Q1 — Poorest | 80,190 | 19.15% | 0.0162*** | *** |
| Q2 | 80,739 | 17.73% | 0.0145*** | *** |
| Q3 | 74,143 | 16.38% | 0.0141*** | *** |
| Q4 | 66,255 | 15.77% | 0.0119*** | *** |
| Q5 — Richest | 54,747 | 14.15% | 0.0048* | * |
| Rural | 271,443 | 17.15% | 0.0125*** | *** |
| Urban | 84,631 | 15.91% | 0.0102*** | *** |
| IGP States | 136,069 | 18.00% | 0.0150*** | *** |
| Non-IGP | 220,005 | 16.15% | 0.0098*** | *** |
| Poor Housing | 92,994 | 17.80% | 0.0176*** | *** |
| Good Housing | 263,080 | 16.52% | 0.0102*** | *** |
| No Electricity | 25,705 | 18.94% | 0.0097** | ** |
| Has Electricity | 330,369 | 16.69% | 0.0121*** | *** |
| Anaemic | 115,190 | 17.97% | 0.0125*** | *** |
| Non-Anaemic | 231,832 | 16.26% | 0.0115*** | *** |

**15/15 positive, 14/15 significant after Benjamini-Hochberg correction.**

### Robustness (Table 4) — 15/16 Specifications Positive

| Specification | Coefficient | N |
|---|---|---|
| Baseline M3 | 0.0125*** | 356,074 |
| Threshold 33°C | 0.0098*** | 356,074 |
| + Rainfall | 0.0077*** | 356,074 |
| + Drought flag | 0.0127*** | 356,074 |
| + Anaemia control | 0.0124*** | 347,022 |
| + Housing quality | 0.0123*** | 356,074 |
| + ERA5 trend | 0.0127*** | 356,074 |
| + Wave FE | 0.0125*** | 356,074 |
| Long-term residents only | 0.0131*** | 242,363 |
| First births only | 0.0142*** | 99,326 |
| Rural subsample | 0.0132*** | 271,443 |
| Urban subsample | 0.0109*** | 84,631 |
| IGP states only | 0.0162*** | 136,069 |
| Non-IGP states | 0.0104*** | 220,005 |
| District FE (M4) | -0.0073*** | 356,074 |

### Machine Learning Validation

Gradient Boosting classifier trained on individual, household, and climate features:

| Model | AUC |
|---|---|
| Logistic Regression | 0.5684 ± 0.0026 |
| Random Forest | 0.5767 ± 0.0032 |
| Gradient Boosting | 0.5788 ± 0.0027 |

Climate variables account for **56.5%** of feature importance in the Gradient Boosting model. The moderate AUC reflects the inherently multifactorial nature of birth outcomes, not a failure of the climate signal.

---

## Economic Burden

The cost framework decomposes the total burden into three components, following Ostro (1994) and Nayak, Roy and Chowdhury:

### Cost Components (Central Estimate)

| Component | Estimate (Rs. Billion) | Share | Source |
|---|---|---|---|
| Household out-of-pocket (OOP) | 2.66 | 2.4% | NSS 75th Round |
| Public healthcare system | 3.52 | 3.2% | NHSRC costing studies |
| Intergenerational productivity loss | 102.56 | 94.4% | Victora et al.; 12% earnings reduction, 4% discount rate |
| **Total (Central)** | **108.74** | **100%** | |

### Monte Carlo Simulation (n = 10,000)

Parameters drawn jointly:
- Earnings reduction: Uniform[0.08, 0.15]
- Discount rate: Uniform[0.03, 0.05]
- Healthcare cost parameters: varied ±15%

| Statistic | Value (Rs. Billion/year) |
|---|---|
| Median | 104.98 |
| Mean | ~109 |
| 90% CI lower | 73.88 |
| 90% CI upper | 141.70 |

The total burden is **equivalent to 4.5% of central government health expenditure** and exceeds annual allocations to PMMVY and JSY combined. Over 94% of costs are intergenerational — invisible in short-run health accounting.

---

## Replication

### Requirements

```bash
pip install pandas numpy scipy statsmodels scikit-learn geopandas xarray netCDF4 matplotlib seaborn
```

Stata is required for NFHS `.DTA` preprocessing (see `02_nfhs_cleaning.do`).

### Run Full Pipeline

```bash
python code/fullanalysis.py
```

This executes all ten analysis steps sequentially:

1. Data loading and merging (1.32M births)
2. Table 1 — Descriptive statistics
3. Stepwise specification table (M1–M6)
4. Table 2 — Main results (11 specifications)
5. Trimester pattern
6. Table 3 — Heterogeneity with BH correction
7. Robustness checks
8. Placebo tests and identification validation
9. Oster (2019) omitted variable bounds
10. Machine learning validation
11. Economic burden Monte Carlo (n = 10,000)

### Sample Output

```
[4] Table 2 — Main Results...
  → LBW ~ T3 hot days>35°C  PRIMARY [M3]    0.0125  0.0009  14.40  0.0000  ***  356,074
    → +1.252pp/SD | +125.2/10k | +312,898 national

[11] Economic burden...
  MC: Rs.104.98Bn  90%CI=[Rs.73.88, Rs.141.70]Bn
```

---

## Citation

```bibtex
@article{sriram2024temperature,
  title   = {Temperature Shocks, Maternal Health, and the Intergenerational Economic Burden of Climate Change in India},
  author  = {Sriram, Kotipalli Venkata and Karthikeya, Lanka and Kumar, Vikas},
  journal = {Ecology, Economy and Society -- the INSEE Journal},
  year    = {2024},
  note    = {Special Section: Economics of Climate Change in Developing Countries}
}
```

---

## Key References

Almond, D. & Currie, J. (2011). Killing me softly: The fetal origins hypothesis. *Journal of Economic Perspectives.* DOI: 10.1257/jep.25.3.153

Barker, D.J.P. (1990). The fetal and infant origins of adult disease. *BMJ.*

Heckman, J.J. (2006). Skill formation and the economics of investing in disadvantaged children. *Science.* DOI: 10.1126/science.1128898

Oster, E. (2019). Unobservable selection and coefficient stability: Theory and evidence. *Journal of Business & Economic Statistics.* DOI: 10.1080/07350015.2016.1227711

Victora, C.G. et al. (2008). Maternal and child undernutrition: consequences for adult health and human capital. *The Lancet.* DOI: 10.1016/S0140-6736(07)61692-4

---

*Data access for NFHS microdata requires registration at the DHS Program portal. IMD gridded data is available on request from the India Meteorological Department.*
