"""
ERA5 Data Conversion Script
Converts ERA5 Excel file to pandas DataFrame and CSV
Path: /Users/sriram/Desktop/HS/dataset/7.FORWARD PROJECTIONS/era5-x0.25_timeseries_tasmax,tas_timeseries_monthly_1950-2023_mean_historical_era5_x0.25_mean.xlsx
"""

import pandas as pd
from pathlib import Path
import subprocess
import sys
import os

# =============================================================================
# Check for required packages and install if missing
# =============================================================================


def check_and_install_package(package_name):
    """Check if package is installed, if not install it"""
    try:
        __import__(package_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} not found. Installing...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
            print(f"✅ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            return False


# Check for openpyxl (required for Excel files)
if not check_and_install_package("openpyxl"):
    print(
        "Cannot proceed without openpyxl. Please install manually: pip install openpyxl"
    )
    sys.exit(1)

# =============================================================================
# CONFIGURATION - YOUR EXACT PATH
# =============================================================================

# Your exact file path
ERA5_FILE = Path(
    "/Users/sriram/Desktop/HS/dataset/7.FORWARD PROJECTIONS/era5-x0.25_timeseries_tasmax,tas_timeseries_monthly_1950-2023_mean_historical_era5_x0.25_mean.xlsx"
)

# Output directory (save in the same folder or in a processed folder)
OUTPUT_DIR = Path("/Users/sriram/Desktop/HS/dataset/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ERA5 DATA CONVERSION")
print("=" * 70)
print(f"Input file: {ERA5_FILE}")
print(f"File exists: {ERA5_FILE.exists()}")
print(f"Output directory: {OUTPUT_DIR}")

# Check if file exists
if not ERA5_FILE.exists():
    print(f"\n❌ ERROR: File not found at {ERA5_FILE}")
    print("\nChecking if file exists with different case...")

    # Try to find the file
    base_dir = Path("/Users/sriram/Desktop/HS/dataset")
    era5_files = list(base_dir.rglob("era5*.xlsx"))

    if era5_files:
        print(f"Found file: {era5_files[0]}")
        ERA5_FILE = era5_files[0]
        print(f"✅ Using: {ERA5_FILE}")
    else:
        print("❌ No ERA5 files found in dataset directory")
        exit(1)

# =============================================================================
# STEP 1: Read all sheets from the Excel file
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: Reading Excel sheets")
print("=" * 70)

try:
    # Get all sheet names
    xl = pd.ExcelFile(ERA5_FILE, engine="openpyxl")
    sheet_names = xl.sheet_names
    print(f"Found sheets: {sheet_names}")
except Exception as e:
    print(f"❌ Error reading Excel file: {e}")
    print("\nTrying with default engine...")
    try:
        xl = pd.ExcelFile(ERA5_FILE)
        sheet_names = xl.sheet_names
        print(f"Found sheets: {sheet_names}")
    except Exception as e2:
        print(f"❌ Still failing: {e2}")
        exit(1)

# Dictionary to store dataframes
dataframes = {}

# Read each sheet
for sheet in sheet_names:
    print(f"\n📊 Reading sheet: '{sheet}'")
    try:
        df = pd.read_excel(ERA5_FILE, sheet_name=sheet, engine="openpyxl")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()[:5]}...")  # Show first 5 columns
        dataframes[sheet] = df
    except Exception as e:
        print(f"❌ Error reading sheet {sheet}: {e}")

# =============================================================================
# STEP 2: Display first few rows of each sheet
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: Preview of original data")
print("=" * 70)

for sheet_name, df in dataframes.items():
    print(f"\n📋 {sheet_name.upper()} - First 5 rows:")
    print(df.head())
    print(f"\n{sheet_name.upper()} - Column names:")
    print(df.columns.tolist())

# =============================================================================
# STEP 3: Reshape from wide to long format
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: Reshaping from wide to long format")
print("=" * 70)

for sheet_name, df in dataframes.items():
    print(f"\n🔄 Processing {sheet_name}...")

    # Melt the dataframe: convert year-month columns to rows
    # The first two columns are 'code' and 'name' (identifiers)
    id_vars = ["code", "name"]

    # Get all date columns (everything after 'name')
    date_cols = [col for col in df.columns if col not in id_vars]

    print(f"   Identifier columns: {id_vars}")
    print(f"   Date columns: {len(date_cols)} (from {date_cols[0]} to {date_cols[-1]})")

    # Melt the dataframe
    df_long = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=date_cols,
        var_name="date",
        value_name=sheet_name,  # Use sheet name as value column name
    )

    print(f"   Melted shape: {df_long.shape}")

    # Extract year and month from date column
    # Date format is 'YYYY-MM' (e.g., '1950-01')
    df_long["year"] = df_long["date"].str.split("-").str[0].astype(int)
    df_long["month"] = df_long["date"].str.split("-").str[1].astype(int)

    # Drop the original date column
    df_long = df_long.drop("date", axis=1)

    # Sort by code, year, month
    df_long = df_long.sort_values(["code", "year", "month"]).reset_index(drop=True)

    print(f"   Final shape with year/month: {df_long.shape}")
    print(f"   Year range: {df_long['year'].min()}-{df_long['year'].max()}")
    print(f"   Sample data:")
    print(df_long.head(10))

    # Store in dictionary
    dataframes[sheet_name] = df_long

# =============================================================================
# STEP 4: Merge tas and tasmax if both exist
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: Merging sheets")
print("=" * 70)

if "tas" in dataframes and "tasmax" in dataframes:
    print("🔄 Merging tas and tasmax on code, name, year, month...")

    # Merge the two dataframes
    df_merged = pd.merge(
        dataframes["tas"],
        dataframes["tasmax"],
        on=["code", "name", "year", "month"],
        how="outer",
        suffixes=("_mean", "_max"),
    )

    print(f"   Merged shape: {df_merged.shape}")
    print(f"   Columns: {df_merged.columns.tolist()}")
    print(f"\n📋 Sample merged data:")
    print(df_merged.head(10))

    dataframes["merged"] = df_merged
    print("✅ Merge successful!")
else:
    print("⚠️ Either tas or tasmax sheet missing, cannot merge")
    if "tas" not in dataframes:
        print("   - Missing 'tas' sheet")
    if "tasmax" not in dataframes:
        print("   - Missing 'tasmax' sheet")

# =============================================================================
# STEP 5: Save all dataframes
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: Saving dataframes")
print("=" * 70)

for name, df in dataframes.items():
    print(f"\n💾 Saving {name}...")

    # Save as Parquet (efficient, preserves types)
    parquet_path = OUTPUT_DIR / f"era5_{name}_monthly_1950_2023.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"   ✅ Parquet saved: {parquet_path}")
    print(f"      File size: {parquet_path.stat().st_size / 1e6:.1f} MB")

    # Save as CSV (for easy viewing in Excel)
    csv_path = OUTPUT_DIR / f"era5_{name}_monthly_1950_2023.csv"
    df.to_csv(csv_path, index=False)
    print(f"   ✅ CSV saved: {csv_path}")
    print(f"      File size: {csv_path.stat().st_size / 1e6:.1f} MB")

    # Save a sample for quick viewing (first 100 rows)
    sample_path = OUTPUT_DIR / f"era5_{name}_sample.csv"
    df.head(100).to_csv(sample_path, index=False)
    print(f"   ✅ Sample saved: {sample_path}")

# =============================================================================
# STEP 6: Summary statistics
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: Summary Statistics")
print("=" * 70)

for name, df in dataframes.items():
    print(f"\n📊 {name.upper()} STATISTICS:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Unique codes: {df['code'].nunique()}")
    print(f"   Unique names: {df['name'].nunique()}")
    print(f"   Years: {df['year'].min()}-{df['year'].max()}")

    if name == "tas" and "tas" in df.columns:
        print(
            f"   Temperature (°C) - Mean: {df['tas'].mean():.2f}, Min: {df['tas'].min():.2f}, Max: {df['tas'].max():.2f}"
        )
    elif name == "tasmax" and "tasmax" in df.columns:
        print(
            f"   Max Temperature (°C) - Mean: {df['tasmax'].mean():.2f}, Min: {df['tasmax'].min():.2f}, Max: {df['tasmax'].max():.2f}"
        )
    elif name == "merged":
        if "tas_mean" in df.columns:
            print(
                f"   Mean Temperature (°C) - Mean: {df['tas_mean'].mean():.2f}, Min: {df['tas_mean'].min():.2f}, Max: {df['tas_mean'].max():.2f}"
            )
        if "tasmax_max" in df.columns:
            print(
                f"   Max Temperature (°C) - Mean: {df['tasmax_max'].mean():.2f}, Min: {df['tasmax_max'].min():.2f}, Max: {df['tasmax_max'].max():.2f}"
            )

# =============================================================================
# STEP 7: Extract India-specific data
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: India-specific data")
print("=" * 70)

if "merged" in dataframes:
    india_data = dataframes["merged"][dataframes["merged"]["code"] == "IND"]
    print(f"\n🇮🇳 INDIA DATA - {len(india_data)} rows")
    print(india_data.head(20))

    # Save India data separately
    india_path = OUTPUT_DIR / "era5_india_temperature.csv"
    india_data.to_csv(india_path, index=False)
    print(f"\n✅ India data saved to: {india_path}")

# =============================================================================
# STEP 8: Simple visualization
# =============================================================================

try:
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("STEP 8: Creating visualization")
    print("=" * 70)

    if "merged" in dataframes:
        df_plot = dataframes["merged"]

        # Get India data (code 'IND')
        india_data = df_plot[df_plot["code"] == "IND"].copy()

        if len(india_data) > 0:
            # Create a date column for plotting
            india_data["date"] = pd.to_datetime(
                india_data["year"].astype(str)
                + "-"
                + india_data["month"].astype(str)
                + "-01"
            )

            fig, axes = plt.subplots(2, 1, figsize=(15, 8))

            # Plot mean temperature
            axes[0].plot(
                india_data["date"],
                india_data["tas_mean"],
                linewidth=0.8,
                color="red",
                alpha=0.7,
                label="Mean Temperature",
            )
            axes[0].set_title("India - Mean Temperature (1950-2023)", fontsize=12)
            axes[0].set_ylabel("Temperature (°C)")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            # Plot max temperature
            axes[1].plot(
                india_data["date"],
                india_data["tasmax_max"],
                linewidth=0.8,
                color="orange",
                alpha=0.7,
                label="Max Temperature",
            )
            axes[1].set_title("India - Maximum Temperature (1950-2023)", fontsize=12)
            axes[1].set_ylabel("Temperature (°C)")
            axes[1].set_xlabel("Year")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            plt.tight_layout()

            # Save plot
            plot_path = OUTPUT_DIR / "era5_india_temperature.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"✅ Plot saved: {plot_path}")

            # Show plot (if in interactive environment)
            plt.show()
        else:
            print("No India data found for plotting")

except ImportError:
    print("matplotlib not installed - skipping visualization")
except Exception as e:
    print(f"Visualization error: {e}")

# =============================================================================
# DONE
# =============================================================================

print("\n" + "=" * 70)
print("✅✅✅ CONVERSION COMPLETE! ✅✅✅")
print("=" * 70)
print(f"\n📁 All files saved to: {OUTPUT_DIR}")
print("\n📄 Files created:")
for name in dataframes.keys():
    print(f"   • era5_{name}_monthly_1950_2023.parquet")
    print(f"   • era5_{name}_monthly_1950_2023.csv")
    print(f"   • era5_{name}_sample.csv")
print("\n📊 India-specific file:")
print("   • era5_india_temperature.csv")
print("\n📈 Plot file:")
print("   • era5_india_temperature.png")

print("\n" + "=" * 70)
print("🎉 NEXT STEPS:")
print("=" * 70)
print(
    """
To load the data in your analysis:

import pandas as pd

# Load merged data
df = pd.read_parquet('dataset/processed/era5_merged_monthly_1950_2023.parquet')

# Load India-specific data
india_df = pd.read_csv('dataset/processed/era5_india_temperature.csv')

# Quick check
print(df.head())
print(india_df.head())
"""
)
