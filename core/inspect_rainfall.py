"""
Run this FIRST to identify what format your .nc files actually are.
python3 inspect_rainfall.py
"""

import os
import glob
import struct

IMD_RAINFALL_DIR = os.path.expanduser("~/Desktop/HS/dataset/2.IMD/imd_rainfall")

nc_files = sorted(glob.glob(os.path.join(IMD_RAINFALL_DIR, "imd_rf25_*.nc")))
print(f"Found {len(nc_files)} files")

# Check magic bytes of first few files
print("\n--- Magic bytes (identifies true file format) ---")
for f in nc_files[:5]:
    with open(f, "rb") as fh:
        magic = fh.read(8)
    hex_magic = magic.hex()
    ascii_magic = "".join(chr(b) if 32 <= b < 127 else "." for b in magic)

    # Identify format
    if magic[:4] == b"CDF\x01":
        fmt = "NetCDF3 Classic"
    elif magic[:4] == b"CDF\x02":
        fmt = "NetCDF3 64-bit"
    elif magic[:8] == b"\x89HDF\r\n\x1a\n":
        fmt = "NetCDF4 / HDF5"
    elif magic[:4] == b"GRIB":
        fmt = "GRIB"
    elif magic[:2] in [b"\x1f\x8b"]:
        fmt = "GZIP compressed"
    else:
        fmt = "UNKNOWN"

    size_mb = os.path.getsize(f) / 1e6
    print(
        f"  {os.path.basename(f):25s} | {fmt:20s} | {size_mb:.2f} MB | hex: {hex_magic[:16]}"
    )

# Try all possible engines on first file
print("\n--- Trying all xarray engines ---")
import xarray as xr

first_file = nc_files[0]

for engine in ["netcdf4", "scipy", "h5netcdf", "pydap"]:
    try:
        ds = xr.open_dataset(first_file, engine=engine)
        print(f"  engine='{engine}': SUCCESS")
        print(f"    Variables: {list(ds.data_vars)}")
        print(f"    Dims: {dict(ds.dims)}")
        print(f"    Coords: {list(ds.coords)}")
        for var in ds.data_vars:
            print(f"    '{var}' shape: {ds[var].shape}, dtype: {ds[var].dtype}")
            vals = ds[var].values
            print(
                f"    '{var}' min: {vals[~(vals!=vals)].min() if len(vals[~(vals!=vals)])>0 else 'N/A'}"
            )
        ds.close()
        break
    except Exception as e:
        print(f"  engine='{engine}': FAILED — {str(e)[:80]}")

# Also try scipy directly
print("\n--- Trying scipy.io.netcdf ---")
try:
    from scipy.io import netcdf_file

    f = netcdf_file(first_file, "r", mmap=False)
    print(f"  scipy: SUCCESS")
    print(f"  Variables: {list(f.variables.keys())}")
    for var in f.variables:
        v = f.variables[var]
        print(f"  '{var}': shape={v.shape}, dtype={v.dtype}")
        data = v.data
        print(f"  '{var}': min={data.min():.3f}, max={data.max():.3f}")
    f.close()
except Exception as e:
    print(f"  scipy: FAILED — {e}")
