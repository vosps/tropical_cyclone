# import xarray as xr
from netCDF4 import Dataset


rainfall_ds = Dataset('/bp1/geog-tropical/data/Obs/TRMM/TRMM_3B42/1998/003/3B42.19980103.15.7.HDF', 'r',disk_format="HDF4")