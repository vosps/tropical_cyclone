import xarray as xr
import numpy as np
from netCDF4 import Dataset
# import h5py
# import seaborn as sns
import pandas as pd
# import xesmf as xe
from pyhdf.SD import SD,SDC
import subprocess,os
from pathlib import Path
from datetime import datetime, timedelta
from cftime import DatetimeNoLeap
import math
import cftime as cf



for year in range(1998,2019):
	print(year)
	for doy in range(1,365):
		doy = "{:03d}".format(doy)
		print(doy)
		rain_dir = f'/bp1/geog-tropical/data/Obs/TRMM/TRMM_3B42/{year}/{doy}/'
		rainfall_fps = os.listdir(rain_dir)
		for f in rainfall_fps:
			rainfall_fp = rain_dir + f
			print(rainfall_fp)
			# ds = Dataset(rainfall_fp,'r',disk_format="HDF4")
			ds = SD(rainfall_fp, SDC.READ)
			# print(ds.variables)
			print(ds.datasets())
			print(ds.datasets()['precipitation'])
			print(ds.datasets().keys())
			all_metadata = ds.attributes()
			print(all_metadata)
			d1 = ds.select('precipitation')
			rain = d1[:]
			print(rain.shape)
			lat = ds.select('lat')[:]
			lon = ds.select('lon')[:]
			print(lat)
			print(lon)
			# specific_metadata = getattr(ds, 'Precipitation')
			# print(specific_metadata)
			ds.end()
			exit()
				