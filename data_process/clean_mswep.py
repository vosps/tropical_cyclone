"""
This dataset flips the latitude and logitude of mswep data
"""

import glob
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import subprocess
import os

# open imerg
filepath = '/bp1store/geog-tropical/data/Obs/IMERG/Final/3B-DAY.MS.MRG.3IMERG.20210531-S000000-E235959.V06.nc4'
d = Dataset(filepath, 'r')
lat = d.variables['lat'][:] #lat
lon = d.variables['lon'][:] #lon
d.close()

print(lat)
print(lon)

filepath = '/bp1store/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2012220.06.nc' #changing to invertlat might have fixed the mswep issue
d = Dataset(filepath, 'r')
lat = d.variables['lat'][:] #lat
lon = d.variables['lon'][:] #lon
d.close()
print(lat)
print(lon)

filepaths = glob.glob('/bp1store/geog-tropical/data/Obs/MSWEP/3hourly/*.nc')
filepaths_complete = os.listdir('/bp1store/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/')


for fp in filepaths:
	filename = fp[47:]
	if filename in filepaths_complete:
		print(filename)
		continue
	outfile = fp[:39] + '3hourly_invertlat' + fp[46:]
	cdo_cmd = ['cdo','invertlat',fp,outfile]
	print(' '.join(cdo_cmd))
	ret = subprocess.call(cdo_cmd)
