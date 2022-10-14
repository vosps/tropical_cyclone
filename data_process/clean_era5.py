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
# filepath = '/bp1store/geog-tropical/data/ERA-5/hour/precipitation/ERA5_precipitation_3hrly_202112.nc'
# d = Dataset(filepath, 'r')
# lat = d.variables['latitude'][:] #lat
# lon = d.variables['longitude'][:] #lon
# d.close()

# print(lat)
# print(lon)

# filepath = '/bp1store/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2012220.06.nc' #changing to invertlat might have fixed the mswep issue
# d = Dataset(filepath, 'r')
# lat = d.variables['lat'][:] #lat
# lon = d.variables['lon'][:] #lon
# d.close()
# print(lat)
# print(lon)

filepaths = glob.glob('/bp1store/geog-tropical/data/ERA-5/hour/precipitation_ensemble_members/*.nc')
filepaths_complete = os.listdir('/bp1store/geog-tropical/data/ERA-5/hour/precipitation_invertlat_em/')


for fp in filepaths:
	filename = fp[52:]
	if filename in filepaths_complete:
		print(filename)
		continue
	outfile1 = '/bp1store/geog-tropical/data/ERA-5/hour/precipitation_invertlat_em/tmp.nc'
	# outfile2 = fp[:53] + '_invertlat' + fp[53:]
	outfile2 = '/bp1store/geog-tropical/data/ERA-5/hour/precipitation_invertlat_em/' + fp[-34:]
	
	cdo_cmd = ['cdo','invertlat',fp,outfile1]
	print(' '.join(cdo_cmd))
	ret = subprocess.call(cdo_cmd)

	# convert to mm
	# cdo_cmd = ['cdo','-r','-b','32','mulc,1000',outfile,outfile]
	cdo_cmd = ['cdo','-r','-b','32','mulc,1000','-setattribute,tp@units=mm',outfile1,outfile2]
	print(' '.join(cdo_cmd))
	ret = subprocess.call(cdo_cmd)
	os.remove(outfile1)
