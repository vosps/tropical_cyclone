"""
Separate monthly era5 data into 3 hourly files

from : /bp1store/geog-tropical/data/ERA-5/hour/specific_humidity
to : /user/home/al18709/work/era5/specific_humidity

"""

# import modules
import glob
# import pandas as pd
# from netCDF4 import Dataset
# import numpy as np
# import xarray as xr
import subprocess
# import os

def generate_yrmonths():
	years = range(2017,2022)
	months = ['01','02','03','04','05','06','07','08','09','10','11','12']
	yrmonths = [ int("%s%s" % (year,month)) for year in years for month in months]
	return yrmonths

# generate year months
year_months = generate_yrmonths()

# generate list of filepaths
filepaths_era = ['/bp1store/geog-tropical/data/ERA-5/hour/specific_humidity/ERA5_q_3hourly_1deg_%s.nc' % ym for ym in year_months]
# filepaths_tmp = ['/user/home/al18709/work/era5/specific_humidity/ERA5_q_3hourly_1deg_%s.nc' % ym for ym in year_months]
obases_day = ['/bp1store/geog-tropical/data/ERA-5/hour/specific_humidity/day/ERA5_q_3hourly_1deg_%s' % ym for ym in year_months]
obases_hour = ['/bp1store/geog-tropical/data/ERA-5/hour/specific_humidity/hour/ERA5_q_3hourly_1deg_%s' % ym for ym in year_months]
print(filepaths_era)

# dates, datasets = zip(*ds.resample(time='1D').mean('time').groupby('time'))
# filenames = [pd.to_datetime(date).strftime('%Y.%m.%d') + '.nc' for date in dates]
# xr.save_mfdataset(datasets, filenames)

for i,fp in enumerate(filepaths_era):
	# tmpfile = filepaths_tmp[i]
	# cp_cmd = ['cp',fp,tmpfile] # normal resolution
	# print(' '.join(cp_cmd))
	# ret = subprocess.call(cp_cmd)
	# if not ret==0:
	# 	raise Exception('Error with copy command')
	
	obase = obases_day[i]
	cdo_cmd = ['cdo','splitday',fp,obase] # normal resolution
	print(' '.join(cdo_cmd))
	ret = subprocess.call(cdo_cmd)
	if not ret==0:
		raise Exception('Error with cdo command')

	for fp2 in glob.glob('/bp1store/geog-tropical/data/ERA-5/hour/specific_humidity/day/*.nc'):
		obase = '/bp1store/geog-tropical/data/ERA-5/hour/specific_humidity/hour/' + fp2[-31:-2]
		# obase = obases_hour[i] TODO: define this to include day label
		cdo_cmd = ['cdo','splithour',fp2,obase] # normal resolution
		print(' '.join(cdo_cmd))
		ret = subprocess.call(cdo_cmd)
		if not ret==0:
			raise Exception('Error with cdo command')
		print(fp, ' saved!')
	
		rm_cmd = ['rm',fp2] 
		print(' '.join(rm_cmd))
		ret = subprocess.call(rm_cmd)
		if not ret==0:
			raise Exception('Error with rm command') # TODO: check this is deleting
		



