"""
	2. this script takes the filepaths generated in 1. and locates the eye of the TC in IMERG or MSWEP to extract the tc

MSWEP latitude data needs to be flipped in advance in order for lats to be in ascending order

https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Read%20IMERG%20Data%20Using%20Python
"""
print('running')
import glob
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import h5py
from multiprocessing import Pool
import os

def process_apply(x):
	# define variables
	print('doing other process')
	filepath = x.loc['filepath_imerg']
	filepath = glob.glob(filepath)[0]
	centre_lat = x.loc['lat']
	centre_lon = x.loc['lon']
	name = x.loc['name']
	cat = x.loc['sshs']
	time = x.loc['hour']
	sid = x.loc['sid']
	i = x[0]

	# open file
	d = Dataset(filepath, 'r')
	lat = d['Grid'].variables['lat'][:] #lat
	lon = d['Grid'].variables['lon'][:] #lon

	# clip to location
	lat_lower_bound = (np.abs(lat-centre_lat+5.)).argmin()
	lat_upper_bound = (np.abs(lat-centre_lat-5.)).argmin()
	lon_lower_bound = (np.abs(lon-centre_lon+5.)).argmin()
	lon_upper_bound = (np.abs(lon-centre_lon-5.)).argmin()

	data = d['Grid'].variables['precipitationCal'][0,lon_lower_bound:lon_upper_bound,lat_lower_bound:lat_upper_bound]
	lat = lat[lat_lower_bound:lat_upper_bound]
	lon = lon[lon_lower_bound:lon_upper_bound]
	d.close()

	# precip = np.transpose(precip)
	da = xr.DataArray(data, 
					  dims=("x", "y"), 
					  coords={"x": lon, "y": lat},
					  attrs=dict(description="Total Precipitation",units="mm"),
					  name = 'precipitation')

	da.to_netcdf('/user/work/al18709/tropical_cyclones/imerg' + str(name) + '_' + str(sid) + '_hour-' + str(time) + '_idx-' + str(i) + '_cat-' + str(int(cat))+ '.nc')
	print('%s saved!' % filepath)

def process_apply_mswep(x):
	# define variables
	
	result = 1
	filepath = x.loc['filepath_mswep']

	# remove filepaths which relate to timestamps we don't have in MSWEP
	if filepath[65:67] not in ['00','03','06','09','12','15','18','21']:
		result = 0

	if result == 1:
		print(filepath)
		if glob.glob(filepath) == []:
			with open('logs/error_filepaths.txt', 'w') as f:
					f.write(filepath)
					f.write('\n')

		else:
			filepath = glob.glob(filepath)[0]
			centre_lat = x.loc['lat']
			centre_lon = x.loc['lon']
			name = x.loc['name']
			cat = x.loc['sshs']
			time = x.loc['hour']
			sid = x.loc['sid']
			basin = x.loc['basin']
			i = x[0]

			# open file
			d = Dataset(filepath, 'r')
			lat = d.variables['lat'][:] #lat
			lon = d.variables['lon'][:] #lon
			
			# clip to location
			lat_lower_bound = (np.abs(lat-centre_lat+5.)).argmin()
			lat_upper_bound = (np.abs(lat-centre_lat-5.)).argmin()
			lon_lower_bound = (np.abs(lon-centre_lon+5.)).argmin()
			lon_upper_bound = (np.abs(lon-centre_lon-5.)).argmin()

			# work on edge cases
			"""
			if lat_lower_bound - lat_upper_bound != 100:
				lat_lower_bound = lat_upper_bound + 100
			"""

			# if lon lower bound is over centre, splice
			print('centre lon: ',centre_lon)
			print('lower bound: ',lon_lower_bound)
			print('upper bound: ',lon_upper_bound)
			if centre_lon > 175: 
				print('goes over centre')
				diff = lon_upper_bound - lon_lower_bound
				second_upper_bound = 100 - diff

				data1 = d.variables['precipitation'][0,lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
				data2 = d.variables['precipitation'][0,lat_lower_bound:lat_upper_bound,0:second_upper_bound]
				lat = lat[lat_lower_bound:lat_upper_bound]
				lon1 = lon[lon_lower_bound:lon_upper_bound]
				lon2 = lon[0:second_upper_bound]
				data = np.concatenate((data1,data2),axis=1)
				lon = np.concatenate((lon1,lon2))
			elif centre_lon < -175:
				diff = lon_upper_bound - lon_lower_bound
				second_upper_bound = 100 - diff
				data1 = d.variables['precipitation'][0,lat_lower_bound:lat_upper_bound,-second_upper_bound:-1]
				data2 = d.variables['precipitation'][0,lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
				lat = lat[lat_lower_bound:lat_upper_bound]
				lon1 = lon[-second_upper_bound:-1]
				lon2 = lon[lon_lower_bound:lon_upper_bound]
				
				data = np.concatenate((data1,data2),axis=1)
				lon = np.concatenate((lon1,lon2))
			else:
				data = d.variables['precipitation'][0,lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
				lat = lat[lat_lower_bound:lat_upper_bound]
				lon = lon[lon_lower_bound:lon_upper_bound]

			"""
			lon_upper_og = lon_upper_bound
			if lon_upper_bound - lon_lower_bound != 100:
				lon_upper_og = lon_upper_bound
				lon_upper_bound = lon_lower_bound + 100
			lon_bool = False

			# data = d.variables['precipitation']
			if lon_upper_bound >= 3599:
				lon_idx = []
				lon_idx = lon_idx + list(range(0,lon_upper_bound - 3599))
				if list(range(lon_lower_bound,3599)) != []:
					lon_idx = lon_idx + list(range(lon_lower_bound,3599))
				
				lon_bool = []
				for i in range(len(lon)):
					if i in lon_idx:
						lon_bool.append(True)
					else:
						lon_bool.append(False)
				data = d.variables['precipitation'][0,lat_upper_bound:lat_lower_bound,lon_bool]

			else:
				data = d.variables['precipitation'][0,lat_upper_bound:lat_lower_bound,lon_lower_bound:lon_upper_bound]
			lat = lat[lat_upper_bound:lat_lower_bound]
			if lon_bool:
				lon = lon[lon_bool]
			else:
				lon = lon[lon_lower_bound:lon_upper_bound]"""
			
			
			d.close()
			print(len(lon))
			print(len(lat))
			if (len(lon) != 100) or (len(lat) != 100): # TODO: figure out why this happens
				print('dimensions do not match')
			else:
				# precip = np.transpose(precip)
			
				da = xr.DataArray(data, 
								dims=("x", "y"), 
								coords={"x": lon, "y": lat},
								attrs=dict(description="Total Precipitation",units="mm"),
								name = 'precipitation')
				print(da)
				da.to_netcdf('/user/work/al18709/tropical_cyclones/mswep/' + str(name) + '_' + str(sid) + '_hour-' + str(time) + '_idx-' + str(i) + '_cat-' + str(int(cat)) + '_basin-' + str(basin) + '.nc')
				print('%s saved!' % filepath)
				# TODO: flip lats
	else:
		print('%s not saved!' % filepath)

def process(df):
	print('doing process...')
	res = df.apply(process_apply,axis=1)
	return res

def process_mswep(df):
	print('doing process...')
	res = df.apply(process_apply_mswep,axis=1)
	return res

if __name__ == '__main__':
	dataset = 'mswep' # or imerg
	df = pd.read_csv('/user/work/al18709/ibtracks/tc_files.csv')
	df_split = np.array_split(df, 32)
	p = Pool(processes=32)
	print('opened csv!')
	print('pooling processes...')

	# run different functions depending on which dataset we use
	if dataset == 'imerg':
		pool_results = p.map(process, df_split)
	elif dataset == 'mswep':
		pool_results = p.map(process_mswep, df_split)
	# data = pd.concat(p.map(process, df_split))
	print('results pooled')
	p.close()

	p.join()

	# TODO fix error in line 20 where somehow filepath doesn't exist?