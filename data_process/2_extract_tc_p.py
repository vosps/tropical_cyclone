"""
	2. this script takes the filepaths generated in 1. and locates the eye of the TC in IMERG or MSWEP to extract the tc

https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Read%20IMERG%20Data%20Using%20Python

inputs: csv with filepaths to rainfall and centre lat lon points

outputs: netcdf file for each storm snapshot, metadata saved within filename
saved in /user/work/al18709/tropical_cyclones/mswep/
"""
print('running')
import glob
import pandas as pd
from netCDF4 import Dataset
import datetime
import numpy as np
import xarray as xr
import h5py
from multiprocessing import Pool
import os
import sys



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
	basin = x.loc['basin']
	# centre_lat = x.loc['centre_lat']
	# centre_lon = x.loc['centre_lon']
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
				data = d.variables['precipitationCal'][0,lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
				lat = lat[lat_lower_bound:lat_upper_bound]
				lon = lon[lon_lower_bound:lon_upper_bound]
				d.close()



	# data = d['Grid'].variables['precipitationCal'][0,lon_lower_bound:lon_upper_bound,lat_lower_bound:lat_upper_bound]
	# lat = lat[lat_lower_bound:lat_upper_bound]
	# lon = lon[lon_lower_bound:lon_upper_bound]
	

	# precip = np.transpose(precip)
	da = xr.DataArray(data, 
					  dims=("x", "y"), 
					  coords={"x": lon, "y": lat},
					  attrs=dict(description="Total Precipitation",units="mm"),
					  name = 'precipitation')

	# da.to_netcdf('/user/work/al18709/tropical_cyclones/imerg' + str(name) + '_' + str(sid) + '_hour-' + str(time) + '_idx-' + str(i) + '_cat-' + str(int(cat))+ '.nc')
	# da.to_netcdf('/user/work/al18709/tropical_cyclones/imerg/' + str(name) + '_' + str(sid) + '_hour-' + str(time) + '_idx-' + str(i) + '_cat-' + str(int(cat)) + '_basin-' + str(basin) + '.nc')
	# da.to_netcdf('/user/work/al18709/tropical_cyclones/imerg/' + str(name) + '_' + str(sid) + '_hour-' + str(time) + '_idx-' + str(i) + '_cat-' + str(int(cat)) + '_basin-' + str(basin) + '_centrelat-' + str(centre_lat) + '_centrelon-' + str(centre_lon) '.nc')
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
		elif 'precipitation' not in Dataset(filepath, 'r').variables:
			with open('logs/error_filepaths.txt', 'w') as f:
					f.write(filepath + ' precipitation variable not included')
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
				# print(da)
				if cat not in [1,2,3,4,5]:
					cat = 0
				
				da.to_netcdf('/user/work/al18709/tropical_cyclones/mswep/' + str(name) + '_' + str(sid) + '_hour-' + str(time) + '_idx-' + str(i) + '_cat-' + str(int(cat)) + '_basin-' + str(basin) + '_centrelat-' + str(centre_lat) + '_centrelon-' + str(centre_lon) + '.nc')
				# da.to_netcdf('/user/work/al18709/tropical_cyclones/mswep_extend/' + str(name) + '_' + str(sid) + '_hour-' + str(time) + '_idx-' + str(i) + '_cat-' + str(int(cat)) + '_basin-' + str(basin) + '_centrelat-' + str(centre_lat) + '_centrelon-' + str(centre_lon) + '.nc')
				# da.to_netcdf('/user/work/al18709/tropical_cyclones/era5_10/' + str(name) + '_' + str(sid) + '_hour-' + str(time) + '_idx-' + str(i) + '_cat-' + str(int(cat)) + '_basin-' + str(basin) + '_centrelat-' + str(centre_lat) + '_centrelon-' + str(centre_lon) + '.nc')
				print('%s saved!' % filepath)
				# TODO: flip lats
	else:
		print('%s not saved!' % filepath)

def process_apply_topography(x):
	# define variables
	filename = x.loc['filepath_var']

	filepath = glob.glob(filename)[0]
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

	if centre_lon > 175: 
		print('goes over centre')
		diff = lon_upper_bound - lon_lower_bound
		second_upper_bound = 100 - diff

		data1 = d.variables['z'][lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
		data2 = d.variables['z'][lat_lower_bound:lat_upper_bound,0:second_upper_bound]
		lat = lat[lat_lower_bound:lat_upper_bound]
		lon1 = lon[lon_lower_bound:lon_upper_bound]
		lon2 = lon[0:second_upper_bound]
		data = np.concatenate((data1,data2),axis=1)
		lon = np.concatenate((lon1,lon2))
	elif centre_lon < -175:
		diff = lon_upper_bound - lon_lower_bound
		second_upper_bound = 100 - diff
		data1 = d.variables['z'][lat_lower_bound:lat_upper_bound,-second_upper_bound:-1]
		data2 = d.variables['z'][lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
		lat = lat[lat_lower_bound:lat_upper_bound]
		lon1 = lon[-second_upper_bound:-1]
		lon2 = lon[lon_lower_bound:lon_upper_bound]
		
		data = np.concatenate((data1,data2),axis=1)
		lon = np.concatenate((lon1,lon2))
	else:
		data = d.variables['z'][lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
		lat = lat[lat_lower_bound:lat_upper_bound]
		lon = lon[lon_lower_bound:lon_upper_bound]

	d.close()
	# print(len(lon))
	# print(len(lat))
	if (len(lon) != 100) or (len(lat) != 100): # TODO: figure out why this happens
		print('dimensions do not match')
		lat,lon,data = fix_dimensions(lat,lon,data)

	# else:
	# precip = np.transpose(precip)

	da = xr.DataArray(data, 
					dims=("x", "y"), 
					coords={"x": lon, "y": lat},
					attrs=dict(description="elevation",units="m"),
					name = 'elevation')
	# print(da)
	if cat not in [1,2,3,4,5]:
		cat = 0
	
	da.to_netcdf('/user/work/al18709/tropical_cyclones/topography/' + str(name) + '_' + str(sid) + '_hour-' + str(time) + '_idx-' + str(i) + '_cat-' + str(int(cat)) + '_basin-' + str(basin) + '_centrelat-' + str(centre_lat) + '_centrelon-' + str(centre_lon) + '.nc')
	# print('%s saved!' % filepath)

def process_apply_era5(x):
	print('applying era5 process...')
	# define variables
	
	result = 1
	filepath = x.loc['filepath_era5']

	# remove filepaths which relate to timestamps we don't have in ERA5
	print('the hour is: ',str(x.loc['hour']))
	if str(x.loc['hour']) not in ['00','3','6','9','12','15','18','21']:
		print(x.loc['hour'])
		result = 0

	if result == 1:
		print(filepath)
		if glob.glob(filepath) == []:
			with open('logs/error_filepaths.txt', 'w') as f:
					f.write(filepath)
					f.write('\n')
		elif 'tp' not in Dataset(filepath, 'r').variables:
			with open('logs/error_filepaths.txt', 'w') as f:
					f.write(filepath + ' precipitation variable not included')
					f.write('\n')

		else:
			filepath = glob.glob(filepath)[0]
			centre_lat = x.loc['lat']
			centre_lon = x.loc['lon']
			if centre_lon < 0:
				centre_lon = centre_lon + 360
			name = x.loc['name']
			cat = x.loc['sshs']
			time = x.loc['hour']
			sid = x.loc['sid']
			basin = x.loc['basin']
			year = x.loc['year']
			month = x.loc['month']
			day = x.loc['day']
			hour = x.loc['hour']		
			tc_time = '%s-%s-%sT%s:00:00' % (year,month,day,hour)
			# tc_time = (datetime.datetime(year,month,day,hour) - datetime.datetime(1900,1,1,0)).seconds/3600 # datetime gregorian?
			i = x[0]

			# open file
			d = xr.open_dataset(filepath)
			lat = d.latitude.values
			lon = d.longitude.values

			# check if variable exists
			
			# era5 don't have the same lat and lons as mswep
			# clip to location
			# lat_lower_bound = (np.abs(lat-centre_lat-5.)).argmin()
			# lat_upper_bound = (np.abs(lat-centre_lat+5.)).argmin()
			lat_lower_bound = (np.abs(lat-centre_lat+5.)).argmin()
			lat_upper_bound = (np.abs(lat-centre_lat-5.)).argmin()
			lon_lower_bound = (np.abs(lon-centre_lon+5.)).argmin()
			lon_upper_bound = (np.abs(lon-centre_lon-5.)).argmin()

			# work on edge cases

			if centre_lon > 355: 
				print('goes over centre')
				diff = lon_upper_bound - lon_lower_bound
				second_upper_bound = 100 - diff
				data1 = d.sel(time=tc_time).variables['tp'][lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
				data2 = d.sel(time=tc_time).variables['tp'][lat_lower_bound:lat_upper_bound,0:second_upper_bound]
				lat = lat[lat_lower_bound:lat_upper_bound]
				lon1 = lon[lon_lower_bound:lon_upper_bound]
				lon2 = lon[0:second_upper_bound]
				data = np.concatenate((data1,data2),axis=1)
				lon = np.concatenate((lon1,lon2))
			elif centre_lon < 5:
				print('goes over centre the other way')
				diff = lon_upper_bound - lon_lower_bound
				second_upper_bound = 100 - diff

				data1 = d.sel(time=tc_time).variables['tp'][lat_lower_bound:lat_upper_bound,-second_upper_bound:-1]
				data2 = d.sel(time=tc_time).variables['tp'][lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
				lat = lat[lat_lower_bound:lat_upper_bound]
				lon1 = lon[-second_upper_bound:-1]
				lon2 = lon[lon_lower_bound:lon_upper_bound]
				
				data = np.concatenate((data1,data2),axis=1)
				lon = np.concatenate((lon1,lon2))
			else:
				print('does not go over centre')
				print(centre_lat,centre_lon,tc_time,lat_lower_bound,lat_upper_bound,lon_lower_bound,lon_upper_bound)
				print(' ')
				# print(d.time.values)
				# print('tc_time',tc_time)

				data = d.sel(time=tc_time).variables['tp'][lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
				lat = lat[lat_lower_bound:lat_upper_bound]
				lon = lon[lon_lower_bound:lon_upper_bound]
			
			print(len(lon))
			print(len(lat))
			if (len(lon) != 40) or (len(lat) != 40): # TODO: figure out why this happens
				print('dimensions do not match')
			else:
				# precip = np.transpose(precip)
				print(data.shape)
				da = xr.DataArray(data, 
								dims=("x", "y"), 
								coords={"x": lon, "y": lat},
								attrs=dict(description="Total Precipitation",units="mm"),
								name = 'precipitation')
				# print(da)
				if cat not in [1,2,3,4,5]:
					cat = 0
				if centre_lon > 180:
					centre_lon = centre_lon - 360
				print('saving new era5 file...')
				da.to_netcdf('/user/work/al18709/tropical_cyclones/era5_10/' + str(name) + '_' + str(sid) + '_hour-' + str(time) + '_idx-' + str(i) + '_cat-' + str(int(cat)) + '_basin-' + str(basin) + '_centrelat-' + str(centre_lat) + '_centrelon-' + str(centre_lon) + '.nc')
				print('%s saved!' % filepath)
	else:
		print('%s not saved!' % filepath)

def process_apply_variable(x):
	print('applying variable process...')
	# define variables
	
	result = 1
	filepath = x.loc['filepath_var']

	# remove filepaths which relate to timestamps we don't have in ERA5
	print('the hour is: ',str(x.loc['hour']))
	if str(x.loc['hour']) not in ['0','3','6','9','12','15','18','21']:
		print(x.loc['hour'])
		result = 0

	if result == 1:
		print(filepath)
		if glob.glob(filepath) == []:
			with open('logs/error_filepaths.txt', 'w') as f:
					f.write(filepath)
					f.write('\n')
		# elif 'tp' not in Dataset(filepath, 'r').variables:
		# 	with open('logs/error_filepaths.txt', 'w') as f:
		# 			f.write(filepath + ' precipitation variable not included')
		# 			f.write('\n')

		else:
			filepath = glob.glob(filepath)[0]
			print('filepath is: ', filepath)
			print('variables are:',Dataset(filepath, 'r').variables.keys())
			variable = list(Dataset(filepath, 'r').variables.keys())[-1]
			print('variable is: ', variable)
			centre_lat = x.loc['lat']
			centre_lon = x.loc['lon']
			if centre_lon < 0:
				centre_lon = centre_lon + 360
			name = x.loc['name']
			cat = x.loc['sshs']
			time = x.loc['hour']
			sid = x.loc['sid']
			basin = x.loc['basin']
			year = x.loc['year']
			month = x.loc['month']
			day = x.loc['day']
			hour = x.loc['hour']		
			tc_time = '%s-%s-%sT%s:00:00' % (year,month,day,hour)
			# tc_time = (datetime.datetime(year,month,day,hour) - datetime.datetime(1900,1,1,0)).seconds/3600 # datetime gregorian?
			i = x[0]

			# open file
			d = xr.open_dataset(filepath)
			lat = d.lat.values
			lon = d.lon.values

			# check if variable exists
			
			# era5 don't have the same lat and lons as mswep
			# clip to location
			lat_lower_bound = (np.abs(lat-centre_lat-5.)).argmin()
			lat_upper_bound = (np.abs(lat-centre_lat+5.)).argmin()
			# lat_lower_bound = (np.abs(lat-centre_lat+5.)).argmin()
			# lat_upper_bound = (np.abs(lat-centre_lat-5.)).argmin()
			lon_lower_bound = (np.abs(lon-centre_lon+5.)).argmin()
			lon_upper_bound = (np.abs(lon-centre_lon-5.)).argmin()
			# lon_lower_bound = (np.abs(lon-centre_lon-5.)).argmin()
			# lon_upper_bound = (np.abs(lon-centre_lon+5.)).argmin()

			# work on edge cases

			if centre_lon > 355: 
				print('goes over centre')
				diff = lon_upper_bound - lon_lower_bound
				second_upper_bound = 100 - diff
				data1 = d.sel(time=tc_time).variables[variable][lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
				data2 = d.sel(time=tc_time).variables[variable][lat_lower_bound:lat_upper_bound,0:second_upper_bound]
				lat = lat[lat_lower_bound:lat_upper_bound]
				lon1 = lon[lon_lower_bound:lon_upper_bound]
				lon2 = lon[0:second_upper_bound]
				data = np.concatenate((data1,data2),axis=1)
				lon = np.concatenate((lon1,lon2))
			elif centre_lon < 5:
				print('goes over centre the other way')
				diff = lon_upper_bound - lon_lower_bound
				second_upper_bound = 100 - diff

				data1 = d.sel(time=tc_time).variables[variable][lat_lower_bound:lat_upper_bound,-second_upper_bound:-1]
				data2 = d.sel(time=tc_time).variables[variable][lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
				lat = lat[lat_lower_bound:lat_upper_bound]
				lon1 = lon[-second_upper_bound:-1]
				lon2 = lon[lon_lower_bound:lon_upper_bound]
				
				data = np.concatenate((data1,data2),axis=1)
				lon = np.concatenate((lon1,lon2))
			else:
				print('does not go over centre')
				print(centre_lat,centre_lon,tc_time,lat_lower_bound,lat_upper_bound,lon_lower_bound,lon_upper_bound)
				print(' ')
				# print(d.time.values)
				print('tc_time',tc_time)

				data = d.sel(time=tc_time).variables[variable][lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
				lat = lat[lat_lower_bound:lat_upper_bound]
				lon = lon[lon_lower_bound:lon_upper_bound]

				if data.shape != (10,10): #TODO: check this doesn't mess with orientation later
					print('wrong data shape')
					data = d.sel(time=tc_time)
					# print('data shape',data.shape)
					print('data time is:',data.time)
					data = data.variables[variable][0,lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
					
			print('lon length: ',len(lon))
			print('lat length: ',len(lat))
			# TODO: for some reason lat and lon don't match up with data :/
			if (len(lon) != 10) or (len(lat) != 10): # TODO: figure out why this happens
				print('dimensions do not match')
			else:
				# precip = np.transpose(precip)
				print('data shape should be (10,10): ',data.shape)
				print(lon)
				print(lat)
				print('lon length: ',len(lon))
				print('lat length: ',len(lat))
				da = xr.DataArray(data, 
								dims=("x", "y"), 
								coords={"x": lon, "y": lat},
								attrs=dict(description="data",units="units"),
								name = variable)
				# print(da)
				if cat not in [1,2,3,4,5]:
					cat = 0
				if centre_lon > 180:
					centre_lon = centre_lon - 360
				print('saving new era5 file...')
				da.to_netcdf('/user/work/al18709/tropical_cyclones/var/' + str(name) + '_' + str(sid) + '_hour-' + str(time) + '_idx-' + str(i) + '_cat-' + str(int(cat)) + '_basin-' + str(basin) + '_centrelat-' + str(centre_lat) + '_centrelon-' + str(centre_lon) + '.nc')
				print('%s saved!' % filepath)
	else:
		print('%s not saved!' % filepath)

def fix_dimensions(lat,lon,data):
	print('fixing dimensions...')
	if len(lon) == 99:
		print('lon too long')
		lon = np.append(lon,lon[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,np.array([data[:,-1]]).T),axis=1)
	if len(lat) == 99:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lon) == 101:
		lon=lon[:-1]
		data = data[:,:-1]
	if len(lat) == 101:
		lat = lat[:-1]
		data = data[:-1,:]
		
	print(lon.shape,lat.shape,data.shape)
	return lat,lon,data

def process(df):
	print('doing process...')
	res = df.apply(process_apply,axis=1)
	return res

def process_mswep(df):
	print('doing process...')
	res = df.apply(process_apply_mswep,axis=1)
	return res

def process_era5(df):
	print('doing process...')
	res = df.apply(process_apply_era5,axis=1)
	return res

def process_variable(df):
	print('doing process...')
	res = df.apply(process_apply_variable,axis=1)
	return res

def process_topography(df):
	print('doing process...')
	res = df.apply(process_apply_topography,axis=1)
	return res

if __name__ == '__main__':
	# dataset can wither be mswep, imerg, era5 or variable
	dataset = sys.argv[1]
	print('dataset: ', dataset)
	# if '/' in variable:
	# 	var = variable.split('/')[0]
	# else: 
	# 	var = variable

	# dataset = 'mswep' # or imerg
	# dataset = 'imerg'
	# dataset = 'era5'
	df = pd.read_csv('/user/work/al18709/ibtracks/tc_files.csv')
	# df = pd.read_csv('/user/work/al18709/ibtracks/tc_files_all.csv')
	df_split = np.array_split(df, 64)
	p = Pool(processes=64)
	# df_split = np.array_split(df, 1)
	# p = Pool(processes=1)
	print('opened csv!')
	print('pooling processes...')

	# run different functions depending on which dataset we use
	if dataset == 'imerg':
		pool_results = p.map(process, df_split)
	elif dataset == 'mswep':
		pool_results = p.map(process_mswep, df_split)
	elif dataset == 'era5':
		pool_results = p.map(process_era5, df_split)
	elif dataset == 't':
		pool_results = p.map(process_topography, df_split)
	else:
		# remove old files to save space
		for file in os.listdir('/user/work/al18709/tropical_cyclones/var'):
			os.remove('/user/work/al18709/tropical_cyclones/var/'+ file)
		pool_results = p.map(process_variable, df_split)
	# data = pd.concat(p.map(process, df_split))
	print('results pooled')
	p.close()

	p.join()

	# TODO fix error in line 20 where somehow filepath doesn't exist?