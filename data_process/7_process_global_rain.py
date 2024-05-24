# script to generate low res data of final advanced imerge dataset

import glob
import xarray as xr
import numpy as np
import xesmf as xe
import pandas as pd
import sys
from itertools import groupby
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
from netCDF4 import Dataset


def flip(tc):
	tc_flipped = np.flip(tc,axis=0)
	return tc_flipped

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
	if len(lat) == 103:
		print('lat too long 103')
		lat = lat[:-3]
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = data[:-3,:]
		print(data.shape)
	if len(lon) == 98:
		print('lon too long')
		lon = np.append(lon,lon[-1])
		lon = np.append(lon,lon[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,np.array([data[:,-1]]).T),axis=1)
		data = np.concatenate((data,np.array([data[:,-1]]).T),axis=1)
	if len(lon) == 94:
		print('lon too long')
		lon = np.append(lon,lon[-1])
		lon = np.append(lon,lon[-1])
		lon = np.append(lon,lon[-1])
		lon = np.append(lon,lon[-1])
		lon = np.append(lon,lon[-1])
		lon = np.append(lon,lon[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,np.array([data[:,-1]]).T),axis=1)
		data = np.concatenate((data,np.array([data[:,-1]]).T),axis=1)
		data = np.concatenate((data,np.array([data[:,-1]]).T),axis=1)
		data = np.concatenate((data,np.array([data[:,-1]]).T),axis=1)
		data = np.concatenate((data,np.array([data[:,-1]]).T),axis=1)
		data = np.concatenate((data,np.array([data[:,-1]]).T),axis=1)
	if len(lat) == 98:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 97:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 94:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 96:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	
	if len(lat) == 95:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		
	if len(lat) == 92:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 89:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 90:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
	
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 91:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
	
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 93:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
	
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 89:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 87:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 86:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 88:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 85:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lat) == 84:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lon) == 101:
		lon=lon[:-1]
		data = data[:,:-1]
	if len(lat) == 101:
		lat = lat[:-1]
		data = data[:-1,:]
		
	print(lon.shape,lat.shape,data.shape)
	return lat,lon,data

def find_and_flip(centre_lat,centre_lon,data):
	print(centre_lat)
	sh_indices = (centre_lat < 0)
	nh_indices = (centre_lat > 0)
	# print(nh_indices)

	# flip the nh tcs so they are rotating anticlockwise (in the raw data array)
	# in mswep they can be plotted correctly with the mswep lats and lons, but the raw data shows them flipped as mswep has descending latitudes
	if nh_indices:
		topography_flipped = flip(data)

	else:
		topography_flipped = data
	
	return topography_flipped

def generate_topography(tc_topography,centre_lat,centre_lon,i,lat,lon,d):

	# open file
	# filepath = '/user/work/al18709/topography/topography_10km_nn.nc'
	# d = Dataset(filepath, 'r')
	# lat = d.variables['lat'][:] #lat
	# lon = d.variables['lon'][:] #lon
		
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
	if (len(lon) != 100) or (len(lat) != 100): # TODO: figure out why this happens
		print('dimensions do not match')
		lat,lon,data = fix_dimensions(lat,lon,data)
	
	data = find_and_flip(centre_lat,centre_lon,data)

	# print('centre_lat',centre_lat)
	# print('centre_lon',centre_lon)
	# print('data shape', data.shape)
	tc_topography[i,:,:] = data
	return tc_topography


dir = '/user/home/al18709/work/CMIP6/MSWEP/storm_rain'
rain_fname = 'storm_rain_tcs_and_ts.npy'
sids_fname = 'storm_sid_tcs_and_ts.csv'
lats_fname = 'storm_lats_tcs_and_ts.npy'
lons_fname = 'storm_lons_tcs_and_ts.npy'

grid_in = xr.Dataset({'longitude': np.linspace(0, 100, 100),
				'latitude': np.linspace(-50, 50, 100)
				})
grid_out = xr.Dataset({'longitude': np.linspace(0, 100, 10),
					'latitude': np.linspace(-50, 50, 10)
				})

# regrid with conservative interpolation so means are conserved spatially
regridder = xe.Regridder(grid_in, grid_out, 'conservative')
print('regrid set up!')

tc_y = np.load(f'{dir}/{rain_fname}')
lats = np.load(f'{dir}/{lats_fname}')
lons = np.load(f'{dir}/{lons_fname}')
meta = pd.read_csv(f'{dir}/storm_sid_tcs_and_ts.csv')
lats = meta['centre_lat']
lons = meta['centre_lon']

nsamples,_,_ = tc_y.shape
tc_y_regrid = np.zeros((nsamples,10,10))
print(tc_y_regrid.shape)

# generate topography
fp_regrid = '/user/home/al18709/work/topography/topography_10km_nn.nc'
tc_topography  = xr.load_dataset(fp_regrid,engine="netcdf4")
tc_topography = np.zeros((tc_y_regrid.shape[0],100,100))

filepath = '/user/work/al18709/topography/topography_10km_nn.nc'
d = Dataset(filepath, 'r')
lat = d.variables['lat'][:] #lat
lon = d.variables['lon'][:]

# create training dataset
# variables = ['precip','mslp','q-925','u-200','u-850','v-200','v-850']
# nstorms,_,_,_ = X.shape
# precip = np.mean(X[:,:,:,0],axis=(1,2))
# mslp = np.mean(X[:,:,:,1],axis=(1,2))
# q925 = np.mean(X[:,:,:,2],axis=(1,2))
# u200 = np.mean(X[:,:,:,3],axis=(1,2))
# u850 = np.mean(X[:,:,:,4],axis=(1,2))
# v200 = np.mean(X[:,:,:,5],axis=(1,2))
# v850 = np.mean(X[:,:,:,6],axis=(1,2))
# shear = np.sqrt(np.abs(np.square(u850-u200) - np.square(v850-v200)))

for i in range(nsamples):
	print(i,end='\n')
	
	centre_lat = lats[i]
	centre_lon = lons[i]
	print(centre_lat)
	print(centre_lon)
	print(d)
	data = regridder(tc_y[i,:,:])
	# tc_y_regrid[i,:,:] = find_and_flip(centre_lat,centre_lon,data)
	if centre_lon > 180:
		centre_lon = centre_lon - 360
	tc_topography = generate_topography(tc_topography,centre_lat,centre_lon,i,lat,lon,d)

print(tc_y_regrid.shape)
print(tc_y.shape)
# np.save(f'{dir}/storm_rain_tcs_and_ts_low_resolution.npy',tc_y_regrid)
np.save(f'{dir}/storm_rain_tcs_and_ts_topography.npy',tc_topography)


