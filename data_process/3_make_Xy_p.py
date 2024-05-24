"""
	4. Create X, y and test and training sets
	
This script formats the individual storm netcdf files into an X and y numpy array. 
Each tc will have it's own X and y npy file

	X shape : (n_timesteps, 10, 10)
	y shape : (n_timesteps,100,100)

# saved in : /user/work/al18709/tc_data/
/user/work/al18709/tc_Xy/

"""

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
sns.set_style("white")
import re

def plot_tc(array,SH):
	lat2d, lon2d = np.linspace(-50, 50, 100),np.linspace(0, 100, 100)
	fig, ax = plt.subplots()
	cmap ='seismic_r'
	c = ax.pcolor(lon2d,lat2d,array,vmin=-60,vmax=60,cmap = cmap,)
	cbar = plt.colorbar(c, shrink=0.54)
	cbar.outline.set_linewidth(0.5)
	cbar.ax.tick_params(labelsize=6,width=0.5)
	plt.savefig('figs/tc_plot_%s.png' % SH,dpi=600,bbox_inches='tight')
	plt.clf()

def save_Xy(grouped_tcs):
	"""
	save X and y npy file for each tc
	"""

	# initial set up
	if dataset == 'mswep':
		# regex = r"/user/work/al18709/tropical_cyclones/mswep/.+?_(.+?)_.*?(..).nc"
		regex = r"/user/work/al18709/tropical_cyclones/mswep/.+?_(.+?)_.*?_centrelat-(.+?)_centrelon-(.+?).nc"
	elif dataset == 'imerg':
		regex = r"/user/work/al18709/tropical_cyclones/imerg/.+?_(.+?)_.*?(..).nc"
	elif dataset == 'era5':
		# regex = r"/user/work/al18709/tropical_cyclones/mswep/.+?_(.+?)_.*?(..).nc"
		regex = r"/user/work/al18709/tropical_cyclones/era5_10/.+?_(.+?)_.*?_idx-(.+?)_.*?_centrelat-(.+?)_centrelon-(.+?).nc"
	elif dataset == 'mswep_extend':
		regex = r"/user/work/al18709/tropical_cyclones/mswep_extend/.+?_(.+?)_.*?_centrelat-(.+?)_centrelon-(.+?).nc"
	elif dataset == 'var':
		regex = r"/user/work/al18709/tropical_cyclones/var/.+?_(.+?)_.*?_centrelat-(.+?)_centrelon-(.+?).nc"
	elif dataset == 't':
		regex = r"/user/work/al18709/tropical_cyclones/topography/.+?_(.+?)_.*?_centrelat-(.+?)_centrelon-(.+?).nc"
	# regex = r"/user/work/al18709/tropical_cyclones/.+?_(.+?)_.*?.nc"
	resolution = 10
	print('dataset is:', dataset)
	print('resolution is: ',resolution)

	# define grid, this doesn't need to be specific, only needs to be the correct resolution
	if (dataset == 'era5'):
		grid_in = xr.Dataset({'longitude': np.linspace(0, 100, 40),
			'latitude': np.linspace(-50, 50, 40)
			})
	else:
		grid_in = xr.Dataset({'longitude': np.linspace(0, 100, 100),
				'latitude': np.linspace(-50, 50, 100)
				})
	if resolution == 40:
		grid_out = xr.Dataset({'longitude': np.linspace(0, 100, 40),
					'latitude': np.linspace(-50, 50, 40)
				})
	else:
		# output grid has a the same coverage at finer resolution
		grid_out = xr.Dataset({'longitude': np.linspace(0, 100, 10),
					'latitude': np.linspace(-50, 50, 10)
				})
	
	# regrid with conservative interpolation so means are conserved spatially
	regridder = xe.Regridder(grid_in, grid_out, 'conservative')

	# loop through tcs and make an X and y array for each
	for tc in grouped_tcs:
		sid = re.match(regex,tc[0]).groups()[0]	

		n_timesteps = len(tc)
		# if dataset == 'era5':
		# 	tc_X = np.zeros((n_timesteps,40,40))
		# 	meta_lats = []
		# 	meta_lons = []
		# 	meta_sids = []
		# else:
		if resolution == 40:
			tc_X = np.zeros((n_timesteps,40,40))
		else:
			tc_X = np.zeros((n_timesteps,10,10))
		meta_lats = np.zeros((n_timesteps))
		meta_lons = np.zeros((n_timesteps))
		meta_sids = []
		if dataset == 'era5':
			tc_y = np.zeros((n_timesteps,40,40))
		else:
			tc_y = np.zeros((n_timesteps,100,100))

		# loop though tc filepaths
		for i,filepath in enumerate(tc):
			print(i,end='\r')
			print('filepath = ', filepath)
			ds = xr.open_dataset(filepath)
			if dataset == 'era5':
				idx = re.match(regex,tc[i]).groups()[1]
				centre_lat = re.match(regex,tc[i]).groups()[2]
				centre_lon = re.match(regex,tc[i]).groups()[3]
			else:
				centre_lat = re.match(regex,tc[i]).groups()[1]
				centre_lon = re.match(regex,tc[i]).groups()[2]
			
			# array_flipped = np.zeros((1,100,100))

			# if dataset == 'era5':
			# 	mswep_fp = glob.glob('/user/work/al18709/tropical_cyclones/mswep/*_idx-%s_*.nc' % idx)
			# 	if len(mswep_fp) == 0 :
			# 		n = 1
			# 		print('skip',n)
			# 		n = n+1
			# 		continue
			# 	else:
			# 		mswep_fp = mswep_fp[0]

			# 	mswep_array = xr.open_dataset(mswep_fp).precipitation.values
			
			if dataset == 'var':
				var = list(ds.variables)[-1]
				array = ds[var].values
			else:
				array = ds.precipitation.values

			if (dataset == 'era5'):
				# print('the limit is 40')
				limit = 40
			elif dataset == 'var':
				# we dont' need y for variables
				limit = 10
			else:
				# print('the limit is 100')
				limit = 100
			# print('array shape: ',array.shape)
			if array.shape != (limit,limit):
				print('skipping')
				continue


			# TODO: going to end up with some zero values here - fix that
			# if dataset == 'era5':
			# 	tc_X[i,:,:] = array
			# 	tc_y[i,:,:] = mswep_array
			# 	meta_lats.append(centre_lat)
			# 	meta_lons.append(centre_lon)
			# 	meta_sids.append(str(sid))
			# else:
			if dataset == 'var':
				# variables already in low resolution
				tc_X[i,:,:] = array
			else:
				tc_X[i,:,:] = regridder(array)
				tc_y[i,:,:] = array
			meta_lats[i] = centre_lat
			meta_lons[i] = centre_lon
			meta_sids.append(str(sid))

		# print('meta_sids',len(meta_sids))
		# print('meta_lats',len(meta_lats))
		# print('meta_lons',len(meta_lons))
		# meta = np.dtype(float, metadata={"dataset": dataset,"sid":meta_sids,"centre_lat": meta_lats,"centre_lons": meta_lons})
		meta = pd.DataFrame({'sid' : meta_sids,'centre_lat' : meta_lats,'centre_lon' : meta_lons})
		if dataset == 'mswep':
			print('saving to mswep...')
			if resolution == 40:
				path = '/user/work/al18709/tc_Xy_40'
			else:
				path = '/user/work/al18709/tc_Xy'
		elif dataset == 'era5':
			print('saving to era5...')
			path = '/user/work/al18709/tc_Xy_era5_10'
		elif dataset == 'mswep_extend':
			print('saving to mswep extend...')
			path = '/user/work/al18709/tc_Xy_extend'
		elif dataset == 'var':
			print('saving to var...')
			path = '/user/work/al18709/tc_Xy_var'

		print(tc_X.shape)
		if dataset == 'var':
			np.save('%s/X_%s.npy' % (path,sid),tc_X)
			meta.to_csv('%s/meta_%s.csv' % (path,sid))
		else:
			np.save('%s/X_%s.npy' % (path,sid),tc_X)
			np.save('%s/y_%s.npy' % (path,sid),tc_y)
			meta.to_csv('%s/meta_%s.csv' % (path,sid))
		print('saved!')

def save_Xy_era5(grouped_tcs):
	"""
	save X and y npy file for each tc
	"""

	# initial set up
	resolution = 40 # or 40
	regex = r"/user/work/al18709/tropical_cyclones/era5/.+?_(.+?)_.*?_idx-(.+?)_.*?_centrelat-(.+?)_centrelon-(.+?).nc"

	# loop through tcs and make an X and y array for each
	for tc in grouped_tcs:
		number = 1
		
		print('storm: ',number,end='\r')
		number = number + 1
		sid = re.match(regex,tc[0]).groups()[0]	
		n_timesteps = len(tc)
		# tc_X = np.zeros((n_timesteps,40,40))
		tc_X = np.zeros((n_timesteps,resolution,resolution))
		meta_lats = np.zeros((n_timesteps))
		meta_lons = np.zeros((n_timesteps))
		meta_sids = np.zeros((n_timesteps),dtype=str)
		remove_i = []

		tc_y = np.zeros((n_timesteps,100,100))

		# loop though tc filepaths
		for i,filepath in enumerate(tc):
			# print(i,end='\r')
			
			ds = xr.open_dataset(filepath)
			idx = re.match(regex,tc[i]).groups()[1]
			centre_lat = re.match(regex,tc[i]).groups()[2]
			centre_lon = re.match(regex,tc[i]).groups()[3]
			
			mswep_fp = glob.glob('/user/work/al18709/tropical_cyclones/mswep/*_idx-%s_*.nc' % idx)
			if len(mswep_fp) == 0:
				remove_i.append(i)
				continue
			else:
				mswep_fp = mswep_fp[0]

			mswep_array = xr.open_dataset(mswep_fp).precipitation.values
			
			array = ds.precipitation.values
			# if array.shape != (40,40):
			if array.shape != (resolution,resolution):
				remove_i.append(i)
				continue

			# assign arrays
			tc_X[i,:,:] = array
			tc_y[i,:,:] = mswep_array
			meta_lats[i] = centre_lat
			meta_lons[i] = centre_lon
			meta_sids[i] = str(sid)

		for index in remove_i:
			np.delete(meta_lats,index)
			np.delete(meta_lons,index)
			np.delete(meta_sids,index)
			np.delete(tc_X,index)
			np.delete(tc_y,index)
		meta = pd.DataFrame({'sid' : meta_sids,'centre_lat' : meta_lats,'centre_lon' : meta_lons})
		path = '/user/work/al18709/tc_Xy_era5_40'
		np.save('%s/X_%s.npy' % (path,sid),tc_X)
		np.save('%s/y_%s.npy' % (path,sid),tc_y)
		meta.to_csv('%s/meta_%s.csv' % (path,sid))
		# print('saved!')

def save_Xy_topography(grouped_tcs):
	"""
	save X and y npy file for each tc
	"""

	# initial set up
	regex = r"/user/work/al18709/tropical_cyclones/topography/.+?_(.+?)_.*?_centrelat-(.+?)_centrelon-(.+?).nc"
	resolution = 10
	print('dataset is:', dataset)
	print('resolution is: ',resolution)

	# loop through tcs and make an X and y array for each
	for tc in grouped_tcs:
		sid = re.match(regex,tc[0]).groups()[0]	
		n_timesteps = len(tc)
		tc_X = np.zeros((n_timesteps,10,10))
		tc_y = np.zeros((n_timesteps,100,100))
		meta_lats = np.zeros((n_timesteps))
		meta_lons = np.zeros((n_timesteps))
		meta_sids = []

		# loop though tc filepaths
		for i,filepath in enumerate(tc):
			# print(i,end='\r')
			# print('filepath = ', filepath)
			ds = xr.open_dataset(filepath)
			centre_lat = re.match(regex,tc[i]).groups()[1]
			centre_lon = re.match(regex,tc[i]).groups()[2]
			print('ds variables',ds.variables)
			
			array = ds.elevation.values
			limit = 100
			print('array shape: ',array.shape)
			if array.shape != (limit,limit):
				print('skipping')
				continue

			tc_y[i,:,:] = array
			meta_lats[i] = centre_lat
			meta_lons[i] = centre_lon
			meta_sids.append(str(sid))

		meta = pd.DataFrame({'sid' : meta_sids,'centre_lat' : meta_lats,'centre_lon' : meta_lons})
		path = '/user/work/al18709/tc_Xy_topography'
		np.save('%s/y_%s.npy' % (path,sid),tc_y)
		meta.to_csv('%s/meta_%s.csv' % (path,sid))
		print('saved!')



def process(filepaths):
	print('doing process...')
	res = save_Xy(filepaths)
	return res

def process_era5(filepaths):
	print('doing process...')
	res = save_Xy_era5(filepaths)
	return res

def process_topography(filepaths):
	print('doing topography process...')
	res = save_Xy_topography(filepaths)
	return res

if __name__ == '__main__':

	# set up
	n_processes = 128
	variable = sys.argv[1]
	print('variable: ', variable)
	if '/' in variable:
		dataset = 'var'
	elif variable in ['mswep','era5','t']:
		dataset = variable
	else: 
		dataset = 'var'

	resolution = 10
	print(dataset)
	
	if dataset == 'era5':
		tc_dir = '/user/work/al18709/tropical_cyclones/%s_10/*.nc' % dataset
	elif dataset == 't':
		tc_dir = '/user/work/al18709/tropical_cyclones/topography/*.nc'
	else:
		tc_dir = '/user/work/al18709/tropical_cyclones/%s/*.nc' % dataset
	filepaths = glob.glob(tc_dir)
	print('number of tropical cyclones',len(filepaths))

	# group by tc sid number
	print('grouping...')
	if dataset == 'mswep':
		regex = r"/user/work/al18709/tropical_cyclones/mswep/.+?_(.+?)_.*?.nc"
	elif dataset == 'imerg':
		regex = r"/user/work/al18709/tropical_cyclones/imerg/.+?_(.+?)_.*?.nc"
	elif dataset == 'era5':
		regex = r"/user/work/al18709/tropical_cyclones/era5_10/.+?_(.+?)_.*?.nc"
	elif dataset == 'mswep_extend':
		regex = r"/user/work/al18709/tropical_cyclones/mswep_extend/.+?_(.+?)_.*?.nc"
	elif dataset == 'var':
		regex = r"/user/work/al18709/tropical_cyclones/var/.+?_(.+?)_.*?.nc"
	elif dataset == 't':
		regex = r"/user/work/al18709/tropical_cyclones/topography/.+?_(.+?)_.*?.nc"
	keyf = lambda text: (re.findall(regex, text)+ [text])[0]
	grouped_tcs = [list(items) for gr, items in groupby(sorted(filepaths), key=keyf)]
	print('grouped!')

	# split into 64 list of tc filepaths
	print('splitting')
	tc_split = np.array(np.array_split(grouped_tcs, n_processes))
	print('split!')

	# start processes
	print('pooling processes...')
	p = Pool(processes=16)
	print('pooling processes...')
	if dataset == 'mswep':
		pool_results = p.map(process, tc_split)
	elif (dataset == 'era5') and (resolution == 10):
		print('processing era5 at low resolution!')
		pool_results = p.map(process, tc_split)	
	elif (dataset == 'era5') and (resolution == 40):
		pool_results = p.map(process_era5, tc_split)
	elif dataset == 'mswep_extend':
		pool_results = p.map(process, tc_split)
	elif dataset == 'var':
		pool_results = p.map(process, tc_split)
	elif dataset == 't':
		pool_results = p.map(process_topography, tc_split)
	print('results pooled')
	p.close()
	p.join()