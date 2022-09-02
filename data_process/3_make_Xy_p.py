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
		regex = r"/user/work/al18709/tropical_cyclones/era5/.+?_(.+?)_.*?_idx-(.+?)_.*?_centrelat-(.+?)_centrelon-(.+?).nc"
	elif dataset == 'mswep_extend':
		regex = r"/user/work/al18709/tropical_cyclones/mswep_extend/.+?_(.+?)_.*?_centrelat-(.+?)_centrelon-(.+?).nc"
	
	# regex = r"/user/work/al18709/tropical_cyclones/.+?_(.+?)_.*?.nc"
	resolution = 10

	# define grid, this doesn't need to be specific, only needs to be the correct resolution
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
		if dataset == 'era5':
			tc_X = np.zeros((n_timesteps,40,40))
			meta_lats = []
			meta_lons = []
			meta_sids = []
		else:
			if resolution == 40:
				tc_X = np.zeros((n_timesteps,40,40))
			else:
				tc_X = np.zeros((n_timesteps,10,10))
			meta_lats = np.zeros((n_timesteps))
			meta_lons = np.zeros((n_timesteps))
			meta_sids = []
		tc_y = np.zeros((n_timesteps,100,100))

		# loop though tc filepaths
		for i,filepath in enumerate(tc):
			print(i,end='\r')
			
			ds = xr.open_dataset(filepath)
			if dataset == 'era5':
				idx = re.match(regex,tc[i]).groups()[1]
				centre_lat = re.match(regex,tc[i]).groups()[2]
				centre_lon = re.match(regex,tc[i]).groups()[3]
			else:
				centre_lat = re.match(regex,tc[i]).groups()[1]
				centre_lon = re.match(regex,tc[i]).groups()[2]
			
			# array_flipped = np.zeros((1,100,100))

			if dataset == 'era5':
				mswep_fp = glob.glob('/user/work/al18709/tropical_cyclones/mswep/*_idx-%s_*.nc' % idx)
				if len(mswep_fp) == 0 :
					# print('skip')
					continue
				else:
					mswep_fp = mswep_fp[0]

				mswep_array = xr.open_dataset(mswep_fp).precipitation.values
			
			array = ds.precipitation.values

			if dataset == 'era5':
				limit = 40
			else:
				limit = 100
			if array.shape != (limit,limit):
				continue


			# TODO: going to end up with some zero values here - fix that
			if dataset == 'era5':
				tc_X[i,:,:] = array
				tc_y[i,:,:] = mswep_array
				meta_lats.append(centre_lat)
				meta_lons.append(centre_lon)
				meta_sids.append(str(sid))
			else:
				tc_X[i,:,:] = regridder(array)
				tc_y[i,:,:] = array
				meta_lats[i] = centre_lat
				meta_lons[i] = centre_lon
				meta_sids.append(str(sid))


		# meta = np.dtype(float, metadata={"dataset": dataset,"sid":meta_sids,"centre_lat": meta_lats,"centre_lons": meta_lons})
		meta = pd.DataFrame({'sid' : meta_sids,'centre_lat' : meta_lats,'centre_lon' : meta_lons})
		if dataset == 'mswep':
			if resolution == 40:
				path = '/user/work/al18709/tc_Xy_40'
			else:
				path = '/user/work/al18709/tc_Xy'
		elif dataset == 'era5':
			path = '/user/work/al18709/tc_Xy_era5_40'
		elif dataset == 'mswep_extend':
			path = '/user/work/al18709/tc_Xy_extend'
		print(tc_X.shape)
		np.save('%s/X_%s.npy' % (path,sid),tc_X)
		np.save('%s/y_%s.npy' % (path,sid),tc_y)
		meta.to_csv('%s/meta_%s.csv' % (path,sid))
		print('saved!')

def save_Xy_era5(grouped_tcs):
	"""
	save X and y npy file for each tc
	"""

	# initial set up
	regex = r"/user/work/al18709/tropical_cyclones/era5/.+?_(.+?)_.*?_idx-(.+?)_.*?_centrelat-(.+?)_centrelon-(.+?).nc"

	# loop through tcs and make an X and y array for each
	for tc in grouped_tcs:
		sid = re.match(regex,tc[0]).groups()[0]	
		n_timesteps = len(tc)
		tc_X = np.zeros((n_timesteps,40,40))
		# meta_lats = []
		# meta_lons = []
		# meta_sids = []
		meta_lats = np.zeros((n_timesteps))
		meta_lons = np.zeros((n_timesteps))
		meta_sids = np.zeros((n_timesteps),dtype=str)
		remove_i = []
		# meta_sids = []

		tc_y = np.zeros((n_timesteps,100,100))

		# loop though tc filepaths
		for i,filepath in enumerate(tc):
			print(i,end='\r')
			
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
			if array.shape != (40,40):
				remove_i.append(i)
				continue

			# assign arrays
			tc_X[i,:,:] = array
			tc_y[i,:,:] = mswep_array
			# meta_lats.append(centre_lat)
			# meta_lons.append(centre_lon)
			# meta_sids.append(str(sid))
			meta_lats[i] = centre_lat
			meta_lons[i] = centre_lon
			meta_sids[i] = str(sid)
			# meta_sids.append(str(sid))
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
		print('saved!')
	
def process(filepaths):
	print('doing process...')
	res = save_Xy(filepaths)
	return res

def process_era5(filepaths):
	print('doing process...')
	res = save_Xy_era5(filepaths)
	return res

if __name__ == '__main__':

	# set up
	n_processes = 128
	dataset = 'mswep'
	dataset = 'era5'
	# dataset = 'imerg'
	dataset = 'mswep_extend'
	
	tc_dir = '/user/work/al18709/tropical_cyclones/%s/*.nc' % dataset
	filepaths = glob.glob(tc_dir)
	# print(filepaths)

	# group by tc sid number
	print('grouping...')
	if dataset == 'mswep':
		regex = r"/user/work/al18709/tropical_cyclones/mswep/.+?_(.+?)_.*?.nc"
	elif dataset == 'imerg':
		regex = r"/user/work/al18709/tropical_cyclones/imerg/.+?_(.+?)_.*?.nc"
	elif dataset == 'era5':
		regex = r"/user/work/al18709/tropical_cyclones/era5/.+?_(.+?)_.*?.nc"
	elif dataset == 'mswep_extend':
		regex = r"/user/work/al18709/tropical_cyclones/mswep_extend/.+?_(.+?)_.*?.nc"
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
	elif dataset == 'era5':
		pool_results = p.map(process_era5, tc_split)
	elif dataset == 'mswep_extend':
		pool_results = p.map(process, tc_split)
	print('results pooled')
	p.close()
	p.join()