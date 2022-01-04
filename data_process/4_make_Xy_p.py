"""
	4. Create X, y and test and training sets
	
This script formats the X and y file as a numpy array. Each tc will have it's own X and y npy file

	X shape : (n_timesteps, 10, 10)
	y shape : (n_timesteps,100,100)

saved in : /user/work/al18709/tc_data/

"""

import glob
import xarray as xr
import numpy as np
import xesmf as xe
from itertools import groupby
from multiprocessing import Pool
import re

def save_Xy(grouped_tcs):
	"""
	save X and y npy file for each tc
	"""

	# initial set up
	if dataset == 'mswep':
		regex = r"/user/work/al18709/tropical_cyclones/mswep/.+?_(.+?)_.*?.nc"
	elif datase == 'imerg':
		regex = r"/user/work/al18709/tropical_cyclones/imerg/.+?_(.+?)_.*?.nc"
	# regex = r"/user/work/al18709/tropical_cyclones/.+?_(.+?)_.*?.nc"
	
	# define grid, this doesn't need to be specific, only needs to be the correct resolution
	# grid_in = xr.Dataset({'longitude': np.linspace(0, 100, 100),
	# 		'latitude': np.linspace(-50, 50, 100)
	# 		})
	# # output grid has a the same coverage at finer resolution
	# grid_out = xr.Dataset({'longitude': np.linspace(0, 100, 10),
	# 			'latitude': np.linspace(-50, 50, 10)
			# })
	
	# define grid, this doesn't need to be specific, only needs to be the correct resolution
	grid_in = xr.Dataset({'longitude': np.linspace(50, -50, 100),
			'latitude': np.linspace(-50, 50, 100)
			})
	# output grid has a the same coverage at finer resolution
	grid_out = xr.Dataset({'longitude': np.linspace(50, -50, 10),
				'latitude': np.linspace(-50, 50, 10)
			})

	# regrid with conservative interpolation so means are conserved spatially
	regridder = xe.Regridder(grid_in, grid_out, 'conservative')

	# loop through tcs and make an X and y array for each
	for tc in grouped_tcs:
		sid = re.match(regex,tc[0]).groups()[0]
		print(sid)
		n_timesteps = len(tc)
		tc_X = np.zeros((n_timesteps,10,10))
		tc_y = np.zeros((n_timesteps,100,100))
		# loop though tc filepaths
		for i,filepath in enumerate(tc):
			ds = xr.open_dataset(filepath)
			array = ds.precipitation.values

			# for now, remove irregularly shaped arrays
			if array.shape != (100,100):
				continue
				# TODO: going to end up with some zero values here - fix that
			
			tc_X[i,:,:] = regridder(array)
			tc_y[i,:,:] = array
		
		np.save('/user/work/al18709/tc_Xy/X_%s.npy' % sid,tc_X)
		np.save('/user/work/al18709/tc_Xy/y_%s.npy' % sid,tc_y)
	
def process(filepaths):
	print('doing process...')
	res = save_Xy(filepaths)
	return res

if __name__ == '__main__':

	# set up
	n_processes = 64
	dataset = 'mswep'
	tc_dir = '/user/work/al18709/tropical_cyclones/%s/*.nc' % dataset
	filepaths = glob.glob(tc_dir)
	print(filepaths)
	

	# group by tc sid number
	if dataset == 'mswep':
		regex = r"/user/work/al18709/tropical_cyclones/mswep/.+?_(.+?)_.*?.nc"
	elif dataset == 'imerg':
		regex = r"/user/work/al18709/tropical_cyclones/imerg/.+?_(.+?)_.*?.nc"
	keyf = lambda text: (re.findall(regex, text)+ [text])[0]
	grouped_tcs = [list(items) for gr, items in groupby(sorted(filepaths), key=keyf)]
	

	# split into 64 list of tc filepaths
	tc_split = np.array(np.array_split(grouped_tcs, n_processes))

	# start processes
	p = Pool(processes=64)
	print('pooling processes...')
	pool_results = p.map(process, tc_split)
	print('results pooled')
	p.close()
	p.join()