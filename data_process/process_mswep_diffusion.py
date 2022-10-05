import xarray as xr
import numpy as np
import xesmf as xe


datasets = ['train','valid','test','extreme_valid','extreme_test']
grid_in = xr.Dataset({'longitude': np.linspace(0, 100, 10),
					'latitude': np.linspace(-50, 50, 10)
				})
grid_out = xr.Dataset({'longitude': np.linspace(0, 100, 100),
				'latitude': np.linspace(-50, 50, 100)
				})

regridder = xe.Regridder(grid_in, grid_out, 'nearest_s2d')

for dataset in datasets:
	X_fp = '/user/home/al18709/work/tc_data_flipped/%s_X.npy' % dataset
	X = np.load(X_fp)
	X_hr = regridder(X)