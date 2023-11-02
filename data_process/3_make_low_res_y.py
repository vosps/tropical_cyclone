
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


grid_in = xr.Dataset({'longitude': np.linspace(0, 100, 100),
				'latitude': np.linspace(-50, 50, 100)
				})
grid_out = xr.Dataset({'longitude': np.linspace(0, 100, 10),
					'latitude': np.linspace(-50, 50, 10)
				})
	
# regrid with conservative interpolation so means are conserved spatially
regridder = xe.Regridder(grid_in, grid_out, 'conservative')
print('regrid set up!')

tc_y = np.load('/user/work/al18709/tc_data_flipped/KE_tracks/extreme_test_y.npy')

nsamples,_,_ = tc_y.shape
tc_y_regrid = np.zeros((nsamples,10,10))
print(tc_y_regrid.shape)

for i in range(nsamples):
	print(i,end='\n')
	# print(regridder(tc_y).shape)
	tc_y_regrid[i,:,:] = regridder(tc_y[i,:,:])

print(tc_y_regrid.shape)
print(tc_y.shape)

np.save('/user/work/al18709/tc_data_flipped/KE_tracks/extreme_test_y_regrid.npy',tc_y_regrid)