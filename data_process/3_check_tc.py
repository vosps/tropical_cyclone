"""
	3. Plot a saved tc to check what it looks like

"""

import glob
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

# open dataset
dataset = 'mswep'
fp = glob.glob('/user/work/al18709/tropical_cyclones/%s/*.nc' % dataset)[15]
ds = xr.open_dataset(fp)
data = ds.precipitation
lats = ds.y
lons = ds.x

print(data)
print(lats)
print(lons)

# use either raw or processed data
# data = np.load('/user/work/al18709/tc_data/extreme_valid_y.npy')[99]
# data = np.load('/user/home/al18709/work/cgan_predictions/validation_real.npy')[0,99,:,:,0]
data = np.load('/user/work/al18709/tc_data/valid_y.npy')[6,:,:]
# data = np.load('/user/work/al18709/tc_Xy/y_2020256N25281.npy')[0,:,:]
# data = np.load('/user/work/al18709/tc_Xy/y_2010293N17277.npy')[0,:,:]
# print('data shape: ',data.shape)

lat2d,lon2d = np.meshgrid(lats,lons)
# data = data.where(data>0.00001)

# plot
fig, ax = plt.subplots()
cmap ='seismic_r'
c = ax.pcolor(lon2d,lat2d,data,vmin=-60,vmax=60,cmap = cmap,)
cbar = plt.colorbar(c, shrink=0.54)
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(labelsize=6,width=0.5)
plt.savefig('figs/tc_plot_test_mswep.png',dpi=600,bbox_inches='tight')
plt.clf()
print(fp)

fp = '/bp1store/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2012220.09.nc'
ds = xr.open_dataset(fp)
data = ds.precipitation
lats = ds.lat
lons = ds.lon

# print(lats.values)
# print(lons.values)
