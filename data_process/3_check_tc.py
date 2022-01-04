"""
	3. Plot a saved tc to check what it looks like

"""

import glob
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import seaborn as sns
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import matplotlib.pyplot as plt
sns.set_style("white")

# open dataset
dataset = 'mswep'
fp = glob.glob('/user/work/al18709/tropical_cyclones/%s/*.nc' % dataset)[99]
ds = xr.open_dataset(fp)
print(ds)
data = ds.precipitation
lats = ds.y
lons = ds.x

data = np.load('/user/work/al18709/tc_data/extreme_valid_y.npy')[99]
data = np.load('/user/home/al18709/work/cgan_predictions/validation_real.npy')[0,99,:,:,0]
print(data.shape)

lat2d,lon2d = np.meshgrid(lats,lons)
# data = data.where(data>0.00001)

# plot
fig, ax = plt.subplots()
cmap ='seismic_r'
c = ax.pcolor(lon2d,lat2d,data,vmin=-30,vmax=30,cmap = cmap,)
# ax.add_feature(cfeature.LAND)
# ax.outline_patch.set_linewidth(0.3)
cbar = plt.colorbar(c, shrink=0.54)
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(labelsize=6,width=0.5)
# plt.savefig('figs/tc_plot.png',dpi=600,bbox_inches='tight',transparent=True)
plt.savefig('figs/tc_plot_test_mswep.png',dpi=600,bbox_inches='tight')
plt.clf()
print(fp)