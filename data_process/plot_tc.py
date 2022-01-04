"""
This script will locate a tc using ibtracks data and plot the corresponding rainfall using IMERG data
"""

import glob
from netCDF4 import Dataset
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
sns.set_style("white")
import scipy.ndimage
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
 
 
def create_color(r, g, b):
	return [r/256, g/256, b/256]
 
def get_custom_color_palette():
	return LinearSegmentedColormap.from_list("", [
		create_color(255, 255, 255), #whites
		create_color(234, 239, 252), create_color(75, 82, 252), #blues
		create_color(252, 75, 179)
	])

year_day = '20050828' # 155959 

year_day = '20040913'
precip_file = glob.glob('/bp1store/geog-tropical/data/Obs/IMERG/half_hourly/final/3B-HHR.MS.MRG.3IMERG.%s*165959*.HDF5' % year_day)[0]

d = Dataset(precip_file, 'r')
lat = d['Grid'].variables['lat'][:] #lat
lon = d['Grid'].variables['lon'][:] #lon
print(lat)
print(lon)
# clip to location
lat_lower_bound = (np.abs(lat-16.)).argmin()
lat_upper_bound = (np.abs(lat-34.)).argmin()
lon_lower_bound = (np.abs(lon-(-92.))).argmin()
lon_upper_bound = (np.abs(lon-(-63.))).argmin()

data = d['Grid'].variables['precipitationCal'][0,lon_lower_bound:lon_upper_bound,lat_lower_bound:lat_upper_bound]
lat = lat[lat_lower_bound:lat_upper_bound]
lon = lon[lon_lower_bound:lon_upper_bound]
d.close()
lat2d,lon2d = np.meshgrid(lat,lon)
# data = scipy.ndimage.zoom(data, 0.1)
# lat2d = scipy.ndimage.zoom(lat2d, 0.1)
# lon2d = scipy.ndimage.zoom(lon2d, 0.1)
data[data < 0.001] = np.nan

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
cmap = get_custom_color_palette()
cmap ='seismic_r'
# c = ax.contourf(lon2d,lat2d,data,60,vmin=-60,vmax=60,cmap = cmap, transform=ccrs.PlateCarree())
c = ax.pcolor(lon2d,lat2d,data,vmin=-60,vmax=60,cmap = cmap, transform=ccrs.PlateCarree())
# ax.add_feature(cfeature.LAND)
ax.outline_patch.set_linewidth(0.3)
cbar = plt.colorbar(c, shrink=0.54)
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(labelsize=6,width=0.5)
# plt.title('Extreme')

# plt.savefig('figs/tc_plot.png',dpi=600,bbox_inches='tight',transparent=True)
plt.savefig('figs/tc_plot.png',dpi=600,bbox_inches='tight')
plt.clf()
print(data)
