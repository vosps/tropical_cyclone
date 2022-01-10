"""
This script runs the CGAN on a selection of tropical cyclones from global climate models
"""

import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

path = '/user/home/al18709/work/gcm_tc/'

df = pd.read_csv('storms.csv')

centre_lats = df['centre_lat']
centre_lons = df['centre_lon']
filename = df['filename'][0]
times = df['time']
fp = path + filename
ds = xr.open_dataset(fp)
lat = ds.lat #lat
lon = ds.lon #lon
tcs = np.zeros((5,10,10))

for i,t in enumerate(times):
	centre_lat = centre_lats[i]
	centre_lon = centre_lons[i] + 360

	# clip to location
	
	lower_lat = (np.abs(lat.values-centre_lat+5.)).argmin()
	upper_lat = (np.abs(lat.values-centre_lat-5.)).argmin()
	lower_lon = (np.abs(lon.values-centre_lon+5.)).argmin()
	upper_lon = (np.abs(lon.values-centre_lon-5.)).argmin()

	if upper_lat - lower_lat > 10:
		upper_lat = upper_lat - 1
	elif upper_lat - lower_lat < 10:
		lower_lat = lower_lat - 1

	if upper_lon - lower_lon > 10:
		upper_lon = upper_lon - 1
	elif upper_lon - lower_lon < 10:
		lower_lon = lower_lon - 1

	
	# crop data and convert to mm/hr
	data = np.array(ds.pr[t+1,lower_lat:upper_lat,lower_lon:upper_lon]) * 86400
	data = (60*(data - np.min(data))/np.ptp(data))
	lats = lat[lower_lat:upper_lat]
	lons = lon[lower_lon:upper_lon]
	print(data.shape)
	tcs[i] = data

print(tcs)
print(tcs.shape)
np.save('/user/work/al18709/tc_data/gcm_X.npy',tcs)
# plot
lat2d,lon2d = np.meshgrid(lats,lons)
fig, ax = plt.subplots()
cmap ='seismic_r'
c = ax.pcolor(lon2d,lat2d,data,vmin=-60,vmax=60,cmap = cmap,)
cbar = plt.colorbar(c, shrink=0.54)
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(labelsize=6,width=0.5)
plt.savefig('test.png',dpi=600,bbox_inches='tight')
plt.clf()
print(fp)
