"""
Take the HighResMIP data and cross reference with either TRACK or TempestExtreme tracking data

inputs : highresmip rainfall data and tracking
outputs : save TC rainfall images for each TC track
https://esgf-node.llnl.gov/search/cmip6/
https://data.ceda.ac.uk/badc/highresmip-derived/data/storm_tracks/TRACK/EC-Earth-Consortium
"""

import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
sns.set_style("white")

rain = np.load('/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/storm_rain.npy')
print(rain.shape)


tracking_fp = '/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/TC-NH_TRACK_EC-Earth3P_hist-1950_r1i1p2f1_gr_19500101-20141231.nc'

rainfall_fps = ['/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/pr_3hr_EC-Earth3P_hist-1950_r1i1p2f1_gr_%s01010000-%s12312100.nc' % (year,year) for year in range(2004,2015)]

tracking_ds = xr.open_dataset(tracking_fp)
print(tracking_ds.variables)
# exit()

# define track variables
storm_start = tracking_ds.FIRST_PT
storm_duration = tracking_ds.NUM_PTS
lats = tracking_ds.lat
lons = tracking_ds.lon
time = tracking_ds.time
wind_speed = tracking_ds.wind_speed

print(time.values)
index = []
all_lats = np.zeros((len(lats),10))
all_lons = np.zeros((len(lats),10))
all_rain = np.zeros((len(lats),10,10))

# for each start point, reference the lats and lons from the start point for the duration
for j,i in enumerate(storm_start.values):
	print(i)
	# define storm variables
	npoints = storm_duration.values[j]
	print(npoints)
	print(len(lats))
	storm_lats = lats.values[i:i+npoints]
	storm_lons = lons.values[i:i+npoints]
	storm_time = time[i:i+npoints]
	storm_speed = wind_speed.values[i:i+npoints]
	
	# define time variables
	storm_year = storm_time.dt.year
	storm_month = storm_time.dt.month
	storm_day = storm_time.dt.day
	storm_hour = storm_time.dt.hour
	# filter out years where we don't have the rainfall file yet
	if not set(storm_year.values) & set(range(2004,2015)):
		print('year %s not in range' % storm_year.values[0])
		continue
	else:
		print('processing %s' % storm_year.values[0])

	# find corresponding rainfall file
	for k in range(npoints):
		# define time variables to find correct filepath and array
		year = storm_year.values[k]
		speed = storm_speed[k]
		if speed < 33.:
			index.append(i+k)
			continue

		rain_time = storm_time[k]
		rainfall_fp = '/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/pr_3hr_EC-Earth3P_hist-1950_r1i1p2f1_gr_%s01010000-%s12312100.nc' % (year,year)
		rainfall_ds = xr.open_dataset(rainfall_fp)
		rainfall_slice = rainfall_ds.sel(time=rain_time).pr

		# find centre lat and lon
		centre_lat = storm_lats[k]
		centre_lon = storm_lons[k]

		ilon = list(rainfall_ds.lon.values).index(rainfall_ds.sel(lon=centre_lon, method='nearest').lon)
		ilat = list(rainfall_ds.lat.values).index(rainfall_ds.sel(lat=centre_lat, method='nearest').lat)

		lat_lower_bound = ilat-5
		lat_upper_bound = ilat+5
		lon_lower_bound = ilon-5
		lon_upper_bound = ilon+5

		if ilon < 5:
			diff = 10 - ilon
			lon_lower_bound = 512 - diff
			data1 = rainfall_ds.pr.values[lat_lower_bound:lat_upper_bound,lon_lower_bound:-1]
			data2 = rainfall_ds.pr.values[lat_lower_bound:lat_upper_bound,0:lon_upper_bound]
			rain_lats = rainfall_ds.lat.values[lat_lower_bound:lat_upper_bound]
			lon1 = rainfall_ds.lon.values[lon_lower_bound:-1]
			lon2 = rainfall_ds.lon.values[0:lon_upper_bound]
			
			rain_data = np.concatenate((data1,data2),axis=1)
			rain_lons = np.concatenate((lon1,lon2))
		elif ilon > 507:
			diff = 512 - ilon
			lon_upper_bound = diff
			data1 = rainfall_ds.pr.values[lat_lower_bound:lat_upper_bound,lon_lower_bound:-1]
			data2 = rainfall_ds.pr.values[lat_lower_bound:lat_upper_bound,0:lon_upper_bound]
			rain_lats = rainfall_ds.lat.values[lat_lower_bound:lat_upper_bound]
			lon1 = rainfall_ds.lon.values[lon_lower_bound:-1]
			lon2 = rainfall_ds.lon.values[0:lon_upper_bound]
			
			rain_data = np.concatenate((data1,data2),axis=1)
			rain_lons = np.concatenate((lon1,lon2))

		else:
			rain_lats = rainfall_ds.lat.values[lat_lower_bound:lat_upper_bound]
			rain_lons = rainfall_ds.lon.values[lon_lower_bound:lon_upper_bound]
			rain_data = rainfall_slice.values[lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
		
		# change units from kg m-2 s-1 to mm h-1
		rain_data = rain_data * 86400 

		# plot to check
		# lat2d,lon2d = np.meshgrid(rain_lats,rain_lons)
		# fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
		# c = ax.pcolor(lon2d,lat2d,rain_data,vmin=0,vmax=60,cmap = 'Blues', transform=ccrs.PlateCarree())
		# ax.add_feature(cfeature.COASTLINE)
		# ax.outline_patch.set_linewidth(0.3)
		# cbar = plt.colorbar(c)
		# cbar.ax.tick_params(labelsize=6,width=0.5)
		# plt.title('HighResMIP TC')
		# plt.savefig('figs/test_cmip_tc.png')
		# plt.clf()
		if rain_data.shape != (10,10):
			print(i,j,k)
			print(rain_data.shape)
			all_lats[i+k] = np.zeros((10))
			all_lons[i+k] = np.zeros((10))
			all_rain[i+k] = np.zeros((10,10))
		else:
			all_lats[i+k] = rain_lats
			all_lons[i+k] = rain_lons
			all_rain[i+k] = rain_data

np.save('/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/storm_lats.npy',all_lats)
np.save('/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/storm_lons.npy',all_lons)
np.save('/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/storm_rain.npy',all_rain)
np.save('/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/storm_index.npy',index)


