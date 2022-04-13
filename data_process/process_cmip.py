"""
Take the HighResMIP data and cross reference with either TRACK or TempestExtreme tracking data

inputs : highresmip rainfall data and tracking
outputs : save TC rainfall images for each TC track
"""

import xarray as xr
import numpy as np

tracking_fp = '/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/TC-NH_TRACK_EC-Earth3P_hist-1950_r1i1p2f1_gr_19500101-20141231.nc'

rainfall_fps = ['/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/pr_3hr_EC-Earth3P_hist-1950_r1i1p2f1_gr_%s01010000-%s12312100.nc' % (year,year) for year in range(2004,2015)]

tracking_ds = xr.open_dataset(tracking_fp)
print(tracking_ds)

# define track variables
storm_start = tracking_ds.FIRST_PT
storm_duration = tracking_ds.NUM_PTS
lats = tracking_ds.lat
lons = tracking_ds.lon
time = tracking_ds.time

print(time.values)


# for each start point, reference the lats and lons from the start point for the duration
for j,i in enumerate(storm_start.values):
	
	# define storm variables
	npoints = storm_duration.values[j]
	storm_lats = lats.values[i:i+npoints]
	storm_lons = lons.values[i:i+npoints]
	storm_time = time[i:i+npoints]
	
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
		time = storm_time[k]
		print(year)
		print(time)
		rainfall_fp = '/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/pr_3hr_EC-Earth3P_hist-1950_r1i1p2f1_gr_%s01010000-%s12312100.nc' % (year,year)
		rainfall_ds = xr.open_dataset(rainfall_fp)
		print(rainfall_ds.time.values)
		rainfall_slice = rainfall_ds.sel(time=time).pr
		print(rainfall_slice)
		print(rainfall_slice.shape)

		# find centre lat and lon
		centre_lat = storm_lats[k]
		centre_lon = storm_lons[k]

		print(centre_lat)
		print(centre_lon)
		print(storm_lats)

		# print(lons)
		print(lats.values)
		print(rainfall_ds.lat)

		# centre_lat_i = np.where(lats == centre_lat)
		# centre_lon_i = np.where(lons == centre_lon)
		# centre_lat_i = rainfall_ds.lat.sel(lat = centre_lat,method='nearest')
		# centre_lon_i = rainfall_ds.lon.sel(lon = centre_lon,method='nearest')

		ilon = list(rainfall_ds.lon.values).index(rainfall_ds.sel(lon=centre_lon, method='nearest').lon)
		ilat = list(rainfall_ds.lat.values).index(rainfall_ds.sel(lat=centre_lat, method='nearest').lat)

		# print(centre_lat_i.values)
		# print(centre_lon_i.values)
		print(ilon)
		print(ilat)

		lat_lower_bound = ilat-5
		lat_upper_bound = ilat+5
		lon_lower_bound = ilon-5
		lon_upper_bound = ilon+5

		print(data)
		# print(lat)
		# print(lon)
		exit()



		if k == 2:
			exit()


