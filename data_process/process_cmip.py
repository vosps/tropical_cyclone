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
import pandas as pd
import xesmf as xe
sns.set_style("white")

rain = np.load('/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/storm_rain.npy')
print(rain.shape)

def generate_yrmonths():
	# 1979 - 2020
	years = range(1979,2023)
	print(list(years))
	months = ['01','02','03','04','05','06','07','08','09','10','11','12']
	yrmonths = [ "%s%s" % (year,month) for year in years for month in months]
	return yrmonths

yrmonths=generate_yrmonths()

tracking_fp = '/user/home/al18709/work/CMIP6/HighResMIP/CMCC-CM2-VHR4/historical/tracks/TC-NH_TRACK_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_19500101-20141231.nc'
rainfall_fps = ['/user/home/al18709/work/CMIP6/HighResMIP/CMCC-CM2-VHR4/historical/pr/pr_Prim6hr_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_%s-%s.nc' % (yrmonth,yrmonth) for yrmonth in yrmonths]
resolution = 25
tracking_ds = xr.open_dataset(tracking_fp)
print(tracking_ds.variables)
# print(tracking_ds.variables.list)
# exit()

# define track variables
storm_start = tracking_ds.FIRST_PT
storm_duration = tracking_ds.NUM_PTS #first point?
lats = tracking_ds.lat
lons = tracking_ds.lon
time = tracking_ds.time
track_id = tracking_ds.TRACK_ID #num points?
w_speed = tracking_ds.sfcWind #time?
print(np.sum(storm_duration))
print(time.values)
index = []
all_lats = np.zeros((len(lats),int(1000/resolution)))
all_lons = np.zeros((len(lats),int(1000/resolution)))
all_rain = np.zeros((len(lats),int(1000/resolution),int(1000/resolution)))
print(len(lats))

# all_id = np.zeros((len(lats)))
all_id = []

# for each start point, reference the lats and lons from the start point for the duration
for j,i in enumerate(storm_start.values):
	print('i is: ',i)
	print('j is: ',j)
	# define storm variables
	npoints = storm_duration.values[j] 
	print(npoints)
	print(len(lats))
	storm_lats = lats.values[i:i+npoints]
	storm_lons = lons.values[i:i+npoints]
	storm_time = time[i:i+npoints]
	storm_speed = w_speed.values[i:i+npoints]
	# storm_id = track_id.values[i:i+npoints]
	# print('storm_id',storm_id)
	
	# define time variables
	storm_year = storm_time.dt.year
	storm_month = storm_time.dt.month
	storm_day = storm_time.dt.day
	storm_hour = storm_time.dt.hour
	# filter out years where we don't have the rainfall file yet
	if not set(storm_year.values) & set(range(1979,2015)):
		print('year %s not in range' % storm_year.values[0])
		for k in range(npoints):
			index.append(i+k)
		continue
	else:
		print('processing %s' % storm_year.values[0])

	# find corresponding rainfall file
	for k in range(npoints):
		# define time variables to find correct filepath and array
		year = storm_year.values[k]
		speed = storm_speed[k]
		
		# storm_id_step = storm_id[k]
		if speed < 33.:
			# print(speed)
			index.append(i+k)
			continue
		
		print('k2 is:',k)
		rain_time = storm_time[k]
		print(storm_year.values[k])
		print(storm_month.values[k])
		if storm_month.values[k] not in [10,11,12]:
			month = f'0{storm_month.values[k]}'
		else:
			month = storm_month.values[k]
		rainfall_fp = f'/user/home/al18709/work/CMIP6/HighResMIP/CMCC-CM2-VHR4/historical/pr/pr_Prim6hr_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_{storm_year.values[k]}{month}-{storm_year.values[k]}{month}.nc'
		rainfall_ds = xr.open_dataset(rainfall_fp)
		rainfall_slice = rainfall_ds.sel(time=rain_time).pr

		# find centre lat and lon
		centre_lat = storm_lats[k]
		centre_lon = storm_lons[k]

		ilon = list(rainfall_ds.lon.values).index(rainfall_ds.sel(lon=centre_lon, method='nearest').lon)
		ilat = list(rainfall_ds.lat.values).index(rainfall_ds.sel(lat=centre_lat, method='nearest').lat)
		
		coord_res = 500/resolution

		lat_lower_bound = int(ilat-coord_res)
		lat_upper_bound = int(ilat+coord_res)
		lon_lower_bound = int(ilon-coord_res)
		lon_upper_bound = int(ilon+coord_res)

		print(lat_lower_bound)
		print(lat_upper_bound)
		print(lon_lower_bound)
		print(lon_upper_bound)


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
		print('rain data shape',rain_data.shape)
		if rain_data.shape != (int(1000/resolution),int(1000/resolution)):
			print(i,j,k)
			print(rain_data.shape)
			all_lats[i+k] = np.zeros((int(1000/resolution)))
			all_lons[i+k] = np.zeros((int(1000/resolution)))
			all_rain[i+k] = np.zeros((int(1000/resolution),int(1000/resolution)))
			# all_id[i+k] = 0
			# all_id.append('0')
			storm_sid = f'TC_EC-Earth3p_hist_{year}_NH_{j}'
			all_id.append(storm_sid)
		else:
			print('k3 is:',k)
			all_lats[i+k] = rain_lats
			all_lons[i+k] = rain_lons
			all_rain[i+k] = rain_data
			storm_sid = f'TC_EC-Earth3p_hist_{year}_NH_{j}'
			all_id.append(storm_sid)
			# all_id[i+k] = f'TC_EC-Earth3p_hist_{year}_NH_{j}'
			# all_id.append(f'TC_EC-Earth3p_hist_{year}_NH_{j}')
print('number of weak timesteps: ',len(index))
print('size of rain array: ',all_rain.shape)
print('size of rain array with weak storms omitted: ',np.delete(all_rain,index,axis=0).shape)
print('size of sids: ',len(all_id))


# TODO: regrid all arrays to be 100 x 100 when they are saved!
# define grid, this doesn't need to be specific, only needs to be the correct resolution
resolution = 25
grid_in = xr.Dataset({'longitude': np.linspace(0, 100, int(1000/resolution)),
		'latitude': np.linspace(-50, 50, int(1000/resolution))
		})

# output grid has a the same coverage at finer resolution
grid_out = xr.Dataset({'longitude': np.linspace(0, 100, 100),
				'latitude': np.linspace(-50, 50, 100)
			})

# regrid with conservative interpolation so means are conserved spatially
regridder = xe.Regridder(grid_in, grid_out, 'bilinear')
all_rain = regridder(np.delete(all_rain,index,axis=0))

np.save('/user/home/al18709/work/CMIP6/HighResMIP/CMCC-CM2-VHR4/historical/storm_rain/storm_lats.npy',np.delete(all_lats,index,axis=0))
np.save('/user/home/al18709/work/CMIP6/HighResMIP/CMCC-CM2-VHR4/historical/storm_rain/storm_lons.npy',np.delete(all_lons,index,axis=0))
np.save('/user/home/al18709/work/CMIP6/HighResMIP/CMCC-CM2-VHR4/historical/storm_rain/storm_rain.npy',all_rain)
np.save('/user/home/al18709/work/CMIP6/HighResMIP/CMCC-CM2-VHR4/historical/storm_rain/storm_index.npy',index)
# np.save('/user/home/al18709/work/CMIP6/HighResMIP/CMCC-CM2-VHR4/historical/storm_rain/storm_sid.npy',all_id)
pd.DataFrame({'sid':all_id}).to_csv('/user/home/al18709/work/CMIP6/HighResMIP/CMCC-CM2-VHR4/historical/storm_rain/storm_sid.csv')


