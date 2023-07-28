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
# import matplotlib.pyplot as plt
# import cartopy.feature as cfeature
# import cartopy.crs as ccrs
import pandas as pd
# import xesmf as xe
import subprocess,os
from pathlib import Path
from datetime import datetime, timedelta
sns.set_style("white")

# rain = np.load('/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/storm_rain.npy')

def generate_yrmonths():
	# 1979 - 2020
	years = range(1979,2023)
	months = ['01','02','03','04','05','06','07','08','09','10','11','12']
	yrmonths = [ "%s%s" % (year,month) for year in years for month in months]
	return yrmonths




# rainfall_fps = ['/user/home/al18709/work/CMIP6/HighResMIP/CMCC-CM2-VHR4/historical/pr/pr_Prim6hr_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_%s-%s.nc' % (yrmonth,yrmonth) for yrmonth in yrmonths]



# if regrid == True:
# 	# cat > mygrid << EOF
# 	# gridtype = lonlat
# 	# xsize    = 3600
# 	# ysize    = 1800
# 	# xfirst   = -179.95
# 	# xinc     = 0.1
# 	# yfirst   = -89.95
# 	# yinc     = 0.1
# 	# EOF
def regrid(fp):
	new_fp = fp[:-3] + '_regrid.nc'
	cdo_cmd = ['cdo','remapbil,mygrid',fp,new_fp]
	print(' '.join(cdo_cmd))
	# ret = subprocess.call(cdo_cmd)
	ret = subprocess.run(cdo_cmd)
	# if not ret==0:
	# 	raise Exception('Error with cdo command')
	return new_fp

# define initial variables
# model = 'CMCC-CM2-VHR4'
# hemisphere = 'NH'
# experiment = 'HighResMIP' # or 'CMIP6'
# scenario = 'historical'
# resolution = 10

# pr_6hrPlev_MIROC6_historical_r1i1p1f1_gn_198001010300-198012312100.nc
# pr_6hrPlev_MIROC6_historical_r1i1p1f1_gn_198001010300-198001312100.nc


model = 'MPI-ESM1-2-LR'
hemisphere = 'SH'
experiment = 'CMIP6' # or 'CMIP6'
scenario = 'historical'
resolution = 10

# model = 'CMCC-CM2-VHR4'
# hemisphere = 'SH'
# experiment = 'HighResMIP'
# scenario = 'historical'
# resolution = 10

model = 'MPI-ESM1-2-HR'
model_short = 'MPI'
hemisphere = 'NH'
experiment = 'HighResMIP'
scenario = 'historical'
resolution = 10

# /bp1/geog-tropical/data/CMIP6/HighResMIP-rain/CMCC/CMCC-CM2-HR4/hist-1950/r1i1p1f1/Prim6hr/pr/gn/latest
# /bp1/geog-tropical/data/CMIP6/HighResMIP-rain/CMCC/CMCC-CM2-HR4/highres-future/r1i1p1f1/Prim6hr/pr/gn/latest
# /bp1/geog-tropical/data/CMIP6/HighResMIP-rain/MPI/MPI-ESM1-2-HR/highres-future/r1i1p1f1/Prim6hr/pr/gn/latest
if scenario == 'historical':
	mini_scen = 'hist'
	scen_yr_start = '1950'
	scen_yr_end = '2014'

# regrid = False
yrmonths=generate_yrmonths()

# define variables depending on which type of dataset we have
if experiment == 'HighResMIP':
	rainfall_fps = [f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/{scenario}/pr/pr_Prim6hr_{model}_{mini_scen}-{scen_yr_start}_r1i1p1f1_gn_{yrmonth}-{yrmonth}_regrid.nc' for yrmonth in yrmonths]
	if scenario == 'historical':
		s = 'hist-1950'
	else:
		s = 'highres-future'
	rainfall_fps = [f'/bp1/geog-tropical/data/CMIP6/HighResMIP-rain/{model_short}/{model}/{s}/r1i1p1f1/Prim6hr/pr/gn/latest/pr_Prim6hr_{model}_{mini_scen}-{scen_yr_start}_r1i1p1f1_gn_{yrmonth}-{yrmonth}_regrid.nc' for yrmonth in yrmonths]
	time1 = ''
	time2 = ''
	rainfall_dir =  f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/{scenario}/pr/pr_Prim6hr_{model}_{mini_scen}-{scen_yr_start}_r1i1p1f1_gn_'
	rainfall_dir = f'/bp1/geog-tropical/data/CMIP6/HighResMIP-rain/{model_short}/{model}/{s}/r1i1p1f1/Prim6hr/pr/gn/latest/pr_Prim6hr_{model}_{mini_scen}-{scen_yr_start}_r1i1p1f1_gn_'
	tracking_fp = f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/{scenario}/tracks/TC-{hemisphere}_TRACK_{model}_{mini_scen}-{scen_yr_start}_r1i1p1f1_gn_{scen_yr_start}0101-{scen_yr_end}1231.nc'
	new_fp = f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/{scenario}/storm_rain/'
	tracking_ds = xr.open_dataset(tracking_fp)
	storm_start = tracking_ds.FIRST_PT
	storm_duration = tracking_ds.NUM_PTS #first point?
	lats = tracking_ds.lat
	lons = tracking_ds.lon
	time = tracking_ds.time
	track_id = tracking_ds.TRACK_ID #num points?
	w_speed = tracking_ds.sfcWind #time?
elif experiment == 'CMIP6':
	rainfall_fps = [f'/user/home/al18709/work/CMIP6/{model}/pr/{scenario}/pr_6hrPlev_{model}_{scenario}_r1i1p1f1_gn_{yrmonth}010300-{yrmonth}312100_regrid.nc' for yrmonth in yrmonths]
	time1 = '010300'
	time2 = '312100'
	new_fp = f'/user/home/al18709/work/CMIP6/{model}/storm_rain/{scenario}/'
	rainfall_dir = f'/user/home/al18709/work/CMIP6/{model}/pr/{scenario}/pr_6hrPlev_{model}_{scenario}_r1i1p1f1_gn_'
	if hemisphere == 'NH':
		tracking_fps = [f'/user/home/al18709/work/CMIP6/{model}/tracks/{scenario}/{hemisphere}/{model}_tracks_r1i1p1f1_{yr}_newtime2.nc' for yr in range(1979,2015)]
	else:
		tracking_fps = [f'/user/home/al18709/work/CMIP6/{model}/tracks/{scenario}/{hemisphere}/{model}_tracks_r1i1p1f1_{yr}{yr+1}_newtime2.nc' for yr in range(1979,2013)]
	years = range(1979,2015)

	for i,f in enumerate(tracking_fps):
		tracking = xr.open_dataset(f)
		print('number of tracking files: ',len(tracking_fps))
		print(f)
		print(tracking.time)
		# y = years[i]
		# change to correct calendar
		# print(f"changing time units to: hours since {y}-01-01 00")
		# print(tracking.time)
		
		# tracking.time.attrs["units"] = f"hours since {y}-01-01 00"
		# tracking.time.encoding["start"] = f"{y}010100"
		# tracking = xr.decode_cf(tracking)
		# print(tracking.time)

		if i == 0:
			storm_start = tracking.FIRST_PT
			storm_duration = tracking.NUM_PTS #first point?
			lats = tracking.latitude
			lons = tracking.longitude
			time = tracking.time
			track_id = tracking.TRACK_ID #num points?
			w_speed = tracking.wind_speed_850 #time?
			print('w_speed shape: ', w_speed.shape)
		else:
			print(i)
			storm_start = np.concatenate((storm_start,tracking.FIRST_PT))
			storm_duration = np.concatenate((storm_duration,tracking.NUM_PTS))
			lats = np.concatenate((lats,tracking.latitude))
			lons = np.concatenate((lons,tracking.longitude))
			time = np.concatenate((time,tracking.time))
			track_id = np.concatenate((track_id,tracking.TRACK_ID))
			w_speed = np.concatenate((w_speed,tracking.wind_speed_850))
			print('w_speed shape: ', w_speed.shape)
			print(time)
	# coords = {'TRACK_ID': tracking.TRACK_ID , 'time': tracking.time, 'latitude': tracking.latitude, 'longitude': tracking.longitude}
	# dims = ['TRACK_ID','time','latitude','longitude']
	# coords = {'time': tracking.time}
	# dims = ['record']
	# storm_start = xr.DataArray(np.concatenate((storm_start,tracking.FIRST_PT)))
	# storm_duration = xr.DataArray(np.concatenate((storm_duration,tracking.NUM_PTS)))
	# lats = xr.DataArray(np.concatenate((lats,tracking.latitude)))
	# lons = xr.DataArray(np.concatenate((lons,tracking.longitude)))
	# time = xr.DataArray(np.concatenate((time,tracking.time)))
	# track_id = xr.DataArray(np.concatenate((track_id,tracking.TRACK_ID)))
	# w_speed = xr.DataArray(np.concatenate((w_speed,tracking.wind_speed_850)))

	storm_start = xr.DataArray(storm_start)
	storm_duration = xr.DataArray(storm_duration)
	lats = xr.DataArray(lats)
	lons = xr.DataArray(lons)
	time = xr.DataArray(time)	
	track_id = xr.DataArray(track_id)
	w_speed = xr.DataArray(w_speed)


# TODO either combine tracks or loop through each one!
# print('tracks', tracking_ds_tracks.shape)
# print('record', tracking_ds_record.shape)
# define track variables
# storm_start = tracking_ds_tracks.FIRST_PT
# storm_duration = tracking_ds_tracks.NUM_PTS #first point?
# lats = tracking_ds_record.lat
# lons = tracking_ds_record.lon
# time = tracking_ds_record.time
# track_id = tracking_ds_tracks.TRACK_ID #num points?
# w_speed = tracking_ds_record.sfcWind #time?
index = []
# not_tc_idx = np.array(range(0,len(lats)))
all_lats = np.zeros((len(lats),int(1000/resolution)))
all_lons = np.zeros((len(lats),int(1000/resolution)))
# all_rain = np.zeros((len(lats),int(1000/resolution),int(1000/resolution)))
all_rain = np.ones((len(lats),int(1000/resolution),int(1000/resolution)))
# all_id = pd.DataFrame({'sid':[0]*len(lats)})
all_id = pd.DataFrame({'sid':['0']*len(lats),'year':[0]*len(lats),'month':[0]*len(lats),'day':[0]*len(lats),'hour':[0]*len(lats)})
print('processing ',len(lats), ' storm samples...')

print('w speed shape',w_speed.values.shape)
print('storm start len', storm_start.values.shape)

# for each start point, reference the lats and lons from the start point for the duration
storm_track_idx = 0
for j,i in enumerate(storm_start.values):
	print('i is: ',i)
	print('j is: ',j)
	# define storm variables
	npoints = storm_duration.values[j]
	print('npoints: ',npoints)

	if experiment == 'CMIP6':
		if j-1 < 0:
			storm_track_idx = storm_track_idx
		else:
			storm_track_idx = storm_track_idx + storm_duration.values[j-1]
	else:
		storm_track_idx = i
	print('storm_track_idx is: ', storm_track_idx)

	# define tracking variables
	storm_lats = lats.values[storm_track_idx:storm_track_idx+npoints]
	storm_lons = lons.values[storm_track_idx:storm_track_idx+npoints]
	storm_time = time[storm_track_idx:storm_track_idx+npoints]
	storm_speed = w_speed.values[storm_track_idx:storm_track_idx+npoints]
	storm_year = storm_time.dt.year
	storm_month = storm_time.dt.month
	storm_day = storm_time.dt.day
	storm_hour = storm_time.dt.hour

	# storm_id = track_id.values[i:i+npoints]
	
	
	# filter out years where we don't have the rainfall file yet 2015
	if not set(storm_year.values) & set(range(1979,2015)):
	# if not set(storm_year.values) & set([1979]):
		print('year %s not in range' % storm_year.values[0])
		print('storm year: ', storm_year.values)
		# for k in range(npoints):
		# 	# index.append(i+k)
		continue
	else:
		print('processing %s' % storm_year.values[0])
		print('storm year: ', storm_year.values)
		
	
	regrid_files = []

	# find corresponding rainfall file
	for k in range(npoints):
		print('k is: ', k, '/', npoints)
		# define time variables to find correct filepath and array
		year = storm_year.values[k]
		speed = storm_speed[k]
		
		# storm_id_step = storm_id[k]
		if speed < 33.:
			# print(speed)
			# index.append(i+k)
			storm_sid = f'{model}_{scenario}_{year}_{hemisphere}_{j}'
			print('storm too weak',storm_sid)
			# all_id.append(storm_sid)
			continue
		
		print('k2 is:',k)
		print('tc strength storm')

		rain_time = storm_time[k]
		if storm_month.values[k] not in [10,11,12]:
			month = f'0{storm_month.values[k]}'
		else:
			month = storm_month.values[k]
		
		if experiment == 'CMIP6':
			rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}01{time1}-{storm_year.values[k]}12{time2}.nc'
		elif experiment == 'HighResMIP':
			if model == 'MPI-ESM1-2-HR':
				rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}01010556-{storm_year.values[k]}12312356.nc'
			else:
				rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}{month}-{storm_year.values[k]}{month}.nc'
		print('month', month)
		print('year: ', storm_year.values[k])
		print(rainfall_fp)
		
		if Path(rainfall_fp[:-3] + '_regrid.nc').is_file():
			print(rainfall_fp[:-3] + '_regrid.nc')
			regrid_rainfall_fp = rainfall_fp[:-3] + '_regrid.nc'
			print('file already regridded')
		else:
			print('regridding file: ',rainfall_fp)
			regrid_rainfall_fp = regrid(rainfall_fp)
			print('regridded ', rainfall_fp)
		
		regrid_files.append(regrid_rainfall_fp)
		print('regrid files: ',regrid_files)
		rainfall_ds = xr.open_dataset(regrid_rainfall_fp)
		print('rainfall time: ',rainfall_ds.time)

		try:
			rainfall_slice = rainfall_ds.sel(time=rain_time).pr
		except:
			if model == 'MPI-ESM1-2-HR':
				rainfall_slice = rainfall_ds.sel(time=rain_time - np.timedelta64(6,'h') - np.timedelta64(4,'m')).pr # midday is 11:56? or midday is 17:56
			else:
				rainfall_slice = rainfall_ds.sel(time=rain_time + np.timedelta64(3,'h')).pr # midday is 3pm - so 3pm represents the end of the timestep and midday represents the middle of the timestep?

		print('rainfall was timesliced successfully')
		print('rainfall time: ',rainfall_slice.time)

		# find centre lat and lon
		centre_lat = storm_lats[k]
		centre_lon = storm_lons[k]
		print('centre_lon: ',centre_lon)
		print('centre_lat: ',centre_lat)
		if centre_lon > 180:
			centre_lon = centre_lon - 360

		ilon = list(rainfall_slice.lon.values).index(rainfall_slice.sel(lon=centre_lon, method='nearest').lon)
		ilat = list(rainfall_slice.lat.values).index(rainfall_slice.sel(lat=centre_lat, method='nearest').lat)
		
		coord_res = 500/resolution

		lat_lower_bound = int(ilat-coord_res)
		lat_upper_bound = int(ilat+coord_res)
		lon_lower_bound = int(ilon-coord_res)
		lon_upper_bound = int(ilon+coord_res)

		# print(lat_lower_bound)
		# print(lat_upper_bound)
		# print(lon_lower_bound)
		# print(lon_upper_bound)

		print('ilon',ilon)
		print('ilat',ilat)


		# if ilon < 100:
		if ilon < 500/resolution:
			# diff = 10 - ilon
			diff = int(500/resolution - ilon)
			# lon_lower_bound = 512 - diff
			print('diff: ',diff)
			lon_lower_bound = 3600 - diff
			print('lon_lower_bound: ',lon_lower_bound)
			print('lon_upper_bound: ',lon_upper_bound)
			data1 = rainfall_slice.values[lat_lower_bound:lat_upper_bound,lon_lower_bound:-1]
			data2 = rainfall_slice.values[lat_lower_bound:lat_upper_bound,0:lon_upper_bound]
			rain_lats = rainfall_ds.lat.values[lat_lower_bound:lat_upper_bound]
			lon1 = rainfall_ds.lon.values[lon_lower_bound:-1]
			lon2 = rainfall_ds.lon.values[0:lon_upper_bound]
			rain_data = np.concatenate((data1,data2),axis=1)
			rain_lons = np.concatenate((lon1,lon2))

		# elif ilon > 3500:
		elif ilon > 3600 - 500/resolution:
			# diff = 512 - ilon
			diff = int(51 - (3600 - ilon))
			print('diff: ',diff)
			lon_upper_bound = diff
			print('lon_lower_bound: ',lon_lower_bound)
			print('lon_upper_bound: ',lon_upper_bound)
			print('rain shape 1',rainfall_slice.values.shape)
			data1 = rainfall_slice.values[lat_lower_bound:lat_upper_bound,lon_lower_bound:-1]
			data2 = rainfall_slice.values[lat_lower_bound:lat_upper_bound,0:lon_upper_bound]
			print('data1 shape: ',data1.shape)
			print('data2 shape: ',data2.shape)
			rain_lats = rainfall_ds.lat.values[lat_lower_bound:lat_upper_bound]
			lon1 = rainfall_ds.lon.values[lon_lower_bound:-1]
			lon2 = rainfall_ds.lon.values[0:lon_upper_bound]	
			rain_data = np.concatenate((data1,data2),axis=1)
			rain_lons = np.concatenate((lon1,lon2))

		else:
			rain_lats = rainfall_ds.lat.values[lat_lower_bound:lat_upper_bound]
			rain_lons = rainfall_ds.lon.values[lon_lower_bound:lon_upper_bound]
			rain_data = rainfall_slice.values[lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
			
		# change units from kg m-2 s-1 to mm day-1
		# rain_data = rain_data * 86400
		# change units from kg m-2 s-1 to mm 6hr-1, also check if kg m-2 s-1 or kg m-2 6hr-1
		rain_data = rain_data * 86400/4 
		
		print('rain data shape',rain_data.shape)
		if rain_data.shape != (int(1000/resolution),int(1000/resolution)):
			print('wrong data shape')
			print(i,j,k)
			print(rain_data.shape)
			all_lats[storm_track_idx+k] = np.zeros((int(1000/resolution)))
			all_lons[storm_track_idx+k] = np.zeros((int(1000/resolution)))
			all_rain[storm_track_idx+k] = np.zeros((int(1000/resolution),int(1000/resolution)))
			# all_id[i+k] = 0
			# all_id.append('0')
			storm_sid = f'TC_{model}_{scenario}_{year}_{hemisphere}_{j}'
			# all_id = all_id.sid.str.replace([storm_track_idx+k],storm_sid)
			# all_id[storm_track_idx+k].sid = storm_sid
			all_id.sid.iloc[storm_track_idx+k] = storm_sid
			all_id.year.iloc[storm_track_idx+k] = storm_year[k]
			all_id.month.iloc[storm_track_idx+k] = storm_month[k]
			all_id.day.iloc[storm_track_idx+k] = storm_day[k]
			all_id.hour.iloc[storm_track_idx+k] = storm_hour[k]
			index.append(storm_track_idx+k)
			# all_id[i+k] = storm_sid
		else:
			print('k3 is:',k)
			all_lats[storm_track_idx+k] = rain_lats
			all_lons[storm_track_idx+k] = rain_lons
			all_rain[storm_track_idx+k] = rain_data
			storm_sid = f'TC_{model}_{scenario}_{year}_{hemisphere}_{j}'
			# all_id.append(storm_sid)
			# all_id[i+k] = storm_sid
			print('storm sid is: ',storm_sid)
			print('storm track id: ', storm_track_idx)
			print('k',k)
			print(storm_track_idx+k)
			# all_id = all_id.sid.str.replace([storm_track_idx+k],storm_sid)
			# all_id[storm_track_idx+k].sid = storm_sid
			all_id.sid.iloc[storm_track_idx+k] = storm_sid
			all_id.year.iloc[storm_track_idx+k] = storm_year[k]
			all_id.month.iloc[storm_track_idx+k] = storm_month[k]
			all_id.day.iloc[storm_track_idx+k] = storm_day[k]
			all_id.hour.iloc[storm_track_idx+k] = storm_hour[k]
			print(all_id)
			# not_tc_idx = np.delete(not_tc_idx,storm_track_idx+k)
			index.append(storm_track_idx+k)
			# all_id[i+k] = f'TC_EC-Earth3p_hist_{year}_NH_{j}'
			# all_id.append(f'TC_EC-Earth3p_hist_{year}_NH_{j}')
			print(storm_sid)

# moved this to left because for some reason it keeps regridding the same files
# for f in regrid_files:
# 	if Path(f).is_file():
# 		print('removing ', f)
# 		ret = subprocess.call(['rm',f])
# 		if not ret==0:
# 			raise Exception('Error with remove command')
# 	else:
# 		continue

	if experiment == 'HighResMIP':
		# rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}01{time1}-{storm_year.values[k]}12{time2}.nc'
		path = f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/{scenario}/pr/'
	elif experiment == 'CMIP6':
		# rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}{month}-{storm_year.values[k]}{month}.nc'
		path = f'/user/home/al18709/work/CMIP6/{model}/pr/{scenario}/'

	regrid_files = os.listdir(path)
	# print(regrid_files)
	n_files = 0
	for item in regrid_files:
		# print(item)
		if item.endswith("regrid.nc"):
			n_files = n_files+1
	if n_files > 2:
		for i in regrid_files:
			print(i)
			if i.endswith("regrid.nc"):
				print('removing ',path,i)
				os.remove(os.path.join(path, i))

	# subprocess.run(['rm', path])

print('number of weak timesteps: ',len(index))
print('removed duplicates: ',set(index))
print(len(set(index)))
print(index)
print('size of rain array: ',all_rain.shape)
print('size of rain array with weak storms omitted: ',np.delete(all_rain,index,axis=0).shape)
print('size of sids: ',all_id.shape)
print(np.sum(all_id!=0))
# all_id = all_id.drop(index)
# print('size of sids: ',all_id.shape)
# all_rain = np.delete(all_rain,index,axis=0)

def exclude_weak(x,exclude):
	return x[~np.isin(np.arange(len(x)), exclude)]

# all_rain = exclude_weak(all_rain,index)
# all_lats = exclude_weak(all_lats,index)
# all_lons = exclude_weak(all_lons,index)

all_rain = all_rain[index,:,:]
all_lats = all_lats[index]
all_lons = all_lons[index]
all_id = all_id.iloc[index]

# all_id = all_id.iloc[all_id.index[~np.isin(np.arange(len(all_id)), index)]]
# print(len(not_tc_idx))
# all_rain = exclude_weak(all_rain,not_tc_idx)
# all_lats = exclude_weak(all_lats,not_tc_idx)
# all_lons = exclude_weak(all_lons,not_tc_idx)



# all_id = all_id.iloc[all_id.index[~np.isin(np.arange(len(all_id)), not_tc_idx)]]
# all_id = all_id.iloc[all_id.sid[~np.isin(np.arange(len(all_id)), not_tc_idx)]]
# all_id = all_id.iloc[all_id.sid[~np.isin(np.arange(len(all_id)), index)]]
# why is this not updating properly?

print(all_rain.shape)
print(all_lats.shape)
print(all_lons.shape)
print(all_id.shape)
print(all_id)
print(new_fp)

np.save(f'{new_fp}storm_lats_{hemisphere}.npy',all_lats)
np.save(f'{new_fp}storm_lons_{hemisphere}.npy',all_lons)
np.save(f'{new_fp}storm_rain_{hemisphere}.npy',all_rain)
np.save(f'{new_fp}storm_index_{hemisphere}.npy',index)
# np.save('{new_fp}storm_sid.npy',all_id)
all_id.to_csv(f'{new_fp}storm_sid_{hemisphere}.csv')
print('files saved!')

