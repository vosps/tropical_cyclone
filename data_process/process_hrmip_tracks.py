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
import pandas as pd
# import xesmf as xe
import subprocess,os
from pathlib import Path
from datetime import datetime, timedelta
from cftime import DatetimeNoLeap
import math
import cftime as cf
from calendar import monthrange
sns.set_style("white")

# rain = np.load('/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/storm_rain.npy')

def generate_yrmonths():
	# 1979 - 2020
	years = range(1979,2023)
	months = ['01','02','03','04','05','06','07','08','09','10','11','12']
	yrmonths = [ "%s%s" % (year,month) for year in years for month in months]
	return yrmonths

def convert_to_np_datetime64(date):
    return np.datetime64(np.datetime_as_string(date, unit='s'))

def round_nearest(n, r):
    return  round(n - math.fmod(n, r),2)

def regrid(fp):
	new_fp = fp[:-3] + '_regrid.nc'
	cdo_cmd = ['cdo','remapbil,mygrid',fp,new_fp]
	print(' '.join(cdo_cmd))
	# ret = subprocess.call(cdo_cmd)
	ret = subprocess.run(cdo_cmd)
	# if not ret==0:
	# 	raise Exception('Error with cdo command')
	return new_fp

def recalendar(fp):
	new_fp = fp[:-3] + '_newcal.nc'
	cdo_cmd = ['cdo', 'setcalendar,gregorian', fp, new_fp]
	ret = subprocess.run(cdo_cmd)
	return new_fp

def handle_regrid(fp):
	if Path(fp[:-3] + '_regrid.nc').is_file():
		print(fp[:-3] + '_regrid.nc')
		regrid_fp = fp[:-3] + '_regrid.nc'				
		print('file already regridded')
	else:
		print('regridding file: ',fp)
		regrid_fp = regrid(fp)
		print('regridded ', fp)
	regrid_files.append(regrid_fp)
	print('regrid files: ',regrid_files)
	return regrid_fp

def calc_days_in_month(month,year):
	return monthrange(int(year), int(month))[1]


cal_365_day = ['TaiESM1','CMCC-CM2-VHR4']
cal_greg = ['MPI-ESM1-2-LR','MIROC6']


# define initial variables
model = 'CMCC-CM2-VHR4'
model_short = 'CMCC'
hemisphere = 'NH'
experiment = 'HighResMIP'
scenario = 'ssp585'
resolution = 10
model_cal = '365_day'
g = 'gn'
run = 'r1i1p1f1'
model_offset = -timedelta(hours=6) # for both hist
model_offset = +timedelta(hours=6) # try for ssp585, 0 was still a bit warbly as was -6

# model = 'MPI-ESM1-2-HR'
# model_short = 'MPI'
# hemisphere = 'SH'
# experiment = 'HighResMIP'
# scenario = 'historical'
# resolution = 10
# model_cal = 'noleap'
# model_offset = -timedelta(hours=3)
# run = 'r1i1p1f1'
# seems to only be 4 timepoints strong enough

# model = 'EC-Earth3P-HR'
# model_short = 'EC-Earth'
# hemisphere = 'NH'
# experiment = 'HighresMIP'
# scenario = 'historical'
# resolution = 10
# g = 'gr'
# model_cal = 'proleptic_gregorian'
# model_offset = -timedelta(hours=3) # NH SH hist
# model_offset = -timedelta(hours=3) # NH SH ssp585
# run = 'r1i1p2f1'


if scenario == 'historical':
	mini_scen = 'hist-1950' #instead of hist
	scen_yr_start = '1950'
	scen_yr_end = '2014'

elif scenario == 'ssp585':
	mini_scen = 'highres-future'
	scen_yr_start = '2015'
	scen_yr_end = '2050'

# regrid = False
yrmonths=generate_yrmonths()

# define variables depending on which type of dataset we have

	# rainfall_fps = [f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/{scenario}/pr/pr_Prim6hr_{model}_{mini_scen}-{scen_yr_start}_r1i1p1f1_gn_{yrmonth}-{yrmonth}_regrid.nc' for yrmonth in yrmonths]
if scenario == 'historical':
	s = 'hist-1950'
else:
	s = 'highres-future'
time1 = ''
time2 = ''
# rainfall_dir = f'/bp1/geog-tropical/data/CMIP6/HighResMIP-rain/{model_short}/{model}/{s}/{run}/Prim6hr/pr/gn/latest/pr_Prim6hr_{model}_{mini_scen}-{scen_yr_start}_r1i1p1f1_gn_'
rainfall_dir = f'/bp1/geog-tropical/data/CMIP6/HighResMIP-rain/{model}/pr/{scenario}/pr_3hr_{model}_{mini_scen}-{scen_yr_start}_{run}_{g}_'
rainfall_dir = f'/bp1/geog-tropical/data/CMIP6/HighResMIP-rain/{model}/pr/{scenario}/pr_3hr_{model}_{mini_scen}_{run}_{g}_'
if model =='CMCC-CM2-VHR4':
	rainfall_dir = f'/bp1/geog-tropical/data/CMIP6/HighResMIP-rain/{model}/pr/{scenario}/pr_Prim6hr_{model}_{s}_{run}_{g}_'
tracking_fp = f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/{scenario}/tracks/TC-{hemisphere}_TRACK_{model}_{mini_scen}-{scen_yr_start}_r1i1p1f1_gn_{scen_yr_start}0101-{scen_yr_end}1231.nc'
tracking_fp = f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/tracks/{scenario}/{hemisphere}/TC-{hemisphere}_TRACK_{model}_{mini_scen}-{scen_yr_start}_{run}_gr_{scen_yr_start}0101-{scen_yr_end}1231.nc'
tracking_fp = f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/tracks/{scenario}/{hemisphere}/TC-{hemisphere}_TRACK_{model}_{mini_scen}_{run}_{g}_{scen_yr_start}0101-{scen_yr_end}1231.nc'
print(tracking_fp)
# tracking_fp = f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/tracks/{scenario}/{hemisphere}/TC-{hemisphere}_TRACK_{model}_{mini_scen}_{run}_gr_{scen_yr_start}0101-{scen_yr_end}1231.nc'
# new_fp = f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/{scenario}/storm_rain/'
new_fp = f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/storm_rain/{scenario}/'
tracking_ds = xr.open_dataset(tracking_fp,use_cftime=True)
storm_start = tracking_ds.FIRST_PT
storm_duration = tracking_ds.NUM_PTS #first point?
lats = tracking_ds.lat_sfcWind
lons = tracking_ds.lon_sfcWind
time = tracking_ds.time
track_id = tracking_ds.TRACK_ID #num points?
w_speed = tracking_ds.sfcWind #time?


index = []

all_lats = np.zeros((len(lats),int(1000/resolution)))
all_lons = np.zeros((len(lats),int(1000/resolution)))
all_rain = np.ones((len(lats),int(1000/resolution),int(1000/resolution)))
all_id = pd.DataFrame({'sid':['0']*len(lats),'year':[0]*len(lats),'month':[0]*len(lats),'day':[0]*len(lats),'hour':[0]*len(lats),'centre_lat':[0]*len(lats),'centre_lon':[0]*len(lats)})
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

	# filter out years where we don't have the rainfall file yet 2015
	if scenario =='ssp585':
		r = set(range(2015,2050))
	else:
		r = set(range(1979,2015))
	if not set(storm_year.values) & r:
	# if not set(storm_year.values) & set(range(1979,1981)):
	# if not set(storm_year.values) & set(range(1987,1988)):
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
		
		# # storm_id_step = storm_id[k]
		# if speed < 33.:
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

		month_length = calc_days_in_month(month,year)
		print('tc month is: ',month)
		if (str(year) in ['2016','2020','2024','2028','2032','2036','2040','2044','2048','2052']) and (str(month) == '02'):
			month_length = '28'

		if model == 'MPI-ESM1-2-HR':
			rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}01010556-{storm_year.values[k]}12312356.nc'
		elif model == 'EC-Earth3P-HR':
			rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}01010000-{storm_year.values[k]}12312100.nc'
			if storm_year.values[k] == 2014:
				rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}01010000-{storm_year.values[k]}12311800.nc'
		else:
			rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}{month}010000-{storm_year.values[k]}{month}{month_length}1800.nc'
		print('month', month)
		print('year: ', storm_year.values[k])
		print('rainfall fp',rainfall_fp)

		
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
		rainfall_ds = xr.open_dataset(regrid_rainfall_fp,use_cftime=True)
		print('rainfall time: ',rainfall_ds.time)

		# try the time slicing
		time_1 = cf.datetime(calendar=model_cal,
									year=rain_time.dt.year,
									month=rain_time.dt.month,
									day=rain_time.dt.day,
									hour=rain_time.dt.hour
									)
		offset = (time_1 + model_offset)

		if (offset.month != month) and (model == 'CMCC-CM2-VHR4'):
			new_month = offset.month
			new_month_length = calc_days_in_month(new_month,year)

			
			
			if new_month not in [10,11,12]:
				new_month = f'0{new_month}'
			else:
				new_month = new_month

			if (str(year) in ['2016','2020','2024','2028','2032','2036','2040','2044','2048','2052']) and (str(new_month) == '02'):
				new_month_length = '28'
				print('new month length')

			rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}{new_month}010000-{storm_year.values[k]}{new_month}{new_month_length}1800.nc'

			
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
			print(regrid_rainfall_fp)
			rainfall_ds = xr.open_dataset(regrid_rainfall_fp,use_cftime=True)
			print('rainfall time: ',rainfall_ds.time)
			print('rain time: ',offset)
		rainfall_slice = rainfall_ds.sel(time=offset).pr

		print('rainfall was timesliced successfully')
		print('rainfall time: ',rainfall_slice.time)

		# find centre lat and lon
		centre_lat = storm_lats[k]
		centre_lon = storm_lons[k]
		print('centre_lon: ',centre_lon)
		print('centre_lat: ',centre_lat)
		if centre_lon > 180:
			centre_lon = centre_lon - 360
		
		# convert centre lats and lons to correct format:
		# round_nearest(centre_lat,0.05)
		# round_nearest(centre_lon,0.05)
		try:
			ilon = list(rainfall_slice.lon.values).index(rainfall_slice.sel(lon=round_nearest(centre_lon,0.05)).lon)
			ilat = list(rainfall_slice.lat.values).index(rainfall_slice.sel(lat=round_nearest(centre_lat,0.05)).lat)
		except:
			ilon = list(rainfall_slice.lon.values).index(rainfall_slice.sel(lon=centre_lon, method='nearest').lon)
			ilat = list(rainfall_slice.lat.values).index(rainfall_slice.sel(lat=centre_lat, method='nearest').lat)

		print(rainfall_slice.sel(lon=centre_lon, method='nearest').lon.values)
		print(rainfall_slice.sel(lat=centre_lat, method='nearest').lat.values)
		# print(rainfall_slice.sel(lon=round_nearest(centre_lon,0.05)).lon.values)
		# print(rainfall_slice.sel(lat=round_nearest(centre_lat,0.05)).lat.values)
		
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
			storm_sid = f'TC_{model}_{scenario}_{year}_{hemisphere}_{j}'
			all_id.sid.iloc[storm_track_idx+k] = storm_sid
			all_id.year.iloc[storm_track_idx+k] = storm_year[k]
			all_id.month.iloc[storm_track_idx+k] = storm_month[k]
			all_id.day.iloc[storm_track_idx+k] = storm_day[k]
			all_id.hour.iloc[storm_track_idx+k] = storm_hour[k]
			all_id.centre_lat[storm_track_idx+k] = centre_lat
			all_id.centre_lon[storm_track_idx+k] = centre_lon
			index.append(storm_track_idx+k)
		else:
			print('k3 is:',k)
			all_lats[storm_track_idx+k] = rain_lats
			all_lons[storm_track_idx+k] = rain_lons
			all_rain[storm_track_idx+k] = rain_data
			storm_sid = f'TC_{model}_{scenario}_{year}_{hemisphere}_{j}'

			print('storm sid is: ',storm_sid)
			print('storm track id: ', storm_track_idx)
			print('k',k)
			print(storm_track_idx+k)
			all_id.sid.iloc[storm_track_idx+k] = storm_sid
			all_id.year.iloc[storm_track_idx+k] = storm_year[k]
			all_id.month.iloc[storm_track_idx+k] = storm_month[k]
			all_id.day.iloc[storm_track_idx+k] = storm_day[k]
			all_id.hour.iloc[storm_track_idx+k] = storm_hour[k]
			all_id.centre_lat[storm_track_idx+k] = centre_lat
			all_id.centre_lon[storm_track_idx+k] = centre_lon
			print(all_id)
			index.append(storm_track_idx+k)
			print(storm_sid)



	path = f'/user/home/al18709/work/CMIP6/HighResMIP/{model}/{scenario}/pr/'
	path = f'/bp1/geog-tropical/data/CMIP6/HighResMIP-rain/{model_short}/{model}/{s}/r1i1p1f1/Prim6hr/pr/gn/latest/'
	path = f'/bp1/geog-tropical/data/CMIP6/HighResMIP-rain/{model}/pr/{scenario}/'
	regrid_files = os.listdir(path)
	n_files = 0
	for item in regrid_files:
		if item.endswith("regrid.nc"):
			n_files = n_files+1
	if n_files > 2:
		for i in regrid_files:
			print(i)
			if i.endswith("regrid.nc"):
				print('removing ',path,i)
				os.remove(os.path.join(path, i))
			if i.endswith("gregorian.nc"):
				os.remove(os.path.join(path, i))
			if i.endswith("newcal.nc"):
				os.remove(os.path.join(path, i))


print('number of weak timesteps: ',len(index))
print('removed duplicates: ',set(index))
print(len(set(index)))
print(index)
print('size of rain array: ',all_rain.shape)
print('size of rain array with weak storms omitted: ',np.delete(all_rain,index,axis=0).shape)
print('size of sids: ',all_id.shape)
print(np.sum(all_id!=0))

def exclude_weak(x,exclude):
	return x[~np.isin(np.arange(len(x)), exclude)]



all_rain = all_rain[index,:,:]
all_lats = all_lats[index]
all_lons = all_lons[index]
all_id = all_id.iloc[index]

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

