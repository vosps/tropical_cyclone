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

cal_365_day = ['TaiESM1','CMCC-CM2-VHR4']
cal_greg = ['MPI-ESM1-2-LR','MIROC6']


# define initial variables

model = 'MPI-ESM1-2-LR'
# done
hemisphere = 'SH'
experiment = 'CMIP6' # or 'CMIP6'
scenario = 'ssp585'
resolution = 10
model_cal = 'proleptic_gregorian'
g = 'gn'
model_offset = -timedelta(hours=4,minutes=30) # histrical SH
model_offset = -timedelta(hours=7,minutes=30) # ssp585 NH
model_offset = -timedelta(hours=4,minutes=30) # hist NH trying, also 
model_offset = -timedelta(hours=7,minutes=30) # ssp585 SH
lat_offset = -1
lon_offset = 0
lat_offset = 0 #SH
lon_offset = 0 #SH
yr1 = 2055
yr2 = 2100

model = 'NorESM2-LM'
# ssp585 ready
hemisphere = 'SH'
experiment = 'CMIP6' # or 'CMIP6'
scenario = 'historical'
resolution = 10
model_cal = 'noleap'
model_offset = -timedelta(hours=3) # NH ssp585 yes
model_offset = -timedelta(hours=3) # SH ssp585 dunno
# model_offset = -timedelta(hours=3) # NH hist yes
# model_offset = -timedelta(hours=3) # SH hist dunno
# lat_offset = 1 #SH ssp585
# lon_offset = -3 #SH ssp585
# lat_offset = 0 #NH
# lon_offset = 1 #NH
# lat_offset = 0 #NH
# lon_offset = 0 #NH
lat_offset = 0 
lon_offset = 0 
g = 'gn'
# lat_offset = 0.5 #SH
# lon_offset = -2 #SH 2
yr1 = 2055
yr2 = 2101


# model = 'TaiESM1'
# # ssp585 ready
# hemisphere = 'NH'
# experiment = 'CMIP6' # or 'CMIP6'
# scenario = 'ssp585'
# resolution = 10
# model_cal = 'noleap'
# g = 'gn'
# model_offset = -timedelta(hours=3) #hours try -3 instead of 3
# yr1 = 2055
# yr2 = 2099
# # # 2055-2099

# model = 'MIROC6'
# hemisphere = 'SH'
# experiment = 'CMIP6'
# scenario = 'historical'
# resolution = 10
# g = 'gn'
# model_cal = 'gregorian'
# model_offset = -timedelta(hours=3) #NH
# model_offset = -timedelta(hours=3) #SH
# lat_offset = -1 # SH
# lon_offset = 1 # SH
# yr1 = 2050
# yr2 = 2100

model = 'IPSL-CM6A-LR'
# ssp585 ready
hemisphere = 'SH'
experiment = 'CMIP6'
scenario = 'historical'
g = 'gr'
resolution = 10
model_cal = 'gregorian'
# model_offset = -timedelta(hours=7,minutes=30) # NH hist correct
model_offset = -timedelta(hours=4,minutes=30) # NH ssp585 correct
# model_offset = -timedelta(hours=10,minutes=30) # SH ssp585 yes
model_offset = -timedelta(hours=7,minutes=30) # SH hist 
# NH
# lat_offset = 1
# lon_offset = 2
# SH
# lat_offset = -1
# lon_offset = 2
# try for all
lat_offset = 0 
lon_offset = 0
# lat_offset = 1 # NH ssp585 and hist
# lon_offset = 1.5 # NH ssp585 and hist
yr1 = 2055
yr2 = 2100

model = 'MRI-ESM2-0'
# ssp585 ready
hemisphere = 'NH'
experiment = 'CMIP6'
scenario = 'ssp585'
g = 'gn'
resolution = 10
model_cal = 'proleptic_gregorian'
model_offset = -timedelta(hours=4,minutes=30) # NH ssp585 verified
# model_offset = -timedelta(hours=7,minutes=30) # NH hist verified
# model_offset = -timedelta(hours=4,minutes=30) #SH hist and ssp585
lat_offset = 0
lon_offset = 0
lon_offset = 1 #NH ssp585
yr1 = 2050
yr2 = 2101

# model = 'EC-Earth3'
# # ssp585 ready
# hemisphere = 'NH'
# experiment = 'CMIP6'
# scenario = 'ssp585'
# g = 'gr'
# resolution = 10
# model_cal = 'proleptic_gregorian'
# model_offset = -timedelta(hours=7,minutes=30) #NH hist
# model_offset = -timedelta(hours=4,minutes=30) #NH ssp585 yes
# # model_offset = -timedelta(hours=4,minutes=30) #SH yes
# # lat_offset = 0
# # lon_offset = 1
# lat_offset = 0
# lon_offset = 0
# # lat_offset = 0 # SH ssp585
# # lon_offset = -1 # SH ssp585
# yr1 = 2050
# yr2 = 2100

# model = 'CMCC-ESM2'
# # need to clean ssp585
# hemisphere = 'SH'
# experiment = 'CMIP6'
# scenario = 'historical'
# g = 'gn'
# resolution = 10
# model_cal = '365_day'
# model_offset = -timedelta(hours=7,minutes=30) #NH
# model_offset = -timedelta(hours=7,minutes=30) #SH
# lat_offset = 0
# lon_offset = 0
# yr1 = 2050
# yr2 = 2100

# model = 'BCC-CSM2-MR'
# hemisphere = 'NH'
# experiment = 'CMIP6'
# scenario = 'ssp585'
# g = 'gn'
# resolution = 10
# model_cal = '365_day'
# model_offset = -timedelta(hours=1,minutes=30) #NH
# # model_offset = -timedelta(hours=10,minutes=30) #SH
# lat_offset = 0
# lon_offset = 0
# yr1 = 2050
# yr2 = 2100

# model = 'NESM3'
# # ssp585 ready
# hemisphere = 'NH'
# experiment = 'CMIP6'
# scenario = 'ssp585'
# g = 'gn'
# resolution = 10
# model_cal = 'standard'
# model_offset = -timedelta(hours=4,minutes=30) #NH ssp585
# # model_offset = -timedelta(hours=1,minutes=30) # NH hist yes
# # model_offset = -timedelta(hours=1,minutes=30) #SH ssp585 and hist yes
# lat_offset = 0
# lon_offset = 0
# # lat_offset = -1 # NH ssp585
# # lon_offset = 0.5 # NH ssp585
# yr1 = 2050
# yr2 = 2100


if scenario == 'historical':
	mini_scen = 'hist'
	scen_yr_start = '1950'
	scen_yr_end = '2014'


cftimes = ['TaiESM1','NorESM2-LM','MPI-ESM1-2-LR','MIROC6','IPSL-CM6A-LR','MRI-ESM2-0','EC-Earth3','CMCC-ESM2','BCC-CSM2-MR','NESM3']
latitudes_2 = ['TaiESM1','MRI-ESM2-0']
# lat 2 has more nans?

shift_left = ['MPI-ESM1-2-LR']
shift = ['NorESM2-LM','MPI-ESM1-2-LR','MIROC6','IPSL-CM6A-LR']
bp1store = ['MPI-ESM1-2-LR','IPSL-CM6A-LR','MRI-ESM2-0','EC-Earth3','CMCC-ESM2','BCC-CSM2-MR','NESM3']
bp1store_6 = ['MIROC6','TaiESM1','NorESM2-LM']

# yrmonths=generate_yrmonths()
# rainfall_fps = [f'/user/home/al18709/work/CMIP6/{model}/pr/{scenario}/pr_6hrPlev_{model}_{scenario}_r1i1p1f1_gn_{yrmonth}010300-{yrmonth}312100_regrid.nc' for yrmonth in yrmonths]
if (model == 'MPI-ESM1-2-LR'):
	time1 = '010300'
	time2 = '312100'
elif (model == 'EC-Earth3'):
	time1 = '010130'
	time2 = '312230'
else:
	time1 = '010300'
	time2 = '312100'

new_fp = f'/user/home/al18709/work/CMIP6/{model}/storm_rain/{scenario}/'

rainfall_dir = f'/user/home/al18709/work/CMIP6/{model}/pr/{scenario}/pr_6hrPlev_{model}_{scenario}_r1i1p1f1_gn_'


# count number of storms timesteps to intiate final variables
if scenario == 'historical':
	range_nh = range(1979,2015)
	range_sh = range(1979,2014)
else:
	range_nh = range(yr1,yr2)
	range_sh = range(yr1,yr2-1)

if hemisphere == 'NH':
	tracking_fps = [f'/user/home/al18709/work/CMIP6/{model}/tracks/{scenario}/{hemisphere}/{model}_tracks_r1i1p1f1_{yr}_newtime2.nc' for yr in range_nh]
else:
	tracking_fps = [f'/user/home/al18709/work/CMIP6/{model}/tracks/{scenario}/{hemisphere}/{model}_tracks_r1i1p1f1_{yr}{yr+1}_newtime2.nc' for yr in range_sh]
for i,f in enumerate(tracking_fps):
	if (model == 'BCC-CSM2-MR') and ('19801981' in f) and (hemisphere == 'SH'):
		print('year not in tracking dataset')
		continue
	print(f)
	tracks = xr.open_dataset(f,use_cftime=True)
	if i == 0:
		n_ts = len(tracks.latitude)
	else:
		n_ts = n_ts + len(tracks.latitude)
n_timesteps = n_ts
print(f'number of timesteps in {model} {hemisphere} is {n_timesteps}!')
index = []
all_lats = np.zeros((n_timesteps,int(1000/resolution)))
all_lons = np.zeros((n_timesteps,int(1000/resolution)))
all_rain = np.ones((n_timesteps,int(1000/resolution),int(1000/resolution)))
all_id = pd.DataFrame({'sid':['0']*n_timesteps,'year':[0]*n_timesteps,'month':[0]*n_timesteps,'day':[0]*n_timesteps,'hour':[0]*n_timesteps,'centre_lat':[0]*n_timesteps,'centre_lon':[0]*n_timesteps})
print('processing ',n_timesteps, ' storm samples...')


if scenario == 'historical':
	if hemisphere == 'NH':
		years = range(1979,2015)
	else:
		years = range(1979,2014)
	# years = range(1980,1982)
else:
	if hemisphere == 'NH':
		years = range(yr1,yr2)
	else:
		years = range(yr1,yr2-1)
	# years = range(2055,2056)



for yr_i,yr in enumerate(years):
	if (model == 'BCC-CSM2-MR') and (yr == 1980) and (hemisphere == 'SH'):
		print(yr,'not found in tracking dataset')
		continue
	if (model == 'MRI-ESM2-0') and (yr == 2091):
		print(yr,'corrupted file year')
		continue
	# define number of time points in the year now
	print('year data...')
	print('yr',yr)
	print('yr_i',yr_i)
	if yr_i == 0:
		n_timepoints_yr = 0
	else:
		if (model == 'BCC-CSM2-MR') and (yr == 1981) and (hemisphere == 'SH'):
			n_timepoints_yr = 0
		else:
			n_timepoints_yr = len(lats)

	# load tracking dataset
	print(f'processing {yr}')
	if hemisphere == 'NH':
		tracking_fp = f'/user/home/al18709/work/CMIP6/{model}/tracks/{scenario}/{hemisphere}/{model}_tracks_r1i1p1f1_{yr}_newtime2.nc'
	else:
		tracking_fp = f'/user/home/al18709/work/CMIP6/{model}/tracks/{scenario}/{hemisphere}/{model}_tracks_r1i1p1f1_{yr}{yr+1}_newtime2.nc'

	# if model in cal_365_day:
	# 	track_ds = xr.open_dataset(tracking_fp,use_cftime=True)
	# 	# track_ds = track_ds.convert_calendar("noleap",dim='time')
	# else:
	track_ds = xr.open_dataset(tracking_fp,use_cftime=True)
	
	print(f'number of storms in {yr}: ', len(track_ds.FIRST_PT))
	print(tracking_fp)
	print('tracking time: ',track_ds.time)
	
	# define the storm variables
	storm_start = track_ds.FIRST_PT
	storm_duration = track_ds.NUM_PTS #first point?
	if model in latitudes_2:
		lats = track_ds.latitude_2
		lons = track_ds.longitude_2
	else:
		lats = track_ds.latitude
		lons = track_ds.longitude
	time = track_ds.time
	track_id = track_ds.TRACK_ID #num points?
	w_speed = track_ds.wind_speed_850 #time?
	print('w_speed shape: ', w_speed.shape)
	print('storm start len', storm_start.values.shape)
	# n_timepoints_yr = len(lats)

	# for each start point, reference the lats and lons from the start point for the duration
	storm_track_idx = 0
	# loop through each storm to link up with rainfall
	for j,i in enumerate(storm_start.values):
		print('i is: ',i)
		print('j is: ',j)
		# define npoints as number of timepoints in the storm
		print(storm_duration.values[j])
		# npoints = int(storm_duration.values[j][0])
		npoints = int(storm_duration.values[j])
		
		if j-1 < 0:
			storm_track_idx = storm_track_idx
		else:
			# storm_track_idx = storm_track_idx + int(storm_duration.values[j-1][0])
			storm_track_idx = storm_track_idx + int(storm_duration.values[j-1])
		print('storm_track_idx is: ', storm_track_idx)

		# slice tracking variables by track time and duration to get individual track data
		storm_lats = lats.values[storm_track_idx:storm_track_idx+npoints]
		storm_lons = lons.values[storm_track_idx:storm_track_idx+npoints]
		storm_time = time[storm_track_idx:storm_track_idx+npoints]
		# storm_speed = w_speed.values[storm_track_idx:storm_track_idx+npoints,0]
		storm_speed = w_speed.values[storm_track_idx:storm_track_idx+npoints]
		storm_year = storm_time.dt.year
		storm_month = storm_time.dt.month
		storm_day = storm_time.dt.day
		storm_hour = storm_time.dt.hour
		print(storm_speed.shape)
		print(w_speed.shape)
		print(storm_lats.shape)
		print('storm time',storm_time)

		npoints2 = int(len(storm_year.values))
		print('npoints: ',npoints)
		print('npoints2: ',npoints2)
		if npoints != npoints2:
			npoints = npoints2
		# print(len(storm_duration.values[j]))
		
		regrid_files = []

		# find corresponding rainfall file
		for k in range(npoints):
			print('k is: ', k, '/', npoints)
			# define time variables to find correct filepath and array
			year = storm_year.values[k]
			speed = storm_speed[k]
			
			# if speed < 33.:
			if speed < 41.25: # have to use adjustment term as using 850 hpa winds
				storm_sid = f'{model}_{scenario}_{year}_{hemisphere}_{j}'
				print('storm too weak',storm_sid)
				continue
			
			print('k2 is:',k)
			print('tc strength storm')

			# rain_time: cf.DatetimeNoLeap = storm_time[k].values
			print('what is storm time',storm_time.time.values[0])
			# print('what is storm time',storm_time.values)
			# rain_time = storm_time.time.values[k]
			# rain_time: cf.DatetimeNoLeap = storm_time.values[k]
			rain_time = storm_time.values[k]
			print('rain time track time is: ', rain_time)
			if storm_month.values[k] not in [10,11,12]:
				month = f'0{storm_month.values[k]}'
			else:
				month = storm_month.values[k]
			print(storm_month)

			if model in bp1store:
				# rainfall_dir = f'/user/home/al18709/work/CMIP6/{model}/pr/{scenario}/pr_3hr_{model}_{scenario}_r1i1p1f1_gn_'
				rainfall_dir = f'/bp1/geog-tropical/data/CMIP6/CMIP6-rain/{model}/pr/{scenario}/pr_3hr_{model}_{scenario}_r1i1p1f1_{g}_'
				print(rainfall_dir)
				rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}01{time1}-{storm_year.values[k]}12{time2}.nc'
				if (model_offset > timedelta(hours=0)): # if the offset takes us to the following year
					rainfall_fp_next = f'{rainfall_dir}{storm_year.values[k]+1}01{time1}-{storm_year.values[k]+1}12{time2}.nc'
				else: # if the offset takes us to the year before
					rainfall_fp_next = f'{rainfall_dir}{storm_year.values[k]-1}01{time1}-{storm_year.values[k]-1}12{time2}.nc'
				print(rainfall_fp)
			elif model in bp1store_6:
				rainfall_dir = f'/bp1/geog-tropical/data/CMIP6/CMIP6-rain/{model}/pr/{scenario}/pr_6hrPlev_{model}_{scenario}_r1i1p1f1_{g}_'
				rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}01{time1}-{storm_year.values[k]}12{time2}.nc'
				if (model_offset > timedelta(hours=0)):
					rainfall_fp_next = f'{rainfall_dir}{storm_year.values[k]+1}01{time1}-{storm_year.values[k]+1}12{time2}.nc'
				else:
					rainfall_fp_next = f'{rainfall_dir}{storm_year.values[k]-1}01{time1}-{storm_year.values[k]-1}12{time2}.nc'
			else:
				rainfall_fp = f'{rainfall_dir}{storm_year.values[k]}01{time1}-{storm_year.values[k]}12{time2}.nc'
				if (model_offset > timedelta(hours=0)):
					rainfall_fp_next = f'{rainfall_dir}{storm_year.values[k]+1}01{time1}-{storm_year.values[k]+1}12{time2}.nc'
				else:
					rainfall_fp_next = f'{rainfall_dir}{storm_year.values[k]-1}01{time1}-{storm_year.values[k]-1}12{time2}.nc'

			print('month', month)
			print('year: ', storm_year.values[k])
			print('rainfall fp',rainfall_fp)

			# regrid files into correct grid
			regrid_rainfall_fp = handle_regrid(rainfall_fp)
			# if Path(rainfall_fp[:-3] + '_regrid.nc').is_file():
			# 	print(rainfall_fp[:-3] + '_regrid.nc')
			# 	regrid_rainfall_fp = rainfall_fp[:-3] + '_regrid.nc'				
			# 	print('file already regridded')
			# else:
			# 	print('regridding file: ',rainfall_fp)
			# 	regrid_rainfall_fp = regrid(rainfall_fp)
			# 	print('regridded ', rainfall_fp)
			# regrid_files.append(regrid_rainfall_fp)
			# print('regrid files: ',regrid_files)
			if model in cftimes:
				print(f'opening {regrid_rainfall_fp} with cftime...')
				rainfall_ds = xr.open_dataset(regrid_rainfall_fp,use_cftime=True)
			else:
				rainfall_ds = xr.open_dataset(regrid_rainfall_fp)
			print('rainfall time: ',rainfall_ds.time)
			print('track_time: ',rain_time)

			# try the time slicing
			time_1 = cf.datetime(calendar=model_cal,
									year=rain_time.year,
									month=rain_time.month,
									day=rain_time.day,
									hour=rain_time.hour
									)
			offset = (time_1 + model_offset)
			print('rain time: ',offset)

			try:
				rainfall_slice = rainfall_ds.sel(time=rain_time).pr
				# rainfall_slice = rainfall_ds.sel(time=time_1,method='nearest').pr
			except:
				print('rain time: ',rain_time.year)
				print('offset year: ',offset.year)
				print('looping through year: ',yr)
				# if (rain_time.year != yr) or ((offset.year != rain_time.year) and (offset != None)):
				if rain_time.year == offset.year:
					rainfall_slice = rainfall_ds.sel(time=offset).pr
				elif (rain_time.year != yr) or ((offset.year != rain_time.year) and (offset != None)):
					if hemisphere == 'NH':
						rainfall_slice = rainfall_ds.sel(time=time_1,method='nearest').pr
					else:
						regrid_rainfall_fp_next = handle_regrid(rainfall_fp_next)
						if model in cftimes:
							rainfall_ds = xr.open_dataset(regrid_rainfall_fp_next,use_cftime=True)
						else:
							rainfall_ds = xr.open_dataset(regrid_rainfall_fp_next)
						print('SH rainfall time yr+1',rainfall_ds.time)
						rainfall_slice = rainfall_ds.sel(time=offset).pr
				else:
					rainfall_slice = rainfall_ds.sel(time=offset).pr
				
				# rainfall_slice = rainfall_ds.sel(time=rain_time,method='nearest').pr


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

			# lat_lower_bound = int(ilat-coord_res)
			# lat_upper_bound = int(ilat+coord_res)

			if model in shift:
				lat_lower_bound = int(ilat-coord_res + lat_offset*(100/resolution))
				lat_upper_bound = int(ilat+coord_res + lat_offset*(100/resolution))
				lon_lower_bound = int(ilon-coord_res + lon_offset*(100/resolution))
				lon_upper_bound = int(ilon+coord_res + lon_offset*(100/resolution))
			else:
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
				all_id.sid.iloc[n_timepoints_yr+storm_track_idx+k] = storm_sid
				all_id.year.iloc[n_timepoints_yr+storm_track_idx+k] = storm_year[k]
				all_id.month.iloc[n_timepoints_yr+storm_track_idx+k] = storm_month[k]
				all_id.day.iloc[n_timepoints_yr+storm_track_idx+k] = storm_day[k]
				all_id.hour.iloc[n_timepoints_yr+storm_track_idx+k] = storm_hour[k]
				all_id.centre_lat[n_timepoints_yr+storm_track_idx+k] = centre_lat
				all_id.centre_lon[n_timepoints_yr+storm_track_idx+k] = centre_lon
				index.append(n_timepoints_yr+storm_track_idx+k)
				# all_id[i+k] = storm_sid
			else:
				print('k3 is:',k)
				all_lats[n_timepoints_yr+storm_track_idx+k] = rain_lats
				all_lons[n_timepoints_yr+storm_track_idx+k] = rain_lons
				all_rain[n_timepoints_yr+storm_track_idx+k] = rain_data
				storm_sid = f'TC_{model}_{scenario}_{year}_{hemisphere}_{j}'
				print('storm sid is: ',storm_sid)
				print('n timepoints yr is: ',n_timepoints_yr)
				print('storm_track_idx is: ', storm_track_idx)
				print('k',k)
				print(n_timepoints_yr+storm_track_idx+k)
				all_id.sid.iloc[n_timepoints_yr+storm_track_idx+k] = storm_sid
				all_id.year.iloc[n_timepoints_yr+storm_track_idx+k] = storm_year[k]
				all_id.month.iloc[n_timepoints_yr+storm_track_idx+k] = storm_month[k]
				all_id.day.iloc[n_timepoints_yr+storm_track_idx+k] = storm_day[k]
				all_id.hour.iloc[n_timepoints_yr+storm_track_idx+k] = storm_hour[k]

				all_id.centre_lat[n_timepoints_yr+storm_track_idx+k] = centre_lat
				all_id.centre_lon[n_timepoints_yr+storm_track_idx+k] = centre_lon
				print(all_id)
				index.append(n_timepoints_yr+storm_track_idx+k)
				# index.append(storm_track_idx+k)
				print(storm_sid)

		if model in bp1store:
			path = f'/bp1/geog-tropical/data/CMIP6/CMIP6-rain/{model}/pr/{scenario}/'
		elif model in bp1store_6:
			path = f'/bp1/geog-tropical/data/CMIP6/CMIP6-rain/{model}/pr/{scenario}/'
		else:
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
			if i.endswith("gregorian.nc"):
				os.remove(os.path.join(path, i))
			if i.endswith("newcal.nc"):
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

