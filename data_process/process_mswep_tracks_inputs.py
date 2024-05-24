"""
Take the HighResMIP data and cross reference with either TRACK or TempestExtreme tracking data

inputs : highresmip rainfall data and tracking
outputs : save TC rainfall images for each TC track
https://esgf-node.llnl.gov/search/cmip6/
https://data.ceda.ac.uk/badc/highresmip-derived/data/storm_tracks/TRACK/EC-Earth-Consortium
"""

import xarray as xr
import numpy as np
from netCDF4 import Dataset
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
	years = range(1979,1981)
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
	# regrid_files.append(regrid_fp)
	# print('regrid files: ',regrid_files)
	return regrid_fp


# define initial variables
model = 'MSWEP'
hemisphere = 'NH'
resolution = 10
model_cal = 'standard'
model_offset = -timedelta(hours=0,minutes=0) #NH
lat_offset = 0
lon_offset = 0
new_fp = f'/user/home/al18709/work/CMIP6/{model}/storm_rain/'
shift = []

# count number of storms timesteps to intiate final variables
# tracking_fp = '/user/home/al18709/work/ibtracks/IBTrACS.ALL.v04r00.nc'
# tracks = xr.open_dataset(tracking_fp)
# open csv file
tracks = pd.read_csv('/user/work/al18709/ibtracks/ibtracs.ALL.list.v04r00.csv',
						usecols=['SID','LAT','LON','BASIN','NAME','SEASON', 'NATURE','ISO_TIME','USA_SSHS'],
						parse_dates = ['ISO_TIME'],keep_default_na=False)

# tidy up columns with multiple dtypes
tracks = tracks.iloc[1: , :]
tracks = tracks.replace(' ', np.nan)
tracks['USA_SSHS'] = pd.to_numeric(tracks['USA_SSHS'])
tracks['SEASON'] = pd.to_numeric(tracks['SEASON'])

# tracks = tracks[tracks['SEASON'] >= 1979]
# tracks = tracks[tracks['SEASON'] <= 2020]
tracks = tracks[tracks['SEASON'] >= 2000]
tracks = tracks[tracks['SEASON'] <= 2014]


# tracks = tracks[tracks['SEASON'] == 2000]
print(tracks)
# tracks = tracks[tracks['ISO_TIME'].dt.dayofyear >= 153]
# tracks = tracks[tracks['ISO_TIME'].dt.dayofyear <= 158]
tracks = tracks[tracks['NATURE'] == 'TS']

# TCs = tracks[tracks['USA_SSHS'] >= 1]['SID']
TCs = tracks[tracks['USA_SSHS'] >= 0]['SID']
TCs = TCs.drop_duplicates()
print('number of TCs: ', len(TCs))
# reference tracks with TCs
tracks = pd.merge(tracks, 
                      TCs, 
                      on ='SID', 
                      how ='inner')
# tracks = tracks[tracks['USA_SSHS'] >= 1]
tracks = tracks[tracks['USA_SSHS'] >= 0]
print(tracks)

# extract datetime data
tracks['ISO_TIME'] = pd.to_datetime(tracks['ISO_TIME'])
year_month_day = list(pd.to_datetime(tracks['ISO_TIME']).dt.strftime('%Y%m%d'))
year_month = list(pd.to_datetime(tracks['ISO_TIME']).dt.strftime('%Y%m'))
# years = list(pd.to_datetime(tracks['ISO_TIME']).dt.strftime('%Y'))
# months = list(pd.to_datetime(tracks['ISO_TIME']).dt.strftime('%m'))
# days = list(pd.to_datetime(tracks['ISO_TIME']).dt.strftime('%d'))
# hours = list(pd.to_datetime(tracks['ISO_TIME']).dt.strftime('%h'))

years = list(tracks['ISO_TIME'].dt.year)
months = list(tracks['ISO_TIME'].dt.month)
days = list(tracks['ISO_TIME'].dt.day)
hours = list(tracks['ISO_TIME'].dt.hour)
print(tracks['ISO_TIME'])
print(hours[0:10])
day_of_year = tracks['ISO_TIME'].dt.dayofyear
for d in list(range(1,10)):

	day_of_year[day_of_year == d] = '00' + str(d)	
for d in list(range(10,100)):
	day_of_year[day_of_year == d] = '0' + str(d)
year_day = [str(y) + str(d) for y,d in zip(list(pd.to_datetime(tracks['ISO_TIME']).dt.year),day_of_year)]


print(tracks.ISO_TIME.attrs)
print(tracks.ISO_TIME)
print('tracks is',tracks)
# itracks = tracks[tracks['NATURE'] == 'TS']
# tcs = (tracks['usa_sshs'] >= 1)
# print('selecting tracks...')
# tracks = tracks.where(tcs,drop=True)
# print(tracks)

n_timesteps = len(tracks.LAT)
lats = list(tracks.LAT)
lons = list(tracks.LON)
print(f'number of timesteps in {model} {hemisphere} is {n_timesteps}!')
index = []
all_lats = np.zeros((n_timesteps,int(1000/resolution)))
all_lons = np.zeros((n_timesteps,int(1000/resolution)))
all_rain = np.ones((n_timesteps,int(1000/resolution),int(1000/resolution)))
all_id = pd.DataFrame({'sid':['0']*n_timesteps,'year':[0]*n_timesteps,'month':[0]*n_timesteps,'day':[0]*n_timesteps,'hour':[0]*n_timesteps,'centre_lat':[0]*n_timesteps,'centre_lon':[0]*n_timesteps})
print('processing ',n_timesteps, ' storm samples...')

# def generate_days(year):
# 	if year in [2004,2008,2012,2016]:
# 		days = ["{0:03}".format(i) for i in range(1,367)]
# 	elif year in [1999]:
# 		days = [500]
# 	elif year == 2000:
# 		days = ["{0:03}".format(i) for i in range(153,267)]
# 	elif year == 2001:
# 		days = [500]
# 	elif year == 2020:
# 		days = ["{0:03}".format(i) for i in range(1,305)]
# 	elif year == 2005:
# 		days = ["{0:03}".format(i) for i in range(191,366)]
# 	elif year == 2002:
# 		days = ["{0:03}".format(i) for i in range(0,100)]
# 	else:
# 		days = ["{0:03}".format(i) for i in range(1,366)]
# 	return days
variables = ['mslp','u850','v850','shear','lat','lon']
print(len(tracks))
print(tracks)
tracks['index'] = range(len(tracks))
tracks = tracks.set_index(tracks['index'])
tracks = tracks.reset_index(drop=True)
print(tracks)
for track_i in range(len(tracks)):
	print('processing tracks: ',track_i)
	if track_i == 22299:
		continue
	track = tracks.loc[track_i]

	year = years[track_i]
	doy = year_day[track_i]
	month = months[track_i]
	day = days[track_i]
	hour = hours[track_i]
	hour = "{:02d}".format(int(hour))


	if str(doy[4:]) == '000':
		continue

	rain_dir = f'/bp1/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/'

	if hour not in ['00','03','06','09','12','15','18','21','24']:
		if hour in ['02','05','08','11','14','17','20','23']:
			hour = int(hour) + 1
			hour = "{:02d}".format(int(hour))
		elif hour in ['01','04','07','10','13','16','19','22']:
			hour = int(hour) - 1
			hour = "{:02d}".format(int(hour))
		if hour == '24':
			hour = '00'
	
	if hour == '24':
		hour == '00'
	rainfall_fp = rain_dir + f'{year}{doy[4:]}.{hour}.nc'
	
	# define the storm variables
	# print('lats',lats)
	centre_lat = lats[track_i]
	centre_lon = lons[track_i]
	time = list(tracks.ISO_TIME)[track_i]
	track_id = list(tracks.SID)[track_i] #num points?
	# w_speed = tracks.WIND_SPEED_850[track_i] #time?
	# print('w_speed shape: ', w_speed.shape)

	# for each start point, reference the lats and lons from the start point for the duration

	# regrid files into correct grid
	# regrid_rainfall_fp = handle_regrid(rainfall_fp)
	# try:
	# print(rainfall_fp)
	if rainfall_fp == '/bp1/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/1986031.21.nc':
		print('fp not found!')
		continue
	if rainfall_fp == '/bp1/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/1988029.21.nc':
		print('fp not found!')
		continue
	try:
		rainfall_ds = Dataset(rainfall_fp, 'r')
	except:
		print(rainfall_ds, 'cannot open/doesnt exist')
		continue
	# except: 
	# 	print('couldnt open dataset')
	# 	print(rainfall_fp)
	# 	continue

	# rainfall_ds = xr.open_dataset(rainfall_fp)

	rain_lat = rainfall_ds.variables['lat'][:] #lat
	rain_lon = rainfall_ds.variables['lon'][:] #lon
	try:
		rain_data = rainfall_ds.variables['precipitation'][0,:,:] #lon
	except:
		print(rainfall_fp,'doesnt have precip as variable?')
		continue

	rain_time = rainfall_ds.variables['time'][:]

	print('centre_lon: ',centre_lon)
	print('centre_lat: ',centre_lat)
	if centre_lon > 180:
		centre_lon = centre_lon - 360
	
	print(rainfall_ds)

	
	rain = xr.DataArray(rainfall_ds, 
	coords={'lat': rain_lat,'lon': rain_lon,'time': rain_time}, 
	dims=["lat", "lon", "time"])
	try:
		ilon = list(rain_lon).index(rain.sel(lon=round_nearest(centre_lon,0.05)).lon)
		ilat = list(rain_lat).index(rain.sel(lat=round_nearest(centre_lat,0.05)).lat)
	except:
		ilon = list(rain_lon).index(rain.sel(lon=centre_lon, method='nearest').lon)
		ilat = list(rain_lat).index(rain.sel(lat=centre_lat, method='nearest').lat)

	# print(rain.sel(lon=centre_lon, method='nearest').lon)
	# print(rain.sel(lat=centre_lat, method='nearest').lat)

	coord_res = 500/resolution

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


	# print('ilon',ilon)
	# print('ilat',ilat)

	# print(lat_lower_bound)
	# print(lat_upper_bound)
	# print(lon_lower_bound)
	# print(lon_upper_bound)


	# if ilon < 100:
	if ilon < 500/resolution:
		diff = int(500/resolution - ilon)
		lon_lower_bound = 3600 - diff
		data1 = rain_data[lat_lower_bound:lat_upper_bound,lon_lower_bound:-1]
		data2 = rain_data[lat_lower_bound:lat_upper_bound,0:lon_upper_bound]
		rain_lats = rain_lat[lat_lower_bound:lat_upper_bound]
		lon1 = rain_lon[lon_lower_bound:-1]
		lon2 = rain_lon[0:lon_upper_bound]
		rain_data = np.concatenate((data1,data2),axis=1)
		rain_lons = np.concatenate((lon1,lon2))

		for var in variables:
			# find var file
			var_path = var
			if var == 'u850':
				var_path = 
			elif var == 'v850':
				var_path = 
			ym = f'{year}{month}'
			filepaths_var = f'/bp1/geog-tropical/data/ERA-5/hour/{var_path}_invertlat/ERA5_{var}_3hourly_1deg_{ym}.nc'

			# find inner core
			# calculate mean of core
			# add value to array
	# elif ilon > 3500:
	elif ilon > 3600 - 500/resolution:
		# print('ilon > 3600 - 500/resolution')
		# diff = 512 - ilon
		diff = int(51 - (3600 - ilon))
		print('diff: ',diff)
		lon_upper_bound = diff
		# print('lon_lower_bound: ',lon_lower_bound)
		# print('lon_upper_bound: ',lon_upper_bound)
		# print('rain shape 1',rain_data.shape)
		data1 = rain_data[lat_lower_bound:lat_upper_bound,lon_lower_bound:-1]
		data2 = rain_data[lat_lower_bound:lat_upper_bound,0:lon_upper_bound]
		# print('data1 shape: ',data1.shape)
		# print('data2 shape: ',data2.shape)
		rain_lats = rain_lat[lat_lower_bound:lat_upper_bound]
		lon1 = rain_lon[lon_lower_bound:-1]
		lon2 = rain_lon[0:lon_upper_bound]	
		rain_data = np.concatenate((data1,data2),axis=1)
		rain_lons = np.concatenate((lon1,lon2))

	else:
		# print('lons in limits')
		rain_lats = rain_lat[lat_lower_bound:lat_upper_bound]
		rain_lons = rain_lon[lon_lower_bound:lon_upper_bound]
		rain_data = rain_data[lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
				
	# change units from kg m-2 s-1 to mm day-1
	# rain_data = rain_data * 86400
	# change units from kg m-2 s-1 to mm 6hr-1, also check if kg m-2 s-1 or kg m-2 6hr-1
	# rain_data = rain_data * 86400/4 
	
	# print('rain data shape',rain_data.shape)
	if rain_data.shape != (int(1000/resolution),int(1000/resolution)):
		print('wrong data shape')
		# print(i,j,k)
		print(rain_data.shape)
		all_lats[track_i] = np.zeros((int(1000/resolution)))
		all_lons[track_i] = np.zeros((int(1000/resolution)))
		all_rain[track_i] = np.zeros((int(1000/resolution),int(1000/resolution)))
		# all_id[i+k] = 0
		# all_id.append('0')
		storm_sid = track_id
		# all_id = all_id.sid.str.replace([track_i],storm_sid)
		# all_id[track_i].sid = storm_sid
		all_id.sid.iloc[track_i] = storm_sid
		all_id.year.iloc[track_i] = year
		all_id.month.iloc[track_i] = month
		all_id.day.iloc[track_i] = day
		all_id.hour.iloc[track_i] = hour
		all_id.centre_lat[track_i] = centre_lat
		all_id.centre_lon[track_i] = centre_lon
		index.append(track_i)
		# all_id[i+k] = storm_sid
	else:
		all_lats[track_i] = rain_lats
		all_lons[track_i] = rain_lons
		all_rain[track_i] = rain_data
		storm_sid = track_id
		# print('storm sid is: ',storm_sid)
		print(track_i)
		all_id.sid.iloc[track_i] = storm_sid
		all_id.year.iloc[track_i] = year
		all_id.month.iloc[track_i] = month
		all_id.day.iloc[track_i] = day
		all_id.hour.iloc[track_i] = hour
		all_id.centre_lat[track_i] = centre_lat
		all_id.centre_lon[track_i] = centre_lon
		# print(all_id)
		index.append(track_i)
		# print(storm_sid)

	path = rain_dir



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


print(all_rain[index].shape)
print(all_lats[index].shape)
print(all_lons[index].shape)
print(all_id.loc[index].shape)
print(all_id.loc[index])
print(new_fp)


# np.save(f'{new_fp}storm_lats_extended_{hemisphere}.npy',all_lats[index])
# np.save(f'{new_fp}storm_lons_extended_{hemisphere}.npy',all_lons[index])
# np.save(f'{new_fp}storm_rain_extended_{hemisphere}.npy',all_rain[index])
# np.save(f'{new_fp}storm_index_extended_{hemisphere}.npy',index)
# # np.save('{new_fp}storm_sid.npy',all_id)
# all_id.loc[index].to_csv(f'{new_fp}storm_sid_extended_{hemisphere}.csv')
# print('files saved!')

# np.save(f'{new_fp}storm_lats_tcs_and_ts.npy',all_lats[index])
# np.save(f'{new_fp}storm_lons_tcs_and_ts.npy',all_lons[index])
np.save(f'{new_fp}storm_inputs_tcs_and_ts.npy',all_rain[index])
# np.save(f'{new_fp}storm_index_tcs_and_ts.npy',index)
# np.save('{new_fp}storm_sid.npy',all_id)
all_id.loc[index].to_csv(f'{new_fp}storm_sid_tcs_and_ts_inputs.csv')
print('files saved!')

