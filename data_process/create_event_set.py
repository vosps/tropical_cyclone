import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import colors
from netCDF4 import Dataset
import pandas as pd
# import properscoring as ps
# import cartopy.feature as cfeature
# import cartopy.crs as ccrs
import warnings
import xarray as xr
from utils.evaluation import find_landfalling_tcs,tc_region,create_xarray,get_storm_coords
from utils.data import load_tc_data
from utils.plot import make_cmap
# import xesmf as xe
import numpy.ma as ma
# import glob
import cftime as cf
import os
from multiprocessing import Pool, cpu_count, set_start_method
warnings.filterwarnings("ignore")
sns.set_style("white")
sns.set_palette(sns.color_palette("Paired"))
sns.set_palette(sns.color_palette("Set2"))
from global_land_mask import globe

# TODO: check which way the storms are rotating and how this is plotted - if using imshow it won't take into account
# the fact that mswep uses reverse latitude
# TODO: are the accumulated ones being plotted in the right places? like is it in the right order?
				

def accumulated_rain(storm,meta,real,pred_gan,inputs,flip=True):
	# grab mswep coordinate variables
	fp = '/bp1/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2000342.00.nc'
	d = Dataset(fp, 'r')
	lat = d.variables['lat'][:] #lat
	lon = d.variables['lon'][:] #lon
	print('lat shape: ',lat.shape)
	print('lon shape: ',lon.shape)
	# calculate lats and lons for storm
	lats,lons = tc_region(meta,storm,lat,lon)
	# initialise accumulated xarray
	# grid_x, grid_y = np.meshgrid(lats, lons)
	grid_x, grid_y = np.meshgrid(lons,lats)
	# a = np.zeros((grid_x.shape))
	print('grid_x shape: ',grid_x.shape)
	print('grid_y.shape: ', grid_y.shape)
	print('lons shape: ',lons.shape)
	print('lats shape: ',lats.shape)
	a = np.zeros((grid_y.shape))
	print('a shape',a.shape)
	accumulated_ds = create_xarray(lats,lons,a)
	accumulated_ds_pred = create_xarray(lats,lons,a)
	# accumulated_ds_input = create_xarray(lats,lons,a)
	# loop through storm time steps o generate accumulated rainfall
	for i in storm:
		storm_lats,storm_lons = get_storm_coords(lat,lon,meta,i)
		ds = create_xarray(storm_lats,storm_lons,real[i])
		ds_pred = create_xarray(storm_lats,storm_lons,pred_gan[i])
		input_lats,input_lons = get_storm_coords(np.arange(-89.5,90,1),np.arange(-179.5,180),meta,i)
		# ds_input = create_xarray(input_lats,input_lons,inputs[i])

		# if flip==True:
		# 	ds.precipitation.values = np.flip(ds.precipitation.values,axis=0)
		# 	ds_pred.precipitation.values = np.flip(ds_pred.precipitation.values,axis=0)

		# regrid so grids match
		regridder = xe.Regridder(ds, accumulated_ds, "bilinear")
		ds_out = regridder(ds)
		ds_pred_out = regridder(ds_pred)

		# regird the inputs
		# regridder = xe.Regridder(ds_input, accumulated_ds, "bilinear")
		# ds_input_out = regridder(ds_input)

		# add up rainfall
		accumulated_ds = accumulated_ds + ds_out
		accumulated_ds_pred = accumulated_ds_pred + ds_pred_out
		# accumulated_ds_input = accumulated_ds_input + ds_input_out

	return accumulated_ds,accumulated_ds_pred


# def tc_subregion(meta,sid_i,lat,lon,era5=False):
# 	"""
# 	find region which contains all points/ranfinall data of tc track
# 			inputs
# 					meta : csv with metadata
# 					sid_i : list of sid indices
# 					lat : mswep grid
# 					lon : mswep grid
# 	"""

# 	storm_lats = np.zeros(len(sid_i),100)
# 	storm_lons = np.zeros(len(sid_i),100)
# 	j = 0
# 	for i in sid_i:
# 		lat_lower_bound = np.abs(lat-meta['centre_lat'][i]+5.).argmin()
# 		lat_upper_bound = np.abs(lat-meta['centre_lat'][i]-5.).argmin()
# 		lon_lower_bound = np.abs(lon-meta['centre_lon'][i]+5.).argmin()
# 		lon_upper_bound = np.abs(lon-meta['centre_lon'][i]-5.).argmin()
# 		lats = lat[lat_lower_bound:lat_upper_bound]
# 		lons = lon[lon_lower_bound:lon_upper_bound]
# 		storm_lats[j,:,:] = lats
# 		storm_lons[j,:,:] = lons
# 		j = j+1

# 	return storm_lats,storm_lons
def lookup(row,cal,scenario):
	# date = cf.datetime(calendar=cal,
	# 				year=row.year,
	# 				month=row.month,
	# 				day=row.day,
	# 				hour=row.hour
	# 				)
	if scenario == 'hist':
		if row.year not in range(1979,2023):
			return 0	
	else:
		if row.year not in range(2069,2100):
			return 0	
		
	if (row.month == 2) and (row.day == 29):
		return 0
	elif (row.month == 2) and (row.day == 30):
		return 0
	else:
		date = cf.datetime(calendar=cal,
						year=row.year,
						month=row.month,
						day=row.day,
						hour = row.hour,
						)

	# date = pd.to_datetime('year' : row.year, 'month' : row.month, 'day' = row.day)
	return date

def create_xarray_2(lats,lons,data,ensemble=None):
        if ensemble==None:
                accumulated_ds = xr.Dataset(
                        data_vars=dict(
                                precipitation=(["y", "x"], data)),  
                        coords=dict(
                                lon=("x", lons),
                                lat=("y", lats),
                        ))
        else:
                
                accumulated_ds = xr.Dataset(
                        data_vars=dict(
                                precipitation=(["y","x","ens"], data)),                        
                        coords=dict(
                                lon=("x", lons),
                                lat=("y", lats),
                                member=("ens", ensemble)
                        )
                )

        return accumulated_ds

def assign_location_coords(storm_rain,storm_meta,flip=True,ens=True):
	# grab mswep coordinate variables
	meta_2 = storm_meta.reset_index()
	fp = '/bp1/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2000342.00.nc'
	d = Dataset(fp, 'r')
	lat = d.variables['lat'][:] #lat
	lon = d.variables['lon'][:] #lon
	order = np.array(meta_2['date']).argsort()
	print(meta_2)
	print(order)
	meta_sorted = meta_2.reindex(order)
	print(meta_sorted)
	storm_rain_sorted = storm_rain[order,:,:,:]
	time = meta_sorted.date
	print('time',time)
	print(time.shape)
	time = list(time)
	x = np.arange(0,100)
	y = np.arange(0,100)
	ensemble = np.arange(0,20)
	if ens == False:
		ensemble = np.arange(0,1)
	dims = ['time','y','x','ens']
	dims2 = ['time','y','x']
	if ens==True:
		ds = xr.Dataset(
						data_vars=dict(
								precipitation=(dims, np.zeros((len(time), 100, 100, 20))),
								storm_lats=(dims2, np.zeros((len(time), 100, 100))),
								storm_lons=(dims2, np.zeros((len(time), 100, 100)))
								),                        
						coords=dict(
								time=('time',time),
								lat=("y", y),
								lon=("x", x),
								member=("ens", ensemble)
						)
				)
	else:
		ds = xr.Dataset(
					data_vars=dict(
							precipitation=(dims, np.zeros((len(time), 100, 100, 1))),
							storm_lats=(dims2, np.zeros((len(time), 100, 100))),
							storm_lons=(dims2, np.zeros((len(time), 100, 100)))
							),                        
					coords=dict(
							time=('time',time),
							lat=("y", y),
							lon=("x", x),
							member=("ens", ensemble)
					)
			)
	
	for i,t in enumerate(time):
		print(i)
		print('t is: ',t)
		print(len(time))
		print(storm_rain_sorted.shape)
		storm_lats,storm_lons = get_storm_coords(lat,lon,meta_sorted,i)
		grid_lons, grid_lats = np.meshgrid(storm_lons,storm_lats)
		print('shapes 1')
		print(storm_lats.shape)
		print(storm_lons.shape)
		lats_lon,lats_lat = grid_lats.shape
		lons_lon,lons_lat = grid_lons.shape
		print('shapes 2')
		print(grid_lats.shape)
		print(grid_lons.shape)
		
		if lats_lon < 100:
			print('here')
			d = 100 - lats_lon
			new_array1 = np.zeros((100, 100))
			new_array2 = np.zeros((100, 100))
			new_array1[:lats_lon, :] = grid_lats
			new_array2[:lats_lon, :] = grid_lons
			new_array1[lats_lon:,:] = grid_lats[-d,:]
			new_array2[lats_lon:,:] = grid_lons[-d,:]
			grid_lats = new_array1
			grid_lons = new_array2
			print(grid_lats.shape)
			print(grid_lons.shape)

		if lats_lon > 100:
			grid_lats = grid_lats[:100,:]
		if lats_lat > 100:
			grid_lats = grid_lats[:,:100]
		# if lats_lat < 100:
		# 	diff = 100 - lats_lon
		# 	grid_lats = grid_lats[,:]
		
		if lons_lon > 100:
			grid_lons = grid_lons[:100,:]
		if lons_lat > 100:
			grid_lons = grid_lons[:,:100]


		# print(storm_lats.shape)
		# print(storm_lons.shape)
		# TODO: make storm lats and lons into correct shape (100,100) grids
		data = xr.Dataset(
					data_vars=dict(
							precipitation=(['y','x','ens'], storm_rain_sorted[i]),
							storm_lats=(['y','x'], grid_lats),
							storm_lons=(['y','x'], grid_lons)
							),                        
					coords=dict(
							lat=("y", y),
							lon=("x", x),
							member=("ens", ensemble)
					)
			)
		ds.loc[dict(time=t)] = data
	return ds

def save_event_set(meta,rain,path,mode,scenario,ens=True,critic=False):
	sids = meta.sid
	sids_unique=sids.drop_duplicates()
	tracks_grouped = meta.groupby('sid')
	print('tracks_grouped:',tracks_grouped)

	for sid in sids_unique:
		print('sid: ',sid)
		storm = tracks_grouped.get_group(sid)
		if critic == False:
			storm_rain = rain[storm.index,:,:,:]
			storm_ds = assign_location_coords(storm_rain,storm,flip=True,ens=ens)
			storm_ds.to_netcdf(f'{path}{mode}_{sid}.nc',format='NETCDF4')
		else:
			storm_scores = rain[storm.index,:]
			np.save(f'{path}{mode}_{sid}.npy',storm_scores)
		# exit()

def globalise_storm_rain(storm,prediction=True):

	# define initial variables
	ntime,_,_,_ = storm.precipitation.shape

	# define global extent
	fp = '/bp1/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2000342.00.nc'
	d = Dataset(fp, 'r')
	lat = d.variables['lat'][:] #lat
	lon = d.variables['lon'][:] #lon
	grid_x, grid_y = np.meshgrid(lon,lat)
	# print(grid_x.shape)

	# superimpose rain onto global grid
	if prediction == True:
		grid_rain = np.zeros((ntime,grid_x.shape[0],grid_y.shape[1],20))
	else:
		grid_rain = np.zeros((ntime,grid_x.shape[0],grid_y.shape[1]))

	for t in range(ntime):
		storm_lons = storm.storm_lons[t,:,:]
		storm_lats = storm.storm_lats[t,:,:]

		Mlon = storm_lons[-1,-1]
		mlon = storm_lons[0,0]
		Mlat = storm_lats[-1,-1]
		mlat = storm_lats[0,0]
		if mlon.values > Mlon.values:
			continue
		# print(Mlon.values,mlon.values)
		# print(Mlat.values,mlat.values)
		if Mlon.values < mlon.values:
			print('longitude not monotonically increasing')
			# print(storm_lons)
		# print(grid_x)
		# print(grid_y)
		# print(Mlon,mlon)
		print(np.where((grid_x <= Mlon.values) & (grid_x >= mlon.values)))
		Xspan = np.where((grid_x <= Mlon.values) & (grid_x >= mlon.values))[1][[0, -1]]
		Yspan = np.where((grid_y <= Mlat.values) & (grid_y >= mlat.values))[0][[0, -1]] #adding in .values here to see if that fixes things

		# Create a selection
		sel = [slice(Xspan[0], Xspan[1] + 1), slice(Yspan[0], Yspan[1] + 1)]

		if prediction == True:
			for i in range(20):
				storm_rain = storm.precipitation[t,:,:,i]
				grid_rain[t,sel[1],sel[0],i] = storm_rain
		else:
			storm_rain = storm.precipitation[t,:,:,0]
			print(grid_rain[t,sel[1],sel[0]].shape)
			print(storm_rain.shape)
			if grid_rain[t,sel[1],sel[0]] != (100,100):
				continue
			grid_rain[t,sel[1],sel[0]] = storm_rain
	return grid_rain

def extremise_storm_rain(storm,prediction=True):
	ntime,_,_,_ = storm.precipitation.shape

	# define global extent
	fp = '/bp1/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2000342.00.nc'
	d = Dataset(fp, 'r')
	lat = d.variables['lat'][:] #lat
	lon = d.variables['lon'][:] #lon
	
	grid_x, grid_y = np.meshgrid(lon,lat)
	z = globe.is_land(grid_y,grid_x)

	# superimpose rain onto global grid
	if prediction == True:
		grid_rain = np.zeros((ntime,grid_x.shape[0],grid_y.shape[1],20))
	else:
		grid_rain = np.zeros((ntime,grid_x.shape[0],grid_y.shape[1]))
		land_maxs = np.zeros((ntime))

	for t in range(ntime):
		storm_lons = storm.storm_lons[t,:,:]
		storm_lats = storm.storm_lats[t,:,:]

		Mlon = storm_lons[-1,-1]
		mlon = storm_lons[0,0]
		Mlat = storm_lats[-1,-1]
		mlat = storm_lats[0,0]
		if mlon.values > Mlon.values:
			continue

		if Mlon.values < mlon.values:
			print('longitude not monotonically increasing')

		Xspan = np.where((grid_x <= Mlon.values) & (grid_x >= mlon.values))[1][[0, -1]]
		Yspan = np.where((grid_y <= Mlat.values) & (grid_y >= mlat.values))[0][[0, -1]] #adding in .values here to see if that fixes things

		# Create a selection
		sel = [slice(Xspan[0], Xspan[1] + 1), slice(Yspan[0], Yspan[1] + 1)]

		if prediction == True:
			for i in range(20):
				storm_rain = storm.precipitation[t,:,:,i]
				
				grid_rain[t,sel[1],sel[0],i] = storm_rain
		else:
			storm_rain = storm.precipitation[t,:,:,0]
			# print('max storm rain',np.max(storm_rain.values))

			if grid_rain[t,sel[1],sel[0]].shape != (100,100):
				print("wrong grid size")
				continue

			grid_rain[t,sel[1],sel[0]] = storm_rain
		
	
	
	# does any grid square get above threshold?
	grid_rain_hrly = grid_rain / 3
	l_rain = (grid_rain_hrly * z).flatten()
	land_rain = l_rain[l_rain > 0]
	grid_rain_max = np.max(grid_rain_hrly,axis=0)
	# print('land_max_is:', land_rain.shape)
	# how many times?
	grid_rain_threshold_count = (grid_rain_hrly > 16.2).sum(axis=0)

	return grid_rain_max,grid_rain_threshold_count,land_rain

def find_dates(sids,model_cal,scenario):

	dates = sids.apply(lambda row : cf.datetime(calendar=model_cal,
									year=row.year,
									month=row.month,
									day=row.day,
									hour=row.hour
									), axis=1)
	return dates

def process_storm(storm_filename):
	"""Process a single storm file."""
	if '.nc' in storm_filename:
		print(storm_filename)
		storm = xr.open_dataset(tc_dir + storm_filename)
		# print('storm 0',np.sum(storm.precipitation))
		storm_rain,threshold_count,storm_maxs = extremise_storm_rain(storm, prediction=False) # in units mm/hour
		# filter for extreme rain greater than 50mm/hour
		# print('storm_rain1',np.sum(storm_rain))
		# storm_rain[storm_rain <= 16.2] = 0
		# print('storm_rain2',np.sum(storm_rain))
		# print('storm threshold count',np.sum(threshold_count))
		
		return storm_rain,threshold_count,storm_maxs
	
def process_file(args):
	"""Process a single file and return the result."""
	file,tc_index = args
	# global global_extreme_rain
	result = process_storm(file)
	# print(result)
	if result is not None:
		rain,count,maxs = result
		# if result is not None:
		# print('assigning data to storm ',tc_index)
		file_dir = f'/user/home/al18709/work/event_sets/processing/{model}_{scenario}/'

		# global_extreme_rain[tc_index, :, :] = result
		np.save(f'{file_dir}{tc_index}_rain.npy',rain)
		np.save(f'{file_dir}{tc_index}_count.npy',count)
		np.save(f'{file_dir}{tc_index}_max.npy',maxs)
		print('assigning data to storm ',tc_index)
		# print('rain',np.sum(rain))
		# print('count',np.sum(count))
	else:
		print(f'result is None, index = {tc_index}')
	

# print('chips')

# ['canesm','cnrm6','ecearth6','ipsl6','miroc6','mpi6','mri6','ukmo']
# 'canesm'#'cnrm6'#'ecearth6'#'ipsl6'#'miroc6'#'ukmo'#'mpi6' #'mri6'
# miroc6 calendar is not standard
# ipsl6 calendar is not standard
# canesm calendar is not standard
# https://loca.ucsd.edu/loca-calendar/
# ukmo, mri6, ecearth6, miroc6,cnrm6,canesm,ipsl6,mpi6
model = 'mpi6'
cal = 'standard'
# cal = 'proleptic_gregorian'
print(model)
scenario = 'ssp245'
print(scenario)

# # 1. globalise storms
# if scenario == 'hist':
# 	yr1 = 2000
# 	yr2 = 2014
# else:
# 	yr1 = 2085
# 	yr2 = 2099
# print(scenario)
# data = np.load(f'/user/home/al18709/work/ke_track_rain/hr/{model}_{scenario}_pred_qm.npy')
# meta = pd.read_csv(f'/user/home/al18709/work/ke_track_inputs/{model}_{scenario}_tracks.csv')
# meta.lon[meta.lon > 180] = meta.lon[meta.lon > 180] - 360
# meta2 = pd.DataFrame({'sid':meta.sid ,'centre_lat':meta.lat, 'centre_lon':meta.lon, 'hour':meta.hour, 'day':meta.day,'month':meta.month, 'year':meta.year})
# condition1 = (meta2.year >= yr1) & (meta2.year <= yr2)
# rain = data[condition1]
# meta3 = meta2[condition1].reset_index()
# # rain2 = rain[meta3.year <= yr2]
# # meta4 = meta3[meta3.year <= yr2].reset_index()
# condition2 = (meta3.month !=2 ) & (meta3.day != 30)
# rain_cond = rain[condition2]
# meta_cond = meta3[condition2].reset_index()
# meta5 = pd.DataFrame({'sid':meta_cond.sid ,'centre_lat':meta_cond.centre_lat, 'centre_lon':meta_cond.centre_lon,'date':find_dates(meta_cond,cal,scenario)})
# path = f'/user/home/al18709/work/event_sets/wgan_{model}_{scenario}/'
# if os.path.exists(path):
# 	print(f'{path} exists!')
# else:
# 	os.makedirs(path)
# mode = 'chips'
# save_event_set(meta5,rain_cond,path,mode,ens=False)



# 2. new threshold
tc_dir = f'/user/home/al18709/work/event_sets/wgan_{model}_{scenario}/'
# tc_files = np.sort(os.listdir(tc_dir))[:]
# tc_files = tc_files['.nc' in tc_files]
tc_files1 = np.array(np.sort(os.listdir(tc_dir))[:])
tc_files2 = ['.nc' in file for file in tc_files1]
tc_files = tc_files1[tc_files2][:]

# Initialize global variables
nstorms = sum('.nc' in file for file in tc_files)
print('number of storms:',nstorms)

# get ready to save a bunch of storm files
file_dir = f'/user/home/al18709/work/event_sets/processing/{model}_{scenario}/'
if os.path.exists(file_dir):
	print(f'{file_dir} exists!')
	# empty old storms
	# files = os.listdir(file_dir)
	# for f in files:
	# 	os.remove(f'{file_dir}{f}')
	# remove any files we already did
	processed_files = os.listdir(file_dir)
	import re
	regex = r"(.+?)_rain.npy"
	remove = []
	for f in processed_files:
		# print(f)
		match = re.match(regex, f)
		# print(match)
		if match:
			file_number = int(match.group(1))
			remove.append(file_number)
else:
	os.makedirs(file_dir)
	remove = []


file_indices = [(file, idx) for idx, file in enumerate(tc_files)]
file_indices = [val for idx, val in enumerate(file_indices) if idx not in remove]
print('this many storms still to do:',len(file_indices))

# split up based on number of samples we select
import sys
a = int(sys.argv[1])
b = int(sys.argv[2])
if b == 2000:
	file_indices = file_indices[a:]
else:
	file_indices = file_indices[a:b]
print('this many storms still to do in this script:',len(file_indices))

# Use multiprocessing to parallelize the processing
if __name__ == '__main__':
	set_start_method("fork", force=True)
	pool = Pool(6)
	# split = np.array_split(file_indices, 24)
	pool.map(process_file, file_indices)
	pool.close()
	pool.join()
print('results pooled!')
# exit()
# exit()
# go through rain arrays and calculate mean
# import gc
# gc.collect()
global_extreme_rain = np.zeros((nstorms, 1800, 3600))
global_extreme_count = np.zeros((nstorms, 1800, 3600))
files = os.listdir(file_dir)
max_rains = []
for i,_ in enumerate(tc_files):
	print(i)
	
	try:
		rain = np.load(f'{file_dir}{i}_rain.npy')
	except:
		print(f'{file_dir}{i}_rain.npy')
		continue
	try:
		rain_count = np.load(f'{file_dir}{i}_count.npy')
	except:
		print(f'{file_dir}{i}_count.npy')
		continue
	try:
		storm_maxs = np.load(f'{file_dir}{i}_max.npy')
	except:
		continue
	global_extreme_rain[i,:,:] = rain
	global_extreme_count[i,:,:] = rain_count
	max_rains.append([m for m in storm_maxs])

print(global_extreme_rain)
global_mean_rain = np.max(global_extreme_rain,axis=0)
global_count = np.sum(global_extreme_count,axis=0)
print(np.sum(global_mean_rain))
np.save(f'/user/home/al18709/work/event_sets/wgan_{model}_{scenario}/{model}_{scenario}_mean_extreme_global.npy',global_mean_rain)
np.save(f'/user/home/al18709/work/event_sets/wgan_{model}_{scenario}/{model}_{scenario}_count_extreme_global.npy',global_count)
np.save(f'/user/home/al18709/work/event_sets/wgan_{model}_{scenario}/{model}_{scenario}_max_rain_over_land.npy',storm_maxs)
print('datasets saved!')
print(f'/user/home/al18709/work/event_sets/wgan_{model}_{scenario}/{model}_{scenario}_count_extreme_global.npy')
print(f'/user/home/al18709/work/event_sets/wgan_{model}_{scenario}/{model}_{scenario}_mean_extreme_global.npy')
print(f'/user/home/al18709/work/event_sets/wgan_{model}_{scenario}/{model}_{scenario}_max_rain_over_land.npy')


exit()
# # 3. accumulated threshold
# global_rain = np.zeros((1800,3600))
# for storm_filename in tc_files:
# 	if ('.nc' in storm_filename):
# 		print(storm_filename)
# 		storm = xr.open_dataset(tc_dir + storm_filename)
# 		storm_rain = globalise_storm_rain(storm,prediction=False)
# 		global_rain = global_rain + np.sum(storm_rain,axis=0)

# np.save(f'/user/home/al18709/work/event_sets/wgan_{model}_{scenario}/{model}_{scenario}_accumulated_global.npy',global_rain)



# print('mswep')
# mswep_rain = np.expand_dims(np.load('/user/home/al18709/work/CMIP6/MSWEP/storm_rain/storm_rain_tcs_and_ts.npy'),axis=-1)
# mswep_lats = np.load('/user/home/al18709/work/CMIP6/MSWEP/storm_rain/storm_lats_tcs_and_ts.npy')
# mswep_lons = np.load('/user/home/al18709/work/CMIP6/MSWEP/storm_rain/storm_lons_tcs_and_ts.npy')
# print(mswep_lats.shape)
# print(mswep_lons.shape)
# meta = pd.read_csv('/user/home/al18709/work/CMIP6/MSWEP/storm_rain/storm_sid_tcs_and_ts.csv')
# mswep_meta = pd.DataFrame({'sid':meta.sid ,'centre_lat':meta.centre_lat, 'centre_lon':meta.centre_lon,'date':find_dates(meta,'standard')})
# path = '/user/home/al18709/work/event_sets/MSWEP/'
# mode = 'mswep'
# save_event_set(mswep_meta,mswep_rain,path,mode,ens=False)

# 2
model = 'mswep'
scenario = 'obs'

# get ready to save a bunch of storm files
file_dir = f'/user/home/al18709/work/event_sets/processing/{model}_{scenario}/'
if os.path.exists(file_dir):
	print(f'{file_dir} exists!')
	# empty old storms
	files = os.listdir(file_dir)
	for f in files:
		os.remove(f'{file_dir}{f}')
else:
	os.makedirs(file_dir)


tc_dir = '/user/home/al18709/work/event_sets/MSWEP/'
tc_files1 = np.array(np.sort(os.listdir(tc_dir))[:])
tc_files2 = ['.nc' in file for file in tc_files1]
tc_files = tc_files1[tc_files2][:]

# Initialize global variables
nstorms = sum('.nc' in file for file in tc_files)
print('number of storms:',nstorms)
global_extreme_rain = np.zeros((nstorms, 1800, 3600))
global_extreme_count = np.zeros((nstorms, 1800, 3600))

# get ready to save a bunch of storm files
file_dir = f'/user/home/al18709/work/event_sets/processing/{model}_{scenario}/'
if os.path.exists(file_dir):
	print(f'{file_dir} exists!')
	# empty old storms
	files = os.listdir(file_dir)
	for f in files:
		os.remove(f'{file_dir}{f}')
else:
	os.makedirs(file_dir)

file_indices = [(file, idx) for idx, file in enumerate(tc_files)]

# Use multiprocessing to parallelize the processing
if __name__ == '__main__':
	set_start_method("fork", force=True)
	pool = Pool(24)
	# split = np.array_split(file_indices, 24)
	pool.map(process_file, file_indices)
	pool.close()
	pool.join()
print('results pooled!')

# go through rain arrays and calculate mean
files = os.listdir(file_dir)
max_rains = []
for i,_ in enumerate(tc_files):
	rain = np.load(f'{file_dir}{i}_rain.npy')
	rain_count = np.load(f'{file_dir}{i}_count.npy')
	storm_maxs = np.load(f'{file_dir}{i}_max.npy')
	global_extreme_rain[i,:,:] = rain
	global_extreme_count[i,:,:] = rain_count
	max_rains.append([m for m in storm_maxs])

global_mean_rain = np.max(global_extreme_rain,axis=0)
global_count = np.sum(global_extreme_count,axis=0)
print(np.sum(global_mean_rain))
np.save(f'/user/home/al18709/work/event_sets/{model}_{scenario}/{model}_{scenario}_mean_extreme_global.npy',global_mean_rain)
np.save(f'/user/home/al18709/work/event_sets/{model}_{scenario}/{model}_{scenario}_count_extreme_global.npy',global_count)
np.save(f'/user/home/al18709/work/event_sets/{model}_{scenario}/{model}_{scenario}_max_rain_over_land.npy',storm_maxs)
print('datasets saved!')
print(f'/user/home/al18709/work/event_sets/{model}_{scenario}/{model}_{scenario}_count_extreme_global.npy')
print(f'/user/home/al18709/work/event_sets/{model}_{scenario}/{model}_{scenario}_mean_extreme_global.npy')
print(f'/user/home/al18709/work/event_sets/{model}_{scenario}/{model}_{scenario}_max_rain_over_land.npy')


# # 3
# tc_dir = '/user/home/al18709/work/event_sets/MSWEP/'
# tc_files = os.listdir(tc_dir)
# global_rain = np.zeros((1800,3600))
# for storm_filename in tc_files:
# 	if ('.nc' in storm_filename):
# 		print(storm_filename)
# 		storm = xr.open_dataset(tc_dir + storm_filename)
# 		storm_rain = globalise_storm_rain(storm,prediction=False)
# 		global_rain = global_rain + np.sum(storm_rain,axis=0)
# np.save('/user/home/al18709/work/event_sets/MSWEP/accumulated_global.npy',global_rain)

# TODO: check longitude goes between -180 and 180 not 0-360.
exit()

# # load scalar wgan data
# real,inputs,rain,meta = load_tc_data(set='test',results='ke_tracks')
# meta = pd.read_csv('/user/work/al18709/tc_data_mswep_40/scalar_wgan_test_meta_with_dates.csv')
# path = '/user/home/al18709/work/event_sets/wgan_scalar/'
# mode = 'test'
# # # print(real.shape)
# save_event_set(meta,rain,path,mode)
# exit()
# save_event_set(meta,real,'/user/home/al18709/work/event_sets/truth/',mode,ens=False)

# real,inputs,rain,meta = load_tc_data(set='extreme_valid',results='ke_tracks')
# meta = pd.read_csv('/user/work/al18709/tc_data_mswep_40/scalar_wgan_extreme_valid_meta_with_dates.csv')
# # path = '/user/home/al18709/work/event_sets/wgan_scalar/'
# mode = 'extreme_valid'
# # print(real.shape)
# # save_event_set(meta,rain,path,mode)
# save_event_set(meta,real,'/user/home/al18709/work/event_sets/truth/',mode,ens=False)
# issues with test set, might need to rerun for all.


# # # load 2D wgan
# # real_2,inputs_2,pred_2,meta_2,imput_og,rain_og,meta_og = load_tc_data(set='test',results='kh_tracks')
# real_og,_,_,_,_,_,rain_og,meta_og = load_tc_data(set='test',results='test')
# # real_og_x,_,_,_,_,_,rain_og_x,meta_og_x = load_tc_data(set='extreme_valid',results='test')
# meta_og = pd.read_csv('/user/work/al18709/tc_data_mswep_40/original_wgan_test_meta_with_dates.csv')
# # # meta_og_x = pd.read_csv('/user/work/al18709/tc_data_mswep_40/original_wgan_extreme_validation_meta_with_dates.csv')
# path_og = '/user/home/al18709/work/event_sets/wgan/'
# # # mode_og_x = 'extreme_validation'
# mode_og = 'test'
# save_event_set(meta_og,rain_og,path_og,mode_og)
# # # save_event_set(meta_og_x,rain_og_x,path_og,mode_og_x)
# exit()

# # load modular wgan part 2 only
# # modular_pred_2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_pred-opt_modular_part2_raw.npy')
# pred_mraw = np.load('/user/home/al18709/work/gan_predictions_20/test_pred-opt_modular_part2_raw.npy')
# disc_pred_mraw = np.load('/user/home/al18709/work/gan_predictions_20/test_disc_pred-opt_modular_part2_raw.npy')
# meta = pd.read_csv('/user/work/al18709/tc_data_mswep_40/modular_wgan_test_meta_with_dates.csv')
# # path_og = '/user/home/al18709/work/event_sets/wgan/'
# # mode_og = 'extreme_test'
# # mode1 = 'test_modular'
# # mode1b = 'test_modular_critic'
# mode2 = 'test_mraw'
# mode2b = 'test_mraw_critic'
# path = '/user/home/al18709/work/event_sets/wgan_modular/'
# # save_event_set(meta,modular_pred_2,mode1)
# save_event_set(meta,pred_mraw,path,mode2)
# save_event_set(meta,disc_pred_mraw,path,mode2b,critic=True)
# # save_event_set(meta_og_x,rain_og_x,path_og,mode_og)

# pred_mraw = np.load('/user/home/al18709/work/gan_predictions_20/extreme_test_pred-opt_modular_part2_raw.npy')
# disc_pred_mraw = np.load('/user/home/al18709/work/gan_predictions_20/extreme_test_disc_pred-opt_modular_part2_raw.npy')
# meta = pd.read_csv('/user/work/al18709/tc_data_mswep_40/modular_wgan_extreme_test_meta_with_dates.csv')
# mode2 = 'extreme_test_mraw'
# mode2b = 'extreme_test_mraw_critic'
# path = '/user/home/al18709/work/event_sets/wgan_modular/'
# save_event_set(meta,pred_mraw,path,mode2)
# save_event_set(meta,disc_pred_mraw,path,mode2b,critic=True)


# # load modular wgan part 1 and 2
# # modular_pred_1_and_2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_pred-opt_modular_part2_raw.npy')
# pred_1and2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_test_pred-opt_scalar_test_run_1_pred-opt_modular_part2_raw.npy')
# disc_pred_1and2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_test_pred-opt_scalar_test_run_1_disc_pred-opt_modular_part2_raw.npy')
# meta = pd.read_csv('/user/work/al18709/tc_data_mswep_40/modular_wgan_test_meta_with_dates.csv')

# mode2 = 'test_1and2'
# mode2b = 'test_1and2_critic'
# path = '/user/home/al18709/work/event_sets/wgan_modular/'
# # # save_event_set(meta,modular_pred_2,mode1)
# save_event_set(meta,pred_1and2,path,mode2)
# save_event_set(meta,disc_pred_1and2,path,mode2b,critic=True)

# pred_1and2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_extreme_test_pred-opt_scalar_test_run_1_pred-opt_modular_part2_raw.npy')
# disc_pred_1and2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_extreme_test_pred-opt_scalar_test_run_1_disc_pred-opt_modular_part2_raw.npy')
# meta = pd.read_csv('/user/work/al18709/tc_data_mswep_40/modular_wgan_extreme_test_meta_with_dates.csv')
# mode2 = 'extreme_test_1and2'
# mode2b = 'extreme_test_1and2_critic'
# save_event_set(meta,pred_1and2,path,mode2)
# save_event_set(meta,disc_pred_1and2,path,mode2b,critic=True)
# # # save_event_set(meta_og_x,rain_og_x,path_og,mode_og)

# path = '/user/home/al18709/work/event_sets/wgan_modular/'
# pred_1and2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_validation_pred-opt_scalar_test_run_1_best_2_pred-opt_modular_part2_raw.npy')
# disc_pred_1and2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_validation_pred-opt_scalar_test_run_1_best_2_disc_pred-opt_modular_part2_raw.npy')
# meta = pd.read_csv('/user/work/al18709/tc_data_mswep_40/modular_wgan_valid_meta_with_dates.csv')
# mode2 = 'validation_1and2'
# mode2b = 'validation_1and2_critic'
# save_event_set(meta,pred_1and2,path,mode2)
# save_event_set(meta,disc_pred_1and2,path,mode2b,critic=True)

# exit()

# path = '/user/home/al18709/work/event_sets/patchgan/'
# pred_1and2 = np.load('/user/home/al18709/work/gan_predictions_20/validation_pred-opt_modular_part2_patchloss_raw_4.npy')
# disc_pred_1and2 = np.load('/user/home/al18709/work/gan_predictions_20/validation_disc_pred-opt_modular_part2_patchloss_raw_4.npy')
# meta = pd.read_csv('/user/work/al18709/tc_data_mswep_40/modular_wgan_valid_meta_with_dates.csv')
# mode2 = 'patchgan'
# mode2b = 'patchgan_critic'
# save_event_set(meta,pred_1and2,path,mode2)
# save_event_set(meta,disc_pred_1and2,path,mode2b,critic=True)
# exit()

# # save global rain data
# tc_dir = '/user/home/al18709/work/event_sets/wgan_modular/'
# # storm_filename = 'validation_mraw_\*.nc'
# tc_files = os.listdir(tc_dir)
# global_rain = np.zeros((1,1800,3600,20))
# for storm_filename in tc_files:
# 	if ('validation_mraw' in storm_filename) & ('.nc' in storm_filename):
# 		print(storm_filename)
# 		storm = xr.open_dataset(tc_dir + storm_filename)
# 		storm_rain = globalise_storm_rain(storm)
# 		global_rain = global_rain + np.sum(storm_rain,axis=0)
# np.save('/user/home/al18709/work/event_sets/wgan_modular/validation_mraw_global.npy',global_rain)

# tc_dir = '/user/home/al18709/work/event_sets/wgan/'
# tc_files = os.listdir(tc_dir)
# global_rain = np.zeros((1,1800,3600,20))
# for storm_filename in tc_files:
# 	if ('validation' in storm_filename) & ('.nc' in storm_filename):
# 		print(storm_filename)
# 		storm = xr.open_dataset(tc_dir + storm_filename)
# 		storm_rain = globalise_storm_rain(storm)
# 		global_rain = global_rain + np.sum(storm_rain,axis=0)
# np.save('/user/home/al18709/work/event_sets/wgan/validation_global.npy',global_rain)


# tc_dir = '/user/home/al18709/work/event_sets/truth/'
# tc_files = os.listdir(tc_dir)
# global_rain = np.zeros((1800,3600))
# for storm_filename in tc_files:
# 	if ('validation' in storm_filename) & ('.nc' in storm_filename):
# 		print(storm_filename)
# 		storm = xr.open_dataset(tc_dir + storm_filename)
# 		storm_rain = globalise_storm_rain(storm,prediction=False)
# 		global_rain = global_rain + np.sum(storm_rain,axis=0)
# np.save('/user/home/al18709/work/event_sets/truth/validation_global.npy',global_rain)