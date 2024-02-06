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
# import glob
import cftime as cf
import os
warnings.filterwarnings("ignore")
sns.set_style("white")
sns.set_palette(sns.color_palette("Paired"))
sns.set_palette(sns.color_palette("Set2"))

# TODO: check which way the storms are rotating and how this is plotted - if using imshow it won't take into account
# the fact that mswep uses reverse latitude
# TODO: are the accumulated ones being plotted in the right places? like is it in the right order?
				

def accumulated_rain(storm,meta,real,pred_gan,inputs,flip=True):
	# grab mswep coordinate variables
	fp = '/bp1store/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2000342.00.nc'
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
def lookup(row,cal):
	# date = cf.datetime(calendar=cal,
	# 				year=row.year,
	# 				month=row.month,
	# 				day=row.day,
	# 				hour=row.hour
	# 				)
	if row.year not in range(1979,2023):
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

def assign_location_coords(storm_rain,meta,flip=True,ens=True):
	# grab mswep coordinate variables
	meta = meta.reset_index()
	fp = '/bp1store/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2000342.00.nc'
	d = Dataset(fp, 'r')
	lat = d.variables['lat'][:] #lat
	lon = d.variables['lon'][:] #lon
	order = np.array(meta['date']).argsort()
	print(meta)
	print(order)
	meta_sorted = meta.reindex(order)
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

def save_event_set(meta,rain,path,mode,ens=True,critic=False):
	sids = meta.sid
	sids_unique=sids.drop_duplicates()
	tracks_grouped = meta.groupby('sid')

	for sid in sids_unique:
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
		Xspan = np.where((grid_x <= Mlon) & (grid_x >= mlon))[1][[0, -1]]
		Yspan = np.where((grid_y <= Mlat) & (grid_y >= mlat))[0][[0, -1]]

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
			grid_rain[t,sel[1],sel[0]] = storm_rain
	return grid_rain



# load scalar wgan data
# real,inputs,rain,meta = load_tc_data(set='validation',results='ke_tracks')
# meta = pd.read_csv('/user/work/al18709/tc_data_mswep_40/scalar_wgan_valid_meta_with_dates.csv')
# path = '/user/home/al18709/work/event_sets/wgan_scalar/'
# mode = 'validation'
# print(real.shape)
# save_event_set(meta,rain,path,mode)
# save_event_set(meta,real,'/user/home/al18709/work/event_sets/truth/',mode,ens=False)

# # load 2D wgan
# real_2,inputs_2,pred_2,meta_2,imput_og,rain_og,meta_og = load_tc_data(set='validation',results='kh_tracks')
# real_og_x,_,_,_,_,_,rain_og_x,meta_og_x = load_tc_data(set='extreme_valid',results='test')
# meta_og = pd.read_csv('/user/work/al18709/tc_data_mswep_40/original_wgan_valid_meta_with_dates.csv')
# # meta_og_x = pd.read_csv('/user/work/al18709/tc_data_mswep_40/original_wgan_extreme_validation_meta_with_dates.csv')
# path_og = '/user/home/al18709/work/event_sets/wgan/'
# # mode_og_x = 'extreme_validation'
# mode_og = 'validation'
# save_event_set(meta_og,rain_og,path_og,mode_og)
# # save_event_set(meta_og_x,rain_og_x,path_og,mode_og_x)

# load modular wgan part 2 only
# modular_pred_2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_pred-opt_modular_part2_raw.npy')
# pred_mraw = np.load('/user/home/al18709/work/gan_predictions_20/validation_pred-opt_modular_part2_raw.npy')
# disc_pred_mraw = np.load('/user/home/al18709/work/gan_predictions_20/validation_disc_pred-opt_modular_part2_raw.npy')
# meta = pd.read_csv('/user/work/al18709/tc_data_mswep_40/scalar_wgan_valid_meta_with_dates.csv')
# path_og = '/user/home/al18709/work/event_sets/wgan/'
# mode_og = 'extreme_test'
# mode1 = 'validation_modular'
# mode1b = 'validation_modular_critic'
# mode2 = 'validation_mraw'
# mode2b = 'validation_mraw_critic'
# path = '/user/home/al18709/work/event_sets/wgan_modular/'
# # save_event_set(meta,modular_pred_2,mode1)
# save_event_set(meta,pred_mraw,path,mode2)
# save_event_set(meta,disc_pred_mraw,path,mode2b,critic=True)
# # save_event_set(meta_og_x,rain_og_x,path_og,mode_og)

# load modular wgan part 1 and 2
# modular_pred_1_and_2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_pred-opt_modular_part2_raw.npy')
pred_1and2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_validation_pred-opt_scalar_test_run_1_pred-opt_modular_part2_raw.npy')
disc_pred_1and2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_validation_pred-opt_scalar_test_run_1_disc_pred-opt_modular_part2_raw.npy')
meta = pd.read_csv('/user/work/al18709/tc_data_mswep_40/scalar_wgan_valid_meta_with_dates.csv')

mode2 = 'validation_1and2'
mode2b = 'validation_1and2_critic'
path = '/user/home/al18709/work/event_sets/wgan_modular/'
# # save_event_set(meta,modular_pred_2,mode1)
save_event_set(meta,pred_1and2,path,mode2)
save_event_set(meta,disc_pred_1and2,path,mode2b,critic=True)

pred_1and2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_extreme_valid_pred-opt_scalar_test_run_1_pred-opt_modular_part2_raw.npy')
disc_pred_1and2 = np.load('/user/home/al18709/work/gan_predictions_20/modular_part2_lowres_predictions_extreme_valid_pred-opt_scalar_test_run_1_disc_pred-opt_modular_part2_raw.npy')
meta = pd.read_csv('/user/work/al18709/tc_data_mswep_40/scalar_wgan_extreme_valid_meta_with_dates.csv')
mode2 = 'extreme_valid_1and2'
mode2b = 'extreme_valid_1and2_critic'
save_event_set(meta,pred_1and2,path,mode2)
save_event_set(meta,disc_pred_1and2,path,mode2b,critic=True)
# # save_event_set(meta_og_x,rain_og_x,path_og,mode_og)

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