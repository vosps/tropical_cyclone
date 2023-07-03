import numpy as np
import pandas as pd

def load_tc_data(set = 'validation',results = 'final'):
	"""
	options for set are:
		validation
		extreme_valid
		train
		test
		extreme_test
	"""
	
	if results == 'final':
		# load data
		real = np.load('/user/home/al18709/work/gan_predictions_20/%s_real-opt_improve.npy' % set)
		# inputs = np.load('/user/home/al18709/work/gan_predictions_20/%s_input-opt_improve.npy' % set)
		inputs = np.load('/user/home/al18709/work/gan_predictions_20/%s_input-opt_5_normal_problem.npy' % set)[:,:,:,0]
		pred_cnn = np.load('/user/home/al18709/work/cnn/unet_valid_2.npy')
		pred_gan = np.load('/user/home/al18709/work/gan_predictions_20/%s_pred-opt_improve.npy' % set)[:,:,:,0]
		pred_vaegan = np.load('/user/home/al18709/work/vaegan_predictions_20/%s_pred-opt_improve.npy' % set)[:,:,:,0]
		pred_gan_ensemble = np.load('/user/home/al18709/work/gan_predictions_20/%s_pred-opt_improve.npy' % set)
		pred_vaegan_ensemble = np.load('/user/home/al18709/work/vaegan_predictions_20/%s_pred-opt_improve.npy' % set)


		if set == 'validation':
			set = 'valid'
		meta = pd.read_csv('/user/work/al18709/tc_data_mswep/%s_meta.csv' % set)

	elif results == 'test':
		# real = np.load('/user/home/al18709/work/tc_data_flipped/%s_y.npy' % set)
		real = np.load('/user/home/al18709/work/gan_predictions_20/%s_real-opt_5_normal_problem.npy' % set)[:,:,:,0]
		inputs = np.load('/user/home/al18709/work/gan_predictions_20/%s_input-opt_5_normal_problem.npy' % set)[:,:,:,0]
		pred_gan = np.load('/user/home/al18709/work/gan_predictions_20/%s_pred-opt_5_normal_problem.npy' % set,mmap_mode='r')[:,:,:,0]
		pred_vaegan = np.load('/user/home/al18709/work/vaegan_predictions_20/%s_pred-opt_7_better_spread-error.npy' % set,mmap_mode='r')[:,:,:,0]
		pred_gan_ensemble = np.load('/user/home/al18709/work/gan_predictions_20/%s_pred-opt_5_normal_problem.npy' % set,mmap_mode='r')
		pred_vaegan_ensemble = np.load('/user/home/al18709/work/vaegan_predictions_20/%s_pred-opt_7_better_spread-error.npy' % set,mmap_mode='r')
		if set == 'validation': 
			set = 'valid'
		meta = pd.read_csv('/user/work/al18709/tc_data_mswep/%s_meta.csv' % set)
		# meta = pd.read_csv('/user/work/al18709/tc_data_flipped/%s_meta.csv' % set)
		pred_cnn = np.load('/user/home/al18709/work/cnn/unet_%s_2.npy' % set)
	elif results == 'era5_corrected':
		if set == 'validation':
			era5 = np.load('/user/home/al18709/work/gan_predictions_20/era5_corrected_pred-opt_5_normal_problem.npy')
			era5_real = np.load('/user/home/al18709/work/gan_predictions_20/era5_corrected_real-opt_5_normal_problem.npy')
			era5_input = np.load('/user/home/al18709/work/gan_predictions_20/era5_corrected_input-opt_5_normal_problem.npy')
			era5_meta = pd.read_csv('/user/home/al18709/work/tc_data_era5_10/valid_meta.csv')
			return era5,era5_real,era5_input,era5_meta
	elif results == 'era5':
		if set == 'validation':
			era5 = np.load('/user/home/al18709/work/gan_predictions_20/era5_pred-opt_5_normal_problem.npy')
			era5_real = np.load('/user/home/al18709/work/gan_predictions_20/era5_real-opt_5_normal_problem.npy')
			era5_input = np.load('/user/home/al18709/work/gan_predictions_20/era5_input-opt_5_normal_problem.npy')
			era5_meta = pd.read_csv('/user/home/al18709/work/tc_data_era5_10/valid_meta.csv')
			return era5,era5_real,era5_input,era5_meta
		else:
			era5 = np.load('/user/home/al18709/work/gan_predictions_20/%s_pred-opt_era5_era-to-mswep_4.npy' % set)
			era5_real = np.load('/user/home/al18709/work/gan_predictions_20/%s_real-opt_era5_era-to-mswep_4.npy' % set)
			era5_input = np.load('/user/home/al18709/work/gan_predictions_20/%s_input-opt_era5_era-to-mswep_4.npy' % set)
			era5_meta = pd.read_csv('/user/home/al18709/work/tc_data_era5/%s_meta.csv' % set)

		mswep = np.load('/user/home/al18709/work/gan_predictions_20/%s_pred-opt_era5_mswep-to-mswep_2.npy' % set)
		mswep_real = np.load('/user/home/al18709/work/gan_predictions_20/%s_real-opt_era5_mswep-to-mswep_2.npy' % set)
		mswep_input = np.load('/user/home/al18709/work/gan_predictions_20/%s_input-opt_era5_mswep-to-mswep_2.npy' % set)
		mswep_real = np.load('/user/home/al18709/work/tc_data_mswep_40/%s_y.npy' % set)
		mswep_input = np.load('/user/home/al18709/work/tc_data_mswep_40/%s_X.npy' % set)
		mswep_meta = pd.read_csv('/user/home/al18709/work/tc_data_mswep_40/%s_meta.csv' % set)
		return era5,era5_real,era5_input,era5_meta,mswep,mswep_real,mswep_input,mswep_meta


	# real_x = np.load('/user/home/al18709/work/gan_predictions_20/extreme_valid_real-opt_improve.npy')
	# inputs_x = np.load('/user/home/al18709/work/gan_predictions_20/extreme_valid_input-opt_improve_7.npy')
	# pred_cnn_x = np.load('/user/home/al18709/work/cnn/unet_valid.npy')
	# pred_gan_x = np.load('/user/home/al18709/work/gan_predictions_20/extreme_valid_pred-opt_improve.npy')[:,:,:,0]
	# pred_vaegan_x = np.load('/user/home/al18709/work/vaegan_predictions_20/extreme_valid_pred-opt_improve.npy')[:,:,:,0]


	# meta_extreme = pd.read_csv('/user/work/al18709/tc_data_mswep/extreme_valid_meta.csv')
	# meta_extreme_test = pd.read_csv('/user/work/al18709/tc_data_mswep/extreme_test_meta.csv')
	# meta_test = pd.read_csv('/user/work/al18709/tc_data_mswep/test_meta.csv')
	# meta_train = pd.read_csv('/user/work/al18709/tc_data_mswep/train_meta.csv')

	return real,inputs,pred_cnn,pred_vaegan,pred_gan,pred_vaegan_ensemble,pred_gan_ensemble,meta


# final data runs
# vaegan - 7_better-spread-error - 3 more residual layers before upsampling, 4 latent variables, problem = 'normal', 100 latent variables
# gan = logs_normal_problem-final (7_normal), 4 noise channels, 3 residual layers before upsampling, problem = 'normal'
# 

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