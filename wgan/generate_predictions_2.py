
from ast import Continue
import numpy as np
import pandas as pd
from tfrecords_generator_ifs import create_fixed_dataset
import setupmodel
import tensorflow as tf
from noise import NoiseGenerator
import gc
import resource
import subprocess


def get_memory_usage():
    """Get memory usage of current process."""
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Convert from kilobytes to megabytes
	# Get GPU memory usage
    try:
        gpu_mem_info = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
        gpu_mem_usage_mb = sum(map(int, gpu_mem_info.decode('utf-8').strip().split('\n')))
    except Exception as e:
        print("Failed to get GPU memory usage:", e)
        gpu_mem_usage_mb = None

    return mem_usage / 1024, gpu_mem_usage_mb
    # return mem_usage / 1024


def flip(tc):
		tc_flipped = np.flip(tc,axis=0)
		return tc_flipped

def find_and_flip(data,meta):
	print(data.shape)
	print(meta)

	sh_indices = meta[meta['centre_lat'] < 0].index
	nh_indices = meta[meta['centre_lat'] > 0].index
	print(np.sum(meta['centre_lat'] < 0))
	print(np.sum(meta['centre_lat'] > 0))

	# flip the nh tcs so they are rotating anticlockwise (in the raw data array)
	# in mswep they can be plotted correctly with the mswep lats and lons, but the raw data shows them flipped as mswep has descending latitudes
	for i in nh_indices:
		X_flipped = flip(data[i])
		data[i] = X_flipped

	
	
	return data


def generate_predictions(*,
					mode,
					data_mode,
					storm,
					checkpoint,
					arch,
					log_folder, 
					# weights_dir,
					# model_numbers=None,
					# problem_type='normal',
					filters_gen=None,
					filters_disc=None,
					padding=None,
					noise_channels=None,
					latent_variables=None,
					predict_year=2019,
					predict_full_image=True,
					# ensemble_members=100,
					gcm = False,
					# plot_ecpoint=True,
					):
		
	# define initial variables
	# input_channels = 1
	# input_channels = 7
	input_channels = 1
	noise_channels = 6 #4
	batch_size = 512
	num_images = 150
	j_ = 0
		

	# initialise model
	print(mode)
	model = setupmodel.setup_model(mode=mode,
								   arch=arch,
								   input_channels=input_channels,
								   filters_gen=filters_gen,
								   filters_disc=filters_disc,
								   padding=padding,
								   noise_channels=noise_channels,
								   latent_variables=latent_variables)

	print('generating predictions...')

	
	if data_mode == 'validation':
		# num_images,_,_ = np.load('/user/work/al18709/tc_data_flipped/valid_X.npy').shape
		num_images,_,_,_ = np.load('/user/work/al18709/tc_data_flipped/valid_combined_X.npy').shape
	elif data_mode == 'validation_weighted':
		# num_images,_,_ = np.load('/user/work/al18709/tc_data_flipped/valid_X.npy').shape
		num_images,_,_,_ = np.load('/user/work/al18709/tc_data_flipped/valid_combined_X.npy').shape
	elif data_mode == 'storm':
		num_images,_,_ = np.load('/user/work/al18709/tc_data_mswep_extend_flipped/y_%s.npy' % storm).shape
	elif (data_mode == 'storm_era5') or (data_mode == 'storm_era5_corrected'):
		num_images,_,_ = np.load('/user/work/al18709/tc_data_era5_flipped_10/y_%s.npy' % storm).shape
	elif (data_mode == 'era5') or (data_mode == 'era5_corrected'):
		num_images,_,_ = np.load('/user/home/al18709/work/tc_data_era5_flipped_10/valid_y.npy').shape
	elif 'scalar' in data_mode:
		print('data mode: ',data_mode)
		num_images,_,_,_ = np.load('/user/home/al18709/work/gan_predictions_20/%s.npy' % data_mode).shape
	elif "event_set" in data_mode:
		input_string = data_mode
        # Find the index of the first and second underscores
		first_underscore_index = input_string.find("_")
		second_underscore_index = input_string.find("_", first_underscore_index + 1)
        # Check if both underscores are found
		if first_underscore_index != -1 and second_underscore_index != -1:
            # Extract the two groups
			model_ = input_string[:first_underscore_index]
			scenario = input_string[first_underscore_index + 1:second_underscore_index]
		if model_ == 'mswep':
			num_images,_,_,_ = np.float32(np.load(f'/user/home/al18709/work/ke_track_rain/lr/{model_}_{scenario}_all_tcs.npy')).shape 
		else:
			num_images,_,_,_ = np.float32(np.load(f'/user/home/al18709/work/ke_track_rain/lr/{model_}_{scenario}_qm.npy')).shape 
		# num_images,_,_,_ = np.float32(np.load(f'/user/home/al18709/work/ke_track_rain/lr/{model_}_{scenario}_pred.npy'))[:40000,:,:,:].shape 
		
	else:
		# num_images,_,_ = np.load('/user/work/al18709/tc_data_flipped/%s_X.npy' % data_mode).shape
		num_images,_,_,_ = np.load('/user/work/al18709/tc_data_flipped/%s_combined_X.npy' % data_mode).shape

	
	print('number of images: ',num_images)

	if gcm == True:
		batch_size = 1
		num_images = 5
	
	if data_mode == 'storm':
		batch_size = num_images
		
	# load relevant data
	# load relevant data
	if "event_set" not in data_mode:
		data_predict = create_fixed_dataset(predict_year,
											batch_size=batch_size,
											downsample=False,
											mode = data_mode,
											storm=storm)
	else:
		print('opening event set in chunks...')
		# cpu_memory_usage_mb,gpu_memory_usage_mb = get_memory_usage()
		# print(f"CPU memory usage: {cpu_memory_usage_mb:.2f} MB")
		# print(f"GPU memory usage: {gpu_memory_usage_mb:.2f} MB")
		x,z,y = create_fixed_dataset(predict_year,
											batch_size=batch_size,
											downsample=False,
											mode = data_mode,
											storm=storm)
		print('shapes are:')
		print(x.shape)
		print(y.shape)
		print(z.shape)
		# cpu_memory_usage_mb,gpu_memory_usage_mb = get_memory_usage()
		# print(f"CPU memory usage: {cpu_memory_usage_mb:.2f} MB")
		# print(f"GPU memory usage: {gpu_memory_usage_mb:.2f} MB")
	# data_predict = create_fixed_dataset(predict_year,
	# 									batch_size=batch_size,
	# 									downsample=False,
	# 									mode = data_mode,
	# 									storm=storm)

		
	# load model weights from main file
	if mode == 'VAEGAN':
		vaegan = True
	elif mode == 'GAN':
		vaegan = False

	# load disc model and make predictions
	disc_weights_file = log_folder + '/models-disc_weights.h5'
	# disc_weights_file = log_folder + '/models/disc_weights-22988800.h5' #patchgan v3 best
	model.disc.built = True
	model.disc.load_weights(disc_weights_file)
	print(disc_weights_file)
	#  load gen model and make predictions
	print('log folder is:',log_folder)
	gen_weights_file = log_folder + '/models-gen_weights.h5'
	# gen_weights_file = log_folder + '/models/gen_weights-22988800.h5' #patchgan v3 best
	print(gen_weights_file)
	# gen_weights_file = log_folder + 'models/gen_weights-9984000.h5' #patchgan v2 best so far?
	gen_weights_file = '/user/home/al18709/work/gan/logs_wgan_modular_v7/models/gen_weights-19814400.h5' #best use this
	# version 7 7552000 best PSD so far
	# 9600000 no
	# 9420800 better - same as mwgan 1 and 2
	# 9497600 quite close to the wgan part 2 but tails off a tiny bit at the end.
	# 9472000 not much better
	# 9190400 not lunch better
	# 9523200 kind of like but not as good 9497600 
	# 9574400 good but not best
	# 9548800 awful
	# 9984000 best so far, is it though?
	# 19148800 no different
	# 19814400 pretty good
	# 20147200 disc no
	# 18892800 disc no try again
	# 18918400 disc no
	# 18790400 no change
	# 16460800

	# patchgan v3/4
	# 18892800 no
	# 18944000 same as above
	# 19097600 similar
	# 9830400 same
	# 0076800 rubbish
	# 20224000 kind of better
	# 21120000 better direction
	# 21196800 slightly worse than 21120000
	# 21299200 good
	# 21324800 best so far
	# 22988800 best

	# gen_weights_file = '/user/home/al18709/work/gan/logs_wgan_modular_patchloss_v1/models/gen_weights-0358400.h5'
	# gen_weights_file = log_folder + '/models-gen_opt_weights.h5' # TODO: this has different construction to gen_weights - ask andrew and lucy
	model.gen.built = True
	model.gen.load_weights(gen_weights_file)
	# cpu_memory_usage_mb,gpu_memory_usage_mb = get_memory_usage()
	# print(f"CPU memory usage: {cpu_memory_usage_mb:.2f} MB")
	# print(f"GPU memory usage: {gpu_memory_usage_mb:.2f} MB")

	# define initial variables
	if "event_set" not in data_mode:
		n_ensembles = 20
	else:
		n_ensembles = 1
	pred = np.zeros((num_images,100,100,n_ensembles))
	seq_real = np.zeros((num_images,100,100,1))
	disc_pred = np.zeros((num_images,1,n_ensembles))
	# disc_real = np.zeros((num_images,1))
	# low_res_inputs = np.zeros((num_images,10,10,1))
	low_res_inputs = np.zeros((num_images,10,10,7))
	# disc_inputs = np.zeros((num_images,100,100,20))
	# cpu_memory_usage_mb,gpu_memory_usage_mb = get_memory_usage()
	# print(f"CPU memory usage: {cpu_memory_usage_mb:.2f} MB")
	# print(f"GPU memory usage: {gpu_memory_usage_mb:.2f} MB")
	if "event_set" not in data_mode:
		data_pred_iter = iter(data_predict)
	else:
		n_b = int(num_images/batch_size)
		print('num images: ',num_images)
		print('batch size: ',batch_size)
		print('n_b: ',n_b)
		n_b_10 = int(n_b/10) * num_images
		# data_predict = tf.data.Dataset.from_tensor_slices((x[:n_b_3], z[:n_b_3], y[:n_b_3]))
		# data_pred_iter = iter(data_predict)
		# x_iter = iter(x)
		# z_iter = iter(z)
		# y_iter = iter(y)

	# data_pred_iter = iter(data_predict)
	
	# unbatch first
	nbatches = int(num_images/batch_size)
	remainder = num_images - nbatches*batch_size

	print(nbatches)	
	# loop through batches
	if nbatches == 1:
		loop = 1
	else:
		loop = nbatches+1
	for i in range(loop):
	# for i in range(nbatches):
		
		print('running batch ',i,'...')
		print(f'Memory usage for batch {i}: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
		# inputs, outputs = next(data_pred_iter)
		# inputs, topography, outputs = next(data_pred_iter)

		if "event_set" not in data_mode:
			inputs, topography, outputs = next(data_pred_iter)
		else:	
			number_of_loaded_batches = 10
			number_of_loaded_batches = 8 #TODO: something not adding up with the remainders
			# for i in range 0, 100, 200, 300, load a new batch of images in
			list = [n * int(n_b/number_of_loaded_batches) for n in range(number_of_loaded_batches)]
			list_diff = list[1] - list[0]
			if i in [n * int(n_b/number_of_loaded_batches) for n in range(number_of_loaded_batches)]:
			# if i in [n * int(n_b/number_of_loaded_batches) for n in range(number_of_loaded_batches+1)]:
				print('batch ', i)
				print('n_b is:',n_b)
				print([n * int(n_b/number_of_loaded_batches) for n in range(number_of_loaded_batches+1)])
				print('j_:',j_)
				first_image_i = i * batch_size 
				second_image_i = (i+list_diff) * batch_size
				print(first_image_i,second_image_i)
				print('number in loaded batch',second_image_i - first_image_i)
				print('Memory usage before assigning x_: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
				x_ = x[first_image_i:second_image_i,:,:,:]
				print('Memory usage before assigning z_: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
				z_ = z[first_image_i:second_image_i,:,:,:]
				print('Memory usage before assigning y_: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
				y_ = y[first_image_i:second_image_i,:,:,:]
				# y_ = y[first_image_i:second_image_i,:,:]
				print('shapes: ',x_.shape,y_.shape,z_.shape)
				print('Memory usage before assigning ds: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
				ds = tf.data.Dataset.from_tensor_slices((x_, z_, y_))
				print('Memory usage before assigning data_predict: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
				data_predict = ds.batch(batch_size)
				print('Memory usage before assigning data_predict_iter: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
				data_pred_iter = iter(data_predict)
				print('Memory usage after assigning data_predict_iter: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
				
				# inputs,topography,outputs = next(data_pred_iter)
				j_ = j_+1
				# print('remainder batch starts on i =',int(np.floor(n_b / (number_of_loaded_batches))))
				print('remainder batch starts on i =',int(number_of_loaded_batches * np.floor(n_b / (number_of_loaded_batches))))
				print(batch_size * np.floor(n_b / (number_of_loaded_batches)))
				
			elif i == int(number_of_loaded_batches * np.floor(n_b / (number_of_loaded_batches))):
				# i == batch_size * np.floor(n_b / (number_of_loaded_batches)): # remainder
				print('last batch ', i)
				print('n_b is:',n_b)
				# print([n * int(n_b/number_of_loaded_batches) for n in range(number_of_loaded_batches+1)])
				first_image_i = i * batch_size 
				# first_image_i = j_*int(num_images/number_of_loaded_batches)
				# second_image_i = (j_+1)*int(num_images/number_of_loaded_batches)
				print(first_image_i)
				# print('number in loaded batch',second_image_i - first_image_i)
				
				x_ = x[first_image_i:,:,:,:]
				z_ = z[first_image_i:,:,:,:]
				y_ = y[first_image_i:,:,:,:]
				print('remainder shapes: ',x_.shape,y_.shape,z_.shape)
				ds = tf.data.Dataset.from_tensor_slices((x_, z_, y_))
				data_predict = ds.batch(batch_size)
				data_pred_iter = iter(data_predict)
				# inputs,topography,outputs = next(data_pred_iter)
				j_ = j_+1
				
			# inputs,topography,outputs = next(data_pred_iter)
			print('Memory usage before doing next data_predict_iter: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
			inputs,topography,_ = next(data_pred_iter)
			print('Memory usage after doing next data_predict_iter: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
			
			outputs = tf.zeros_like(topography)


		if (data_mode == 'era5') or (data_mode == 'era5_corrected') or (data_mode == 'storm_era5') or (data_mode == 'storm_era5_corrected') or ('event_set' in data_mode):
			if i == batch_size:
				n = remainder
			else:
				n = batch_size
			outputs = np.zeros((n,100,100,1))
		# if i !=nbatches:
		# 	continue
		print('remainder',remainder)
		print('number of batches',nbatches)
		print(num_images)

		img_real = outputs
		img_pred = []	   
		noise_shape = inputs[0,...,0].shape + (noise_channels,)
		print('noise shape: ',noise_shape)
		print('inputs shape: ',inputs.shape)
		print('topography shape: ',topography.shape)
		if i == nbatches:
			noise_gen = NoiseGenerator(noise_shape, batch_size=remainder) # does noise gen need to be outside of the for loop?
			img_pred = np.zeros((remainder,100,100,n_ensembles))
			disc_img_pred = np.zeros((remainder,1,n_ensembles))
		else:
			noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size) # does noise gen need to be outside of the for loop?
			img_pred = np.zeros((batch_size,100,100,n_ensembles))
			disc_img_pred = np.zeros((batch_size,1,n_ensembles))

		for j in range(n_ensembles): #do 50 ensemble members
				
			if vaegan:
				noise_shape = np.array(inputs)[0, ..., 0].shape + (latent_variables,)
				if i == nbatches: #can remove if statement as redundant? maybe
					noise_gen = NoiseGenerator(noise_shape, batch_size=remainder)
				else:
					noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
				
				mean, logvar = model.gen.encoder([inputs])
				print('inputs shape: ',inputs.shape)
				print('mean shape: ',mean.shape)
				print('logvar shape: ',logvar.shape)
				pred_single = np.array(model.gen.decoder.predict([mean, logvar, noise_gen()]))[:,:,:,0]
			
			else:
				nn = noise_gen()
				# pred_single = np.array(model.gen.predict([inputs,nn]))[:,:,:,0] # this one
				pred_single = np.array(model.gen.predict([inputs,topography,nn]))[:,:,:,0]
				pred_single_disc = np.array(model.disc.predict([inputs,topography,pred_single]))
				print('pred single disc shape',pred_single_disc.shape)
				# [generator_input, const_input, generator_output]

				# pred_single = np.array(model.gen.predict_on_batch([inputs,nn]))[:,:,:,0]
				gc.collect()
				
			# print(pred_single.shape)
			if i == nbatches:
				img_pred[:remainder,:,:,j] = pred_single
				disc_img_pred[:remainder,:,j] = pred_single_disc
			else:
				img_pred[:,:,:,j] = pred_single
				disc_img_pred[:,:,j] = pred_single_disc
			# print(img_pred.shape)
			

		print('img pred shape: ',img_pred.shape)
		print('img real shape: ',img_real.shape)
		print('seq_real.shape: ',seq_real.shape)
		print('disc_pred shape', disc_pred.shape)
		print('assigning images ',i*batch_size,' to ',i*batch_size + batch_size,'...')
		if i == nbatches:
			print(i,nbatches)
			print('match')
			# seq_real[i*batch_size:,:,:,:] = img_real[:remainder]
			if "event_set" not in data_mode:
				seq_real[i*batch_size:,:,:,0] = img_real[:remainder]
				low_res_inputs[i*batch_size:,:,:,:] = inputs[:remainder]
			# else:
				# seq_real[i*batch_size:,:,:,:] = img_real[:remainder]
			pred[i*batch_size:,:,:,:] = img_pred[:remainder]
			disc_pred[i*batch_size:,:,:] = disc_img_pred[:remainder]

			gc.collect()
			
		else:
			# seq_real[i*batch_size:i*batch_size + batch_size,:,:,:] = img_real
			if "event_set" not in data_mode:
				seq_real[i*batch_size:i*batch_size + batch_size,:,:,0] = img_real
				low_res_inputs[i*batch_size:i*batch_size + batch_size,:,:,:] = inputs
			# else:
				# seq_real[i*batch_size:i*batch_size + batch_size,:,:,:] = img_real
				# seq_real[i*batch_size:i*batch_size + batch_size,:,:,:] = img_real
			pred[i*batch_size:i*batch_size + batch_size,:,:,:] = img_pred
			disc_pred[i*batch_size:i*batch_size + batch_size,:,:] = disc_img_pred
			
			gc.collect()

			
	# TODO: transfer to cpu memory not gpu memory
	if 'event_set' in data_mode:
		tf.keras.backend.clear_session()

	print(mode)
	print(data_mode)
	print(seq_real.shape)
	print(pred.shape)
	print(low_res_inputs.shape)

	# print(seq_real)
	flip = True

	if flip == True:
		if data_mode == 'validation':
			meta = pd.read_csv('/user/work/al18709/tc_data_mswep/valid_meta.csv')
		elif data_mode == 'validation_weighted':
			meta = pd.read_csv('/user/work/al18709/tc_data_mswep/valid_meta.csv')
		elif data_mode == 'storm':
			meta = pd.read_csv('/user/work/al18709/tc_data_mswep_extend_flipped/meta_%s.csv' % storm)
		elif data_mode == 'storm_era5':
			meta = pd.read_csv('/user/work/al18709/tc_data_era5_flipped_10/meta_%s.csv' % storm)
		elif data_mode == 'storm_era5_corrected':
			meta = pd.read_csv('/user/work/al18709/tc_data_era5_flipped_10/meta_%s.csv' % storm)
		elif (data_mode == 'era5') or (data_mode == 'era5_corrected'):
			meta = pd.read_csv('/user/work/al18709/tc_data_era5_10/valid_meta.csv')
		elif 'scalar' in data_mode:
			if 'test' in data_mode:
				if 'extreme' in data_mode:
					meta = pd.read_csv('/user/work/al18709/tc_data_mswep/extreme_test_meta.csv')
				else:
					meta = pd.read_csv('/user/work/al18709/tc_data_mswep/test_meta.csv')
			elif 'extreme' in data_mode:
				meta = pd.read_csv('/user/work/al18709/tc_data_mswep/extreme_valid_meta.csv')
			else:
				meta = pd.read_csv('/user/work/al18709/tc_data_mswep/valid_meta.csv')
		elif 'event_set' in data_mode:
			if 'mswep' in data_mode:
				meta = pd.read_csv(f'/user/work/al18709/tc_data_flipped/KE_tracks/tcs_and_storms_meta2.csv')
				meta['centre_lat'] = meta.centre_lat
				meta['centre_lon'] = meta.centre_lon
				print(meta.columns)
			else:
				meta = pd.read_csv(f'/user/home/al18709/work/ke_track_inputs/{model_}_{scenario}_tracks.csv')
				meta['centre_lat'] = meta.lat
				meta['centre_lon'] = meta.lon
				print(meta.columns)
		else:
			meta = pd.read_csv('/user/work/al18709/tc_data_mswep/%s_meta.csv' % data_mode)
			
		print('restoring correct TC orientation...')
		if 'event_set' not in data_mode:
			seq_real = find_and_flip(seq_real,meta)
			low_res_inputs = find_and_flip(low_res_inputs,meta)
			pred = find_and_flip(pred,meta)
		else:
			print('Memory usage before deleting variables: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
			del nn, seq_real, low_res_inputs, model, img_pred, disc_img_pred, pred_single, pred_single_disc, topography, ds, x, y, z, x_, y_, z_, data_predict
			gc.collect()
			print('Memory usage after deleting variables: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
			pred_flipped = find_and_flip(pred,meta)
			del pred,meta
			gc.collect()


			




	if vaegan == True:
		model = 'vaegan'
		# problem = '7_better_spread-error'
	else:
		model = 'gan'	
		# problem = '5_normal_problem'
		problem = 'modular_part2_raw'
		# problem = 'modular_part2_patchloss_raw_4'

	if data_mode == 'storm':
		problem = storm

	if data_mode == 'era5_corrected':
		problem = '3_hrly'

	if 'scalar' in data_mode:
		data_mode = 'modular_part2_lowres_predictions_%s_2' % data_mode
	
	if 'event_set' in data_mode:
		print('saving event set to: /user/home/al18709/work/ke_track_rain/hr/')
		print('Memory usage: ',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,'MB')
		problem = 'event_set'
		np.save(f'/user/home/al18709/work/ke_track_rain/hr/{model_}_{scenario}_pred.npy',pred_flipped)
		np.save(f'/user/home/al18709/work/ke_track_rain/hr/{model_}_{scenario}_disc_pred.npy',disc_pred)
		# del pred_flipped
		# vars_to_keep = ['np','problem','model_','scenario','resource']
		# for var_name in globals().keys():
		# 	if var_name not in vars_to_keep and not callable(globals()[var_name]):
		# 		print('deleting ',var_name,'...')
		# 		del globals()[var_name]
		# gc.collect()
		
		# data_pred = np.load(f'/user/home/al18709/work/ke_track_rain/hr/{model_}_{scenario}_pred.npy')
		# print(data_pred.shape)
		pred_flipped_normalised = 10**pred_flipped - 1
		print('saving event set to: /user/home/al18709/work/ke_track_rain/hr/')	
		# np.save(f'/user/home/al18709/work/ke_track_rain/hr/{model_}_{scenario}_pred.npy',pred_flipped_normalised)
		np.save(f'/user/home/al18709/work/ke_track_rain/hr/{model_}_{scenario}_pred_qm.npy',pred_flipped_normalised)	
		print(f'saved! /user/home/al18709/work/ke_track_rain/hr/{model_}_{scenario}_pred_qm.npy')	
		# print(f'saved! /user/home/al18709/work/ke_track_rain/hr/{model_}_{scenario}_pred.npy')
		
	else:

		seq_real = 10**seq_real - 1
		pred = 10**pred - 1
		low_res_inputs = 10**low_res_inputs - 1

		print(pred)
		print(np.sum(pred))

		np.save('/user/home/al18709/work/%s_predictions_20/%s_real-%s_%s.npy' % (model,data_mode,checkpoint,problem),seq_real)
		np.save('/user/home/al18709/work/%s_predictions_20/%s_pred-%s_%s.npy' % (model,data_mode,checkpoint,problem),pred)
		np.save('/user/home/al18709/work/%s_predictions_20/%s_disc_pred-%s_%s.npy' % (model,data_mode,checkpoint,problem),disc_pred)
		np.save('/user/home/al18709/work/%s_predictions_20/%s_input-%s_%s.npy' % (model,data_mode,checkpoint,problem),low_res_inputs)
		print('/user/home/al18709/work/%s_predictions_20/%s_real-%s_%s.npy' % (model,data_mode,checkpoint,problem))
		print('/user/home/al18709/work/%s_predictions_20/%s_disc_pred-%s_%s.npy' % (model,data_mode,checkpoint,problem))









