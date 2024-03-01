
from ast import Continue
import numpy as np
import pandas as pd
from tfrecords_generator_ifs import create_fixed_dataset
import setupmodel
from noise import NoiseGenerator
import tensorflow as tf
import gc,os


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
	input_channels = 6
	# input_channels = 6
	# noise_channels = 6 #4
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
	elif data_mode == 'storm':
		num_images,_,_ = np.load('/user/work/al18709/tc_data_mswep_extend_flipped/y_%s.npy' % storm).shape
	elif (data_mode == 'storm_era5') or (data_mode == 'storm_era5_corrected'):
		num_images,_,_ = np.load('/user/work/al18709/tc_data_era5_flipped_10/y_%s.npy' % storm).shape
	elif (data_mode == 'era5') or (data_mode == 'era5_corrected'):
		num_images,_,_ = np.load('/user/home/al18709/work/tc_data_era5_flipped_10/valid_y.npy').shape
	elif (data_mode == 'miroc') or (data_mode == 'miroc_corrected'):
		num_images,_ = np.load('/user/work/al18709/tc_data_flipped/KE_tracks/ke_miroc6-hist_qm_corrected.npy')[:30000,:].shape
	elif (data_mode == 'miroc_ssp585'):
		num_images,_ = 30000
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
		num_images,_,_,_ = np.float32(np.expand_dims(np.expand_dims(np.load(f'/user/home/al18709/work/ke_track_inputs/{model_}_{scenario}_tracks.npy'),axis=1),axis=1)).shape      
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
	if "event_set" not in data_mode:
		data_predict = create_fixed_dataset(predict_year,
											batch_size=batch_size,
											downsample=False,
											mode = data_mode,
											storm=storm)
	else:
		x,z,y = create_fixed_dataset(predict_year,
											batch_size=batch_size,
											downsample=False,
											mode = data_mode,
											storm=storm)

		
	# load model weights from main file
	if mode == 'VAEGAN':
		vaegan = True
	elif mode == 'GAN':
		vaegan = False

	print('log folder is:',log_folder)
	print(vaegan)
	gen_weights_file = log_folder + '/models-gen_weights.h5'
	files = os.listdir(log_folder + '/models/')
	checkpoints = []
	for file in files:
		cp = file[-10:-3]
		checkpoints.append(int(cp))
	print(checkpoints)
	latest_checkpoint = max(checkpoints)
	# latest_checkpoint = '64000'
	# latest_checkpoint = '0640000'
	# latest_checkpoint = '0716800'
	# gen_weights_file = log_folder + '/models/' +'gen_weights-0' + str(latest_checkpoint) + '.h5'
	# latest_checkpoint = '0960000' #this one best so far on model 31
	gen_weights_file = log_folder + '/models/' +'gen_weights-' + str(latest_checkpoint) + '.h5'
	# gen_weights_files = '/user/home/al18709/work/gan/logs_scalar_wgan_v13/models/gen_weights-1075200.h5' # best option
	# gen_weights_file = log_folder + '/models-gen_opt_weights.h5' # TODO: this has different construction to gen_weights - ask andrew and lucy
	model.gen.built = True
	model.gen.load_weights(gen_weights_file) 

	# print(data_predict)
	# print(data_predict.batch(batch_size))
	# print(iter(data_predict.batch(batch_size)))
	# define initial variables
	# pred = np.zeros((num_images,100,100,20))
	pred = np.zeros((num_images,10,10,20))
	# seq_real = np.zeros((num_images,100,100,1))
	seq_real = np.zeros((num_images,10,10,1))
	# low_res_inputs = np.zeros((num_images,10,10,1))
	low_res_inputs = np.zeros((num_images,10,10,6))
	
	if "event_set" not in data_mode:
		data_pred_iter = iter(data_predict)
	else:
		n_b = int(num_images/batch_size)
		n_b_10 = int(n_b/10) * num_images
		# data_predict = tf.data.Dataset.from_tensor_slices((x[:n_b_3], z[:n_b_3], y[:n_b_3]))
		# data_pred_iter = iter(data_predict)
		# x_iter = iter(x)
		# z_iter = iter(z)
		# y_iter = iter(y)
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
		# inputs, outputs = next(data_pred_iter)
		if "event_set" not in data_mode:
			inputs, topography, outputs = next(data_pred_iter)
		else:
			number_of_loaded_batches = 10
			if i in [n * int(n_b/number_of_loaded_batches) for n in range(number_of_loaded_batches+1)]:
				print('batch ', i)
				first_image_i = j_*int(num_images/number_of_loaded_batches)
				second_image_i = (j_+1)*int(num_images/number_of_loaded_batches)
				print(first_image_i,second_image_i)
				print('number in loaded batch',second_image_i - first_image_i)
				x_ = x[first_image_i:second_image_i,:,:,:]
				z_ = z[first_image_i:second_image_i,:,:,:]
				y_ = y[first_image_i:second_image_i,:,:,:]
				print('shapes: ',x_.shape,y_.shape,z_.shape)
				ds = tf.data.Dataset.from_tensor_slices((x_, z_, y_))
				data_predict = ds.batch(batch_size)
				data_pred_iter = iter(data_predict)
				inputs,topography,outputs = next(data_pred_iter)
				j_ = j_+1
		print(data_pred_iter)
		print(data_predict)
			
			
		if (data_mode == 'era5') or (data_mode == 'era5_corrected') or (data_mode == 'storm_era5') or (data_mode == 'storm_era5_corrected'):
			if i == batch_size:
				n = remainder
			else:
				n = batch_size
			# outputs = np.zeros((n,100,100,1))
			outputs = np.zeros((n,10,10,1))
		# if i !=nbatches:
		# 	continue
		# print(inputs.shape)
		# print(topography.shape)
		# print(outputs.shape)
		print('remainder',remainder)
		print('number of batches',nbatches)
		print(num_images)
		print(batch_size)
		

		img_real = outputs
		img_pred = []	   
		# noise_shape = inputs[0,...,0].shape + (noise_channels,)
		# noise_shape = (5,5) + (noise_channels,)
		noise_shape = (10,10) + (noise_channels,)
		# noise_hr_shape = (100,100) + (noise_channels,)
		noise_hr_shape = (50,50) + (noise_channels,)
		print('noise shape: ',noise_shape)
		print('noise_hr shape" ',noise_hr_shape)
		if i == nbatches:
			noise_gen = NoiseGenerator(noise_shape, batch_size=remainder) # does noise gen need to be outside of the for loop?
			noise_hr_gen = NoiseGenerator(noise_hr_shape, batch_size=remainder)
			# img_pred = np.zeros((remainder,100,100,20))
			img_pred = np.zeros((remainder,10,10,20))
		else:
			noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size) # does noise gen need to be outside of the for loop?
			noise_hr_gen = NoiseGenerator(noise_hr_shape, batch_size=batch_size)
			# img_pred = np.zeros((batch_size,100,100,20))
			img_pred = np.zeros((batch_size,10,10,20))
			
		
		print('noise gen shape',noise_gen().shape)
		print('noise gen hr shape',noise_hr_gen().shape)

		for j in range(20): #do 50 ensemble members
				
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
				nn_hr = noise_hr_gen()
				# print('inputs shape: ', inputs.shape)
				# print('topography.shape: ',topography.shape)
				# print('noise shape: ',nn.shape)
				# print('noise_hr shape: ', nn_hr.shape)
				if "event_set" in data_mode:
					# nn = tf.data.Dataset.from_tensor_slices((nn))
					# nn_hr = tf.data.Dataset.from_tensor_slices((nn_hr))
					print(nn.dtype)
					print(nn_hr.dtype)
					# print(inputs.dtype)
					# print(topography.dtype)
					# inputs = np.array(inputs)
					# topography = np.array(topography)
					# print(inputs.dtype)
					# print(topography.dtype)

				# pred_single = np.array(model.gen.predict([inputs,nn]))[:,:,:,0] # this one
				print('inputs shape',inputs.shape)
				# print(topography)
				pred_single = np.array(model.gen.predict([inputs,topography,nn,nn_hr]))[:,:,:,0]

				# pred_single = np.array(model.gen.predict_on_batch([inputs,nn]))[:,:,:,0]
				gc.collect()
				
			# print(pred_single.shape)
			if i == nbatches:
				img_pred[:remainder,:,:,j] = pred_single
			else:
				img_pred[:,:,:,j] = pred_single
			# print(img_pred.shape)
			

		print('img pred shape: ',img_pred.shape)
		print('img real shape: ',img_real.shape)
		print('seq_real.shape: ',seq_real.shape)
		print('assigning images ',i*batch_size,' to ',i*batch_size + batch_size,'...')
		if i == nbatches:
			if "event_set" not in data_mode:
				seq_real[i*batch_size:,:,:,:] = img_real[:remainder]
			else:
				seq_real[i*batch_size:,:,:,:] = img_pred[:remainder,:,:,0:1]
			# seq_real[i*batch_size:,:,:,0] = img_real[:remainder]
			pred[i*batch_size:,:,:,:] = img_pred[:remainder]
			low_res_inputs[i*batch_size:,:,:,:] = inputs[:remainder]
		else:
			if "event_set" not in data_mode:
				seq_real[i*batch_size:i*batch_size + batch_size,:,:,:] = img_real
			else:
				seq_real[i*batch_size:i*batch_size + batch_size,:,:,:] = img_pred[:,:,:,0:1]
			# seq_real[i*batch_size:i*batch_size + batch_size,:,:,0] = img_real
			pred[i*batch_size:i*batch_size + batch_size,:,:,:] = img_pred
			low_res_inputs[i*batch_size:i*batch_size + batch_size,:,:,:] = inputs

			
	# TODO: transfer to cpu memory not gpu memory
	print(mode)
	print(data_mode)
	print(seq_real.shape)
	print(pred.shape)
	print(low_res_inputs.shape)

	# print(seq_real)
	flip = True
	flip = False

	if flip == True:
		if data_mode == 'validation':
			meta = pd.read_csv('/user/work/al18709/tc_data_mswep/valid_meta.csv')
		elif data_mode == 'storm':
			meta = pd.read_csv('/user/work/al18709/tc_data_mswep_extend_flipped/meta_%s.csv' % storm)
		elif data_mode == 'storm_era5':
			meta = pd.read_csv('/user/work/al18709/tc_data_era5_flipped_10/meta_%s.csv' % storm)
		elif data_mode == 'storm_era5_corrected':
			meta = pd.read_csv('/user/work/al18709/tc_data_era5_flipped_10/meta_%s.csv' % storm)
		elif (data_mode == 'era5') or (data_mode == 'era5_corrected'):
			meta = pd.read_csv('/user/work/al18709/tc_data_era5_10/valid_meta.csv')
		elif (data_mode == 'miroc') or (data_mode == 'miroc_corrected') or (data_mode == 'miroc_ssp585'):
			meta = pd.read_csv('/user/home/al18709/work/ibtracks/miroc6_hist_tracks.csv').head(30000)
			meta.rename(columns={'lat':'centre_lat'},inplace=True)
			meta.rename(columns={'lon':'centre_lon'},inplace=True)
		else:
			meta = pd.read_csv('/user/work/al18709/tc_data_mswep/%s_meta.csv' % data_mode)
		
		seq_real = find_and_flip(seq_real,meta)
		pred = find_and_flip(pred,meta)
		low_res_inputs = find_and_flip(low_res_inputs,meta)




	if vaegan == True:
		model = 'vaegan'
		# problem = '7_better_spread-error'
	else:
		model = 'gan'	
		# problem = '5_normal_problem'
		# problem = 'scalar_test_run_5'
		# problem = 'scalar_test_run_more_inputs'
		problem = 'scalar_test_run_1'
		problem = 'scalar_test_run_1b'

	if data_mode == 'storm':
		problem = storm

	if data_mode == 'era5_corrected':
		problem = '3_hrly'
	
	if 'event_set' in data_mode:
		problem = 'event_set'

	seq_real = 10**seq_real - 1
	# don't need to denormalise the results, actually seems like you do
	pred = 10**pred - 1
	# low_res_inputs = 10**low_res_inputs - 1

	np.save('/user/home/al18709/work/%s_predictions_20/%s_real-%s_%s.npy' % (model,data_mode,checkpoint,problem),seq_real)
	np.save('/user/home/al18709/work/%s_predictions_20/%s_pred-%s_%s.npy' % (model,data_mode,checkpoint,problem),pred)
	np.save('/user/home/al18709/work/%s_predictions_20/%s_input-%s_%s.npy' % (model,data_mode,checkpoint,problem),low_res_inputs)
	print('/user/home/al18709/work/%s_predictions_20/%s_pred-%s_%s.npy' % (model,data_mode,checkpoint,problem))









