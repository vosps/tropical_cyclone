
from ast import Continue
import numpy as np
import pandas as pd
from tfrecords_generator_ifs import create_fixed_dataset
import setupmodel
from noise import NoiseGenerator
import gc


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
	data_predict = create_fixed_dataset(predict_year,
										batch_size=batch_size,
										downsample=False,
										mode = data_mode,
										storm=storm)

		
	# load model weights from main file
	if mode == 'VAEGAN':
		vaegan = True
	elif mode == 'GAN':
		vaegan = False

	# load disc model and make predictions
	disc_weights_file = log_folder + '/models-disc_weights.h5'
	model.disc.built = True
	model.disc.load_weights(disc_weights_file)

	#  load gen model and make predictions
	print('log folder is:',log_folder)
	gen_weights_file = log_folder + '/models-gen_weights.h5'
	gen_weights_files = log_folder + 'models/gen_weights-9984000.h5'
	# gen_weights_file = '/user/home/al18709/work/gan/logs_wgan_modular_v7/models/gen_weights-19814400.h5' #best use this
	# version 7 7552000 best PSD so far
	# 9600000 no
	# 9420800 better - same as mwgan 1 and 2
	# 9497600 quite close to the wgan part 2 but tails off a tiny bit at the end.
	# 9472000 not much better
	# 9190400 not lunch better
	# 9523200 kind of like but not as good 9497600 
	# 9574400 good but not best
	# 9548800 awful
	# 9984000 best so far

	# gen_weights_file = '/user/home/al18709/work/gan/logs_wgan_modular_patchloss_v1/models/gen_weights-0358400.h5'
	# gen_weights_file = log_folder + '/models-gen_opt_weights.h5' # TODO: this has different construction to gen_weights - ask andrew and lucy
	model.gen.built = True
	model.gen.load_weights(gen_weights_file)

	# define initial variables
	pred = np.zeros((num_images,100,100,20))
	seq_real = np.zeros((num_images,100,100,1))
	disc_pred = np.zeros((num_images,1,20))
	# disc_real = np.zeros((num_images,1))
	# low_res_inputs = np.zeros((num_images,10,10,1))
	low_res_inputs = np.zeros((num_images,10,10,7))
	# disc_inputs = np.zeros((num_images,100,100,20))
	data_pred_iter = iter(data_predict)
	
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
		inputs, topography, outputs = next(data_pred_iter)
		if (data_mode == 'era5') or (data_mode == 'era5_corrected') or (data_mode == 'storm_era5') or (data_mode == 'storm_era5_corrected'):
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
			img_pred = np.zeros((remainder,100,100,20))
			disc_img_pred = np.zeros((remainder,1,20))
		else:
			noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size) # does noise gen need to be outside of the for loop?
			img_pred = np.zeros((batch_size,100,100,20))
			disc_img_pred = np.zeros((batch_size,1,20))

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
			# seq_real[i*batch_size:,:,:,:] = img_real[:remainder]
			seq_real[i*batch_size:,:,:,0] = img_real[:remainder]
			pred[i*batch_size:,:,:,:] = img_pred[:remainder]
			disc_pred[i*batch_size:,:,:] = disc_img_pred[:remainder]
			low_res_inputs[i*batch_size:,:,:,:] = inputs[:remainder]
		else:
			# seq_real[i*batch_size:i*batch_size + batch_size,:,:,:] = img_real
			seq_real[i*batch_size:i*batch_size + batch_size,:,:,0] = img_real
			pred[i*batch_size:i*batch_size + batch_size,:,:,:] = img_pred
			disc_pred[i*batch_size:i*batch_size + batch_size,:,:] = disc_img_pred
			low_res_inputs[i*batch_size:i*batch_size + batch_size,:,:,:] = inputs

			
	# TODO: transfer to cpu memory not gpu memory
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
		# problem = 'modular_part2_raw'
		problem = 'modular_part2_patchloss_raw'

	if data_mode == 'storm':
		problem = storm

	if data_mode == 'era5_corrected':
		problem = '3_hrly'

	if 'scalar' in data_mode:
		data_mode = 'modular_part2_lowres_predictions_%s' % data_mode
	

	seq_real = 10**seq_real - 1
	pred = 10**pred - 1
	low_res_inputs = 10**low_res_inputs - 1

	print(pred)

	np.save('/user/home/al18709/work/%s_predictions_20/%s_real-%s_%s.npy' % (model,data_mode,checkpoint,problem),seq_real)
	np.save('/user/home/al18709/work/%s_predictions_20/%s_pred-%s_%s.npy' % (model,data_mode,checkpoint,problem),pred)
	np.save('/user/home/al18709/work/%s_predictions_20/%s_disc_pred-%s_%s.npy' % (model,data_mode,checkpoint,problem),disc_pred)
	np.save('/user/home/al18709/work/%s_predictions_20/%s_input-%s_%s.npy' % (model,data_mode,checkpoint,problem),low_res_inputs)
	print('/user/home/al18709/work/%s_predictions_20/%s_real-%s_%s.npy' % (model,data_mode,checkpoint,problem))









