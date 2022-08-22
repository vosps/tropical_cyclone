
from ast import Continue
import numpy as np
import pandas as pd
from tfrecords_generator_ifs import create_fixed_dataset
import setupmodel
from noise import NoiseGenerator
import gc
import matplotlib.pyplot as plt

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
	print('generating predictions...')
	input_channels = 1
	noise_channels = 4 #4
	batch_size = 512 #512

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

	mode = 'extreme_valid'
	if mode == 'validation':
		num_images,_,_ = np.load('/user/work/al18709/tc_data_era5_flipped/valid_X.npy').shape
	elif mode == 'extreme_valid':
		num_images,_,_ = np.load('/user/work/al18709/tc_data_era5_flipped/extreme_valid_X.npy').shape
	print('number of images: ',num_images)

	# set initial variables

	# mode = 'validation'
	# mode = 'cmip'
	# mode = 'train'
	if gcm == True:
		mode = 'gcm'
	
	if gcm == True:
		batch_size = 1
		num_images = 5
		
		
	# load relevant data
	data_predict = create_fixed_dataset(predict_year,
										batch_size=batch_size,
										downsample=False,
										mode = mode)

		
	# load model weights from main file
	# TODO: update logs file locations
	# print('checkpoint = ',checkpoint)
	# if checkpoint == 'opt':
	# 	gen_weights_file = "/user/home/al18709/work/dsrnngan/logs/models-gen_weights.h5"
	# else:
	# 	gen_weights_file = "/user/home/al18709/work/dsrnngan/logs/models/gen_weights-%s.h5" % checkpoint

	vaegan = False
	# if vaegan:
	# 	gen_weights_file = "/user/home/al18709/work/vaegan/logs/models-gen_weights.h5"
	# else:
	# 	gen_weights_file = "/user/home/al18709/work/dsrnngan/logs/models-gen_weights.h5"
	print('log folder is:',log_folder)
	print(vaegan)
	gen_weights_file = log_folder + '/models-gen_weights.h5'
	# gen_weights_file = log_folder + '/models-gen_opt_weights.h5' # TODO: this has different construction to gen_weights - ask andrew and lucy
	model.gen.built = True
	model.gen.load_weights(gen_weights_file)

	print(data_predict)
	print(data_predict.batch(batch_size))
	print(iter(data_predict.batch(batch_size)))
	# define initial variables
	# pred = []
	# seq_real = []
	# low_res_inputs = []
	pred = np.zeros((num_images,100,100,20))
	seq_real = np.zeros((num_images,100,100,1))
	low_res_inputs = np.zeros((num_images,40,40,1))
	data_pred_iter = iter(data_predict)
	# unbatch first
	# data_predict = data_predict.unbatch()
	# data_pred_iter = iter(data_predict.batch(batch_size))
	nbatches = int(num_images/batch_size)
	remainder = num_images - nbatches*batch_size

	print(nbatches)	
	# loop through batches
	for i in range(nbatches+1):
		
		print('running batch ',i,'...')
		inputs, outputs = next(data_pred_iter)
		# if i !=nbatches:
		# 	continue
		print(inputs.shape)
		print(outputs.shape)
		print('remainder',remainder)
		print('number of batches',nbatches)
		print(num_images)
		print(batch_size)
		

		img_real = outputs
		img_pred = []	   
		noise_shape = inputs[0,...,0].shape + (noise_channels,)
		print('noise shape: ',noise_shape)
		if i == nbatches:
			noise_gen = NoiseGenerator(noise_shape, batch_size=remainder) # does noise gen need to be outside of the for loop?
			img_pred = np.zeros((remainder,100,100,20))
		else:
			noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size) # does noise gen need to be outside of the for loop?
			img_pred = np.zeros((batch_size,100,100,20))

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
				pred_single = np.array(model.gen.predict([inputs,nn]))[:,:,:,0]
				print('prediction shape: ',pred_single.shape)
				plt.imshow(pred_single[0])
				plt.savefig('figs/test.png')
				
				# pred_single = np.array(model.gen.predict_on_batch([inputs,nn]))[:,:,:,0]
				gc.collect()
				
			# print(pred_single.shape)
			if i == nbatches:
				img_pred[:remainder,:,:,j] = pred_single
			else:
				img_pred[:,:,:,j] = pred_single
			# print(img_pred.shape)
			

		print('img pred shape: ',img_pred.shape)
		print('assigning images ',i*batch_size,' to ',i*batch_size + batch_size,'...')
		if i == nbatches:
			seq_real[i*batch_size:,:,:,:] = img_real[:remainder]
			pred[i*batch_size:,:,:,:] = img_pred[:remainder]
			low_res_inputs[i*batch_size:,:,:,:] = inputs[:remainder]
		else:
			seq_real[i*batch_size:i*batch_size + batch_size,:,:,:] = img_real
			pred[i*batch_size:i*batch_size + batch_size,:,:,:] = img_pred
			low_res_inputs[i*batch_size:i*batch_size + batch_size,:,:,:] = inputs

			
	# TODO: transfer to cpu memory not gpu memory
	print(mode)
	print(seq_real.shape)
	print(pred.shape)
	print(low_res_inputs.shape)

	

	# print(seq_real)
	flip = True

	if flip == True:
		if mode == 'validation':
			meta = pd.read_csv('/user/work/al18709/tc_data_era5/valid_meta.csv')
		else:
			meta = pd.read_csv('/user/work/al18709/tc_data_era5/%s_meta.csv' % mode)
		seq_real = find_and_flip(seq_real,meta)
		pred = find_and_flip(pred,meta)
		low_res_inputs = find_and_flip(low_res_inputs,meta)




	if vaegan == True:
		model = 'vaegan'
	else:
		model = 'gan'
	np.save('/user/home/al18709/work/%s_predictions_20/%s_real-%s_era5_era-to-mswep_2.npy' % (model,mode,checkpoint),seq_real)
	np.save('/user/home/al18709/work/%s_predictions_20/%s_pred-%s_era5_era-to-mswep_2.npy' % (model,mode,checkpoint),pred)
	np.save('/user/home/al18709/work/%s_predictions_20/%s_input-%s_era5_era-to-mswep_2.npy' % (model,mode,checkpoint),low_res_inputs)









