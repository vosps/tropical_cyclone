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
		inputs = np.load('/user/home/al18709/work/gan_predictions_20/%s_input-opt_improve.npy' % set)
		pred_cnn = np.load('/user/home/al18709/work/cnn/unet_valid_2.npy')
		pred_gan = np.load('/user/home/al18709/work/gan_predictions_20/%s_pred-opt_improve.npy' % set)[:,:,:,0]
		pred_vaegan = np.load('/user/home/al18709/work/vaegan_predictions_20/%s_pred-opt_improve.npy' % set)[:,:,:,0]
		pred_gan_ensemble = np.load('/user/home/al18709/work/gan_predictions_20/%s_pred-opt_improve.npy' % set)
		pred_vaegan_ensemble = np.load('/user/home/al18709/work/vaegan_predictions_20/%s_pred-opt_improve.npy' % set)


		if set == 'validation':
			set = 'valid'
		meta = pd.read_csv('/user/work/al18709/tc_data_mswep/%s_meta.csv' % set)

	elif results == 'test':
		real = np.load('/user/home/al18709/work/gan_predictions_20/%s_real-opt_improve.npy' % set)
		inputs = np.load('/user/home/al18709/work/gan_predictions_20/%s_input-opt_improve.npy' % set)
		pred_cnn = np.load('/user/home/al18709/work/cnn/unet_valid_2.npy')
		pred_gan = np.load('/user/home/al18709/work/gan_predictions_20/%s_pred-opt_improve.npy' % set)[:,:,:,0]
		pred_vaegan = np.load('/user/home/al18709/work/vaegan_predictions_20/%s_pred-opt_1_better-noise.npy' % set)[:,:,:,0]


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

