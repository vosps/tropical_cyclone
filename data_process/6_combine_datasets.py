
"""
combine all the datasets in tc_data_flipped_var and tc_data_flipped_mswep to create a set of datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from multiprocessing import Pool
from itertools import groupby
import re
import sys

indir_var = '/user/home/al18709/work/tc_data_flipped_var/'
indir_mswep = '/user/home/al18709/work/tc_data_mswep_flipped/'
# indir_topography = ''
outdir = '/user/home/al18709/work/tc_data_flipped/'
split = False

if split == False:
	mswep_X = np.load(indir_var + 'mslp' + '_all_X.npy')
	print('var shape: ',mswep_X.shape)
	X = np.zeros((mswep_X.shape[0],mswep_X.shape[1],mswep_X.shape[2],9))
	# X[:,:,:,0] = mswep_X
	i=0
	for var in ['mslp','q-925','u-200','u-850','v-200','v-850','t-600','rh-600']:
		var_X = np.load(indir_var + var + '_all_X.npy')
		print(var)
		i = i+1
		print('var_X shape: ',var_X.shape)
		X[:,:,:,i] = var_X
	print(X.shape)
	np.save(outdir + 'all_combined_X_final.npy',X)

	# y = np.load(indir_mswep+'mswep_' + dataset+'_y.npy')
	# np.save(outdir + dataset+ '_combined_y_final.npy',y)
	# meta = pd.read_csv('/user/home/al18709/work/tc_data_mswep/' +dataset+'_meta_tcs_and_storms.csv')
	meta = pd.read_csv('/user/home/al18709/work/tc_data_var/mslp_' + 'all_meta.csv')
	meta.to_csv(outdir + 'all_meta_final.csv')

else:

	for dataset in ['train','test','valid','extreme_valid','extreme_test']:
		print(dataset,'\n')
		mswep_X = np.load(indir_mswep+'mswep_' + dataset+'_X_.npy')
		print('mswep shape: ',mswep_X.shape)
		X = np.zeros((mswep_X.shape[0],mswep_X.shape[1],mswep_X.shape[2],9))
		# (17467, 10, 10)
		mswep_X = np.load(indir_var + 'mslp' + '_' + dataset + '_X.npy')
		print('var shape: ',mswep_X.shape)
		X = np.zeros((mswep_X.shape[0],mswep_X.shape[1],mswep_X.shape[2],9))
		# X[:,:,:,0] = mswep_X
		i=0
		for var in ['mslp','q-925','u-200','u-850','v-200','v-850','t-600','rh-600']:
			var_X = np.load(indir_var + var + '_' + dataset + '_X.npy')
			print(var)
			i = i+1
			print('var_X shape: ',var_X.shape)
			X[:,:,:,i] = var_X
		print(X.shape)
		np.save(outdir + dataset+ '_combined_X_final.npy',X)

		y = np.load(indir_mswep+'mswep_' + dataset+'_y.npy')
		np.save(outdir + dataset+ '_combined_y_final.npy',y)
		# meta = pd.read_csv('/user/home/al18709/work/tc_data_mswep/' +dataset+'_meta_tcs_and_storms.csv')
		meta = pd.read_csv('/user/home/al18709/work/tc_data_var/mslp_' +dataset+'_meta.csv')
		meta.to_csv(outdir + dataset + '_meta_final.csv')



	

