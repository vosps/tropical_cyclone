
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

for dataset in ['train','test','valid','extreme_valid','extreme_test']:
	mswep_X = np.load(indir_mswep+'mswep_' + dataset+'_X.npy')
	X = np.zeros((mswep_X.shape[0],mswep_X.shape[1],mswep_X.shape[2],7))
	X[:,:,:,0] = mswep_X
	i=0
	for var in ['mslp','q-925','u-200','u-850','v-200','v-850']:
		var_X = np.load(indir_var + var + '_' + dataset + '_X.npy')
		i = i+1
		X[:,:,:,i] = var_X
	print(X.shape)
	np.save(outdir + dataset+ '_combined_X.npy',X)

	y = np.load(indir_mswep+'mswep_' + dataset+'_y.npy')
	np.save(outdir + dataset+ '_combined_y.npy',y)
	meta = pd.read_csv('/user/home/al18709/work/tc_data_mswep/' +dataset+'_meta.csv')
	meta.to_csv(outdir + dataset + '_meta.csv')



	

