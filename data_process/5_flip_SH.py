"""
using imshow, we plot the raw data array. This reveals the issue with mswep: the mswep data is fortmatted with descending latitude points. 

If you plot with the actual lat values using contourf then the image is as expected (nh tcs rotate anticlockwise).

But the raw values show the storm rotating the opposite way. So we must:
1. flip the NH tcs so that the data sees them rotating anticlockwise
2. remember that the SH tcs are spinning anticlockwise
3. remember when feeding in new TCs (e.g. from GCMs) that they will have to be fed in rotating anticlockwise for appropriate predictions

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from multiprocessing import Pool
from itertools import groupby
import re
import sys
# sns.set_style("white")

def flip(tc):
	tc_flipped = np.flip(tc,axis=0)
	return tc_flipped

def plot_tc(tc,title):
	plt.imshow(tc)
	plt.savefig(title)
	plt.clf()


def find_and_flip(X,y,meta,dataset='mswep'):
	print(X.shape)
	print(y.shape)
	# print(meta)

	sh_indices = meta[meta['centre_lat'] < 0].index
	nh_indices = meta[meta['centre_lat'] > 0].index
	print(np.sum(meta['centre_lat'] < 0))
	print(np.sum(meta['centre_lat'] > 0))

	# flip the nh tcs so they are rotating anticlockwise (in the raw data array)
	# in mswep they can be plotted correctly with the mswep lats and lons, but the raw data shows them flipped as mswep has descending latitudes
	for i in nh_indices:
		print(i,end='\r')
		# plot_tc(y[i],'normal')
		
		# era5 doesn't need to be flipped because it's already correct, t doesn't need to flip X because it is high res (y)
		if dataset in ['mswep','var']:
			X_flipped = flip(X[i])
			X[i] = X_flipped

		if dataset == 'var':
			y_flipped = y
		else:
			y_flipped = flip(y[i])
			y[i] = y_flipped

	# flip the y values in the hr era5 dataset because they rotate differently
	if (dataset == 'era5') or (dataset == 'era5_storm'):
		for i in sh_indices:
			print('flipping era5...')
			y_flipped = flip(y[i])
			y[i] = y_flipped
		
		
		# plot_tc(y[i],'flipped')
	
	return X,y

print('running script 5...')
# dataset = 'era5'
# dataset = 'era5_storm'
variable = sys.argv[1]
print('variable: ', variable)
if '/' in variable:
	dataset = 'var'
	variable = variable.replace('/','-')
elif variable in ['mswep','era5','t']:
	dataset = variable
else: 
	dataset = 'var'

print(variable)
print(dataset)
resolution = 100


def save_flipped(grouped_sids):
	# print(sid)
	for sid in grouped_sids:
		if 'NAMED' in sid:
			continue
		if 'extreme_valid' in sid:
			continue
		# if dataset == 'mswep_extend':
		# 	X = np.load('/user/work/al18709/tc_Xy_extend/X_%s.npy' % sid)
		# 	y = np.load('/user/work/al18709/tc_Xy_extend/y_%s.npy' % sid)
		# 	meta = pd.read_csv('/user/work/al18709/tc_Xy_extend/meta_%s.csv' % sid)
		# else:
		X = np.load('/user/work/al18709/tc_Xy_era5_10/X_%s.npy' % sid)
		y = np.load('/user/work/al18709/tc_Xy_era5_10/y_%s.npy' % sid)
		meta = pd.read_csv('/user/work/al18709/tc_Xy_era5_10/meta_%s.csv' % sid)
			# dataset == 'era5_10'
		dataset = 'era5'
		X,y = find_and_flip(X,y,meta)
		print(X.shape)
		print(dataset)
		
		np.save('/user/work/al18709/tc_data_%s_flipped_10/X_%s.npy' % (dataset,sid),X)
		np.save('/user/work/al18709/tc_data_%s_flipped_10/y_%s.npy' % (dataset,sid),y)
		meta.to_csv('/user/work/al18709/tc_data_%s_flipped_10/meta_%s.csv' % (dataset,sid))
	print('complete')


def process(filepaths):
	print('doing process...')
	res = save_flipped(filepaths)
	return res



if (dataset == 'mswep_extend') or (dataset == 'era5_storm'):

	n_processes = 24
	if dataset == 'mswep_extend':
		files = glob.glob('/user/work/al18709/tc_Xy_extend/X_*.npy')
		sids = [file[34:47] for file in files]
	else:
		files = glob.glob('/user/work/al18709/tc_Xy_era5_10/X_*.npy')
		sids = [file[35:48] for file in files]
	
	save_flipped(sids)
	exit()
	print('about to do process...')
	tc_split = np.array(np.array_split(sids, n_processes))
	p = Pool(processes=n_processes)
	pool_results = p.map(process, tc_split)
	p.close()
	p.join()

elif (dataset == 'era5') and (resolution == 100):
	print('saving era5')
	valid_X = np.load('/user/work/al18709/tc_data_%s_10/valid_X.npy' % dataset)
	valid_y = np.load('/user/work/al18709/tc_data_%s_10/valid_y.npy' % dataset)
	valid_meta = pd.read_csv('/user/work/al18709/tc_data_%s_10/valid_meta.csv' % dataset)
	train_X = np.load('/user/work/al18709/tc_data_%s_10/train_X.npy' % dataset)
	train_y = np.load('/user/work/al18709/tc_data_%s_10/train_y.npy' % dataset)
	train_meta = pd.read_csv('/user/work/al18709/tc_data_%s_10/train_meta.csv' % dataset)
	test_X = np.load('/user/work/al18709/tc_data_%s_10/test_X.npy' % dataset)
	test_y = np.load('/user/work/al18709/tc_data_%s_10/test_y.npy' % dataset)
	test_meta = pd.read_csv('/user/work/al18709/tc_data_%s_10/test_meta.csv' % dataset)
	extreme_test_X = np.load('/user/work/al18709/tc_data_%s_10/extreme_test_X.npy' % dataset)
	extreme_test_y = np.load('/user/work/al18709/tc_data_%s_10/extreme_test_y.npy' % dataset)
	extreme_test_meta = pd.read_csv('/user/work/al18709/tc_data_%s_10/extreme_test_meta.csv' % dataset)
	extreme_valid_X = np.load('/user/work/al18709/tc_data_%s_10/extreme_valid_X.npy' % dataset)
	extreme_valid_y = np.load('/user/work/al18709/tc_data_%s_10/extreme_valid_y.npy' % dataset)
	extreme_valid_meta = pd.read_csv('/user/work/al18709/tc_data_%s_10/extreme_valid_meta.csv' % dataset)

elif  (dataset != 'era5') and (resolution == 100) and (dataset != 'var') and (dataset != 't'):
	# have added in _ to get separate files of condensed TCs and storms only for final figures.
	valid_X = np.load('/user/work/al18709/tc_data_%s/valid_X.npy' % dataset)
	valid_y = np.load('/user/work/al18709/tc_data_%s/valid_y.npy' % dataset)
	valid_meta = pd.read_csv('/user/work/al18709/tc_data_%s/valid_meta.csv' % dataset)
	train_X = np.load('/user/work/al18709/tc_data_%s/train_X.npy' % dataset)
	train_y = np.load('/user/work/al18709/tc_data_%s/train_y.npy' % dataset)
	train_meta = pd.read_csv('/user/work/al18709/tc_data_%s/train_meta.csv' % dataset)
	test_X = np.load('/user/work/al18709/tc_data_%s/test_X.npy' % dataset)
	test_y = np.load('/user/work/al18709/tc_data_%s/test_y.npy' % dataset)
	test_meta = pd.read_csv('/user/work/al18709/tc_data_%s/test_meta.csv' % dataset)
	extreme_test_X = np.load('/user/work/al18709/tc_data_%s/extreme_test_X.npy' % dataset)
	extreme_test_y = np.load('/user/work/al18709/tc_data_%s/extreme_test_y.npy' % dataset)
	extreme_test_meta = pd.read_csv('/user/work/al18709/tc_data_%s/extreme_test_meta.csv' % dataset)
	extreme_valid_X = np.load('/user/work/al18709/tc_data_%s/extreme_valid_X.npy' % dataset)
	extreme_valid_y = np.load('/user/work/al18709/tc_data_%s/extreme_valid_y.npy' % dataset)
	extreme_valid_meta = pd.read_csv('/user/work/al18709/tc_data_%s/extreme_valid_meta.csv' % dataset)

elif dataset == 'var':
	split = False
	if split == False:
		all_X = np.load('/user/work/al18709/tc_data_var/%s_all_X.npy' % variable)
		all_meta = pd.read_csv('/user/work/al18709/tc_data_var/%s_all_meta.csv' % variable)
	else:
		valid_X = np.load('/user/work/al18709/tc_data_var/%s_valid_X.npy' % variable)
		valid_y = np.zeros((1,1))
		valid_meta = pd.read_csv('/user/work/al18709/tc_data_var/%s_valid_meta.csv' % variable)
		train_X = np.load('/user/work/al18709/tc_data_var/%s_train_X.npy' % variable)
		train_y = np.zeros((1,1))
		train_meta = pd.read_csv('/user/work/al18709/tc_data_var/%s_train_meta.csv' % variable)
		test_X = np.load('/user/work/al18709/tc_data_var/%s_test_X.npy' % variable)
		test_y = np.zeros((1,1))
		test_meta = pd.read_csv('/user/work/al18709/tc_data_var/%s_test_meta.csv' % variable)
		extreme_test_X = np.load('/user/work/al18709/tc_data_var/%s_extreme_test_X.npy' % variable)
		extreme_test_y = np.zeros((1,1))
		extreme_test_meta = pd.read_csv('/user/work/al18709/tc_data_var/%s_extreme_test_meta.csv' % variable)
		extreme_valid_X = np.load('/user/work/al18709/tc_data_var/%s_extreme_valid_X.npy' % variable)
		extreme_valid_y = np.zeros((1,1))
		extreme_valid_meta = pd.read_csv('/user/work/al18709/tc_data_var/%s_extreme_valid_meta.csv' % variable)

elif dataset == 't':
	split = False
	if split == False:
		all_X = np.load('/user/work/al18709/tc_data_t/%s_all_X.npy' % variable)
		all_meta = pd.read_csv('/user/work/al18709/tc_data_t/%s_all_meta.csv' % variable)
	else:
		valid_X = np.load('/user/work/al18709/tc_data_t/valid_X.npy')
		valid_y = np.load('/user/work/al18709/tc_data_t/valid_y.npy')
		valid_meta = pd.read_csv('/user/work/al18709/tc_data_t/valid_meta.csv')
		train_X = np.load('/user/work/al18709/tc_data_t/train_X.npy')
		train_y = np.load('/user/work/al18709/tc_data_t/train_y.npy')
		train_meta = pd.read_csv('/user/work/al18709/tc_data_t/train_meta.csv')
		test_X = np.load('/user/work/al18709/tc_data_t/test_X.npy')
		test_y = np.load('/user/work/al18709/tc_data_t/test_y.npy')
		test_meta = pd.read_csv('/user/work/al18709/tc_data_t/test_meta.csv')
		extreme_test_X = np.load('/user/work/al18709/tc_data_t/extreme_test_X.npy')
		extreme_test_y = np.load('/user/work/al18709/tc_data_t/extreme_test_y.npy')
		extreme_test_meta = pd.read_csv('/user/work/al18709/tc_data_t/extreme_test_meta.csv')
		extreme_valid_X = np.load('/user/work/al18709/tc_data_t/extreme_valid_X.npy')
		extreme_valid_y = np.load('/user/work/al18709/tc_data_t/extreme_valid_y.npy')
		extreme_valid_meta = pd.read_csv('/user/work/al18709/tc_data_t/extreme_valid_meta.csv')

else:
	valid_X = np.load('/user/work/al18709/tc_data_%s_40/valid_X.npy' % dataset)
	valid_y = np.load('/user/work/al18709/tc_data_%s_40/valid_y.npy' % dataset)
	valid_meta = pd.read_csv('/user/work/al18709/tc_data_%s_40/valid_meta.csv' % dataset)
	train_X = np.load('/user/work/al18709/tc_data_%s_40/train_X.npy' % dataset)
	train_y = np.load('/user/work/al18709/tc_data_%s_40/train_y.npy' % dataset)
	train_meta = pd.read_csv('/user/work/al18709/tc_data_%s_40/train_meta.csv' % dataset)
	test_X = np.load('/user/work/al18709/tc_data_%s_40/test_X.npy' % dataset)
	test_y = np.load('/user/work/al18709/tc_data_%s_40/test_y.npy' % dataset)
	test_meta = pd.read_csv('/user/work/al18709/tc_data_%s_40/test_meta.csv' % dataset)
	extreme_test_X = np.load('/user/work/al18709/tc_data_%s_40/extreme_test_X.npy' % dataset)
	extreme_test_y = np.load('/user/work/al18709/tc_data_%s_40/extreme_test_y.npy' % dataset)
	extreme_test_meta = pd.read_csv('/user/work/al18709/tc_data_%s_40/extreme_test_meta.csv' % dataset)
	extreme_valid_X = np.load('/user/work/al18709/tc_data_%s_40/extreme_valid_X.npy' % dataset)
	extreme_valid_y = np.load('/user/work/al18709/tc_data_%s_40/extreme_valid_y.npy' % dataset)
	extreme_valid_meta = pd.read_csv('/user/work/al18709/tc_data_%s_40/extreme_valid_meta.csv' % dataset)

if dataset == 'mswep_extend':
	exit()

if dataset == 'era5_storm':
	exit()

if split == False:
	print('valid')
	all_X,all_y = find_and_flip(all_X,all_X,all_meta,dataset=dataset)
else:
	print('valid')
	valid_X,valid_y = find_and_flip(valid_X,valid_y,valid_meta,dataset=dataset)


	print('train')
	train_X,train_y = find_and_flip(train_X,train_y,train_meta,dataset=dataset)

	print('test')
	test_X,test_y = find_and_flip(test_X,test_y,test_meta,dataset=dataset)

	print('extreme_test')
	extreme_test_X,extreme_test_y = find_and_flip(extreme_test_X,extreme_test_y,extreme_test_meta,dataset=dataset)


	print('extreme valid')
	extreme_valid_X,extreme_valid_y = find_and_flip(extreme_valid_X,extreme_valid_y,extreme_valid_meta,dataset=dataset)




if resolution == 100 and dataset == 'mswep':
	# np.save('/user/work/al18709/tc_data_%s_flipped_10/valid_X.npy' % dataset,valid_X)
	# np.save('/user/work/al18709/tc_data_%s_flipped_10/valid_y.npy' % dataset,valid_y)
	# np.save('/user/work/al18709/tc_data_%s_flipped_10/train_X.npy' % dataset,train_X)
	# np.save('/user/work/al18709/tc_data_%s_flipped_10/train_y.npy' % dataset,train_y)
	# np.save('/user/work/al18709/tc_data_%s_flipped_10/test_X.npy' % dataset,test_X)
	# np.save('/user/work/al18709/tc_data_%s_flipped_10/test_y.npy' % dataset,test_y)
	# np.save('/user/work/al18709/tc_data_%s_flipped_10/extreme_test_X.npy' % dataset,extreme_test_X)
	# np.save('/user/work/al18709/tc_data_%s_flipped_10/extreme_test_y.npy' % dataset,extreme_test_y)
	# np.save('/user/work/al18709/tc_data_%s_flipped_10/extreme_valid_X.npy' % dataset,extreme_valid_X)
	# np.save('/user/work/al18709/tc_data_%s_flipped_10/extreme_valid_y.npy' % dataset,extreme_valid_y)
	np.save('/user/work/al18709/tc_data_mswep_flipped/%s_valid_X_.npy' % dataset,valid_X)
	np.save('/user/work/al18709/tc_data_mswep_flipped/%s_valid_y_.npy' % dataset,valid_y)
	np.save('/user/work/al18709/tc_data_mswep_flipped/%s_train_X_.npy' % dataset,train_X)
	np.save('/user/work/al18709/tc_data_mswep_flipped/%s_train_y_.npy' % dataset,train_y)
	np.save('/user/work/al18709/tc_data_mswep_flipped/%s_test_X_.npy' % dataset,test_X)
	np.save('/user/work/al18709/tc_data_mswep_flipped/%s_test_y_.npy' % dataset,test_y)
	np.save('/user/work/al18709/tc_data_mswep_flipped/%s_extreme_test_X_.npy' % dataset,extreme_test_X)
	np.save('/user/work/al18709/tc_data_mswep_flipped/%s_extreme_test_y_.npy' % dataset,extreme_test_y)
	np.save('/user/work/al18709/tc_data_mswep_flipped/%s_extreme_valid_X_.npy' % dataset,extreme_valid_X)
	np.save('/user/work/al18709/tc_data_mswep_flipped/%s_extreme_valid_y_.npy' % dataset,extreme_valid_y)
	# np.save('/user/work/al18709/tc_data_%s_flipped/valid_X.npy' % dataset,valid_X)
	# np.save('/user/work/al18709/tc_data_%s_flipped/valid_y.npy' % dataset,valid_y)
	# np.save('/user/work/al18709/tc_data_%s_flipped/train_X.npy' % dataset,train_X)
	# np.save('/user/work/al18709/tc_data_%s_flipped/train_y.npy' % dataset,train_y)
	# np.save('/user/work/al18709/tc_data_%s_flipped/test_X.npy' % dataset,test_X)
	# np.save('/user/work/al18709/tc_data_%s_flipped/test_y.npy' % dataset,test_y)
	# np.save('/user/work/al18709/tc_data_%s_flipped/extreme_test_X.npy' % dataset,extreme_test_X)
	# np.save('/user/work/al18709/tc_data_%s_flipped/extreme_test_y.npy' % dataset,extreme_test_y)
	# np.save('/user/work/al18709/tc_data_%s_flipped/extreme_valid_X.npy' % dataset,extreme_valid_X)
	# np.save('/user/work/al18709/tc_data_%s_flipped/extreme_valid_y.npy' % dataset,extreme_valid_y)
elif dataset == 'var':
	if split == False:
		np.save('/user/work/al18709/tc_data_flipped_var/%s_all_X.npy' % variable,all_X)
	else:
		np.save('/user/work/al18709/tc_data_flipped_var/%s_valid_X.npy' % variable,valid_X)
		np.save('/user/work/al18709/tc_data_flipped_var/%s_valid_y.npy' % variable,valid_y)
		np.save('/user/work/al18709/tc_data_flipped_var/%s_train_X.npy' % variable,train_X)
		np.save('/user/work/al18709/tc_data_flipped_var/%s_train_y.npy' % variable,train_y)
		np.save('/user/work/al18709/tc_data_flipped_var/%s_test_X.npy' % variable,test_X)
		np.save('/user/work/al18709/tc_data_flipped_var/%s_test_y.npy' % variable,test_y)
		np.save('/user/work/al18709/tc_data_flipped_var/%s_extreme_test_X.npy' % variable,extreme_test_X)
		np.save('/user/work/al18709/tc_data_flipped_var/%s_extreme_test_y.npy' % variable,extreme_test_y)
		np.save('/user/work/al18709/tc_data_flipped_var/%s_extreme_valid_X.npy' % variable,extreme_valid_X)
		np.save('/user/work/al18709/tc_data_flipped_var/%s_extreme_valid_y.npy' % variable,extreme_valid_y)

elif dataset == 't':
	if split == False:
		np.save('/user/work/al18709/tc_data_flipped_t/%s_all_X.npy' % variable,all_X)
	else:
		np.save('/user/work/al18709/tc_data_flipped_t/valid_X_.npy',valid_X)
		np.save('/user/work/al18709/tc_data_flipped_t/valid_y_.npy',valid_y)
		np.save('/user/work/al18709/tc_data_flipped_t/train_X_.npy',train_X)
		np.save('/user/work/al18709/tc_data_flipped_t/train_y_.npy',train_y)
		np.save('/user/work/al18709/tc_data_flipped_t/test_X_.npy',test_X)
		np.save('/user/work/al18709/tc_data_flipped_t/test_y_.npy',test_y)
		np.save('/user/work/al18709/tc_data_flipped_t/extreme_test_X_.npy',extreme_test_X)
		np.save('/user/work/al18709/tc_data_flipped_t/extreme_test_y_.npy',extreme_test_y)
		np.save('/user/work/al18709/tc_data_flipped_t/extreme_valid_X_.npy',extreme_valid_X)
		np.save('/user/work/al18709/tc_data_flipped_t/extreme_valid_y_.npy',extreme_valid_y)


else:
	np.save('/user/work/al18709/tc_data_%s_flipped_40/valid_X.npy' % dataset,valid_X)
	np.save('/user/work/al18709/tc_data_%s_flipped_40/valid_y.npy' % dataset,valid_y)
	np.save('/user/work/al18709/tc_data_%s_flipped_40/train_X.npy' % dataset,train_X)
	np.save('/user/work/al18709/tc_data_%s_flipped_40/train_y.npy' % dataset,train_y)
	np.save('/user/work/al18709/tc_data_%s_flipped_40/test_X.npy' % dataset,test_X)
	np.save('/user/work/al18709/tc_data_%s_flipped_40/test_y.npy' % dataset,test_y)
	np.save('/user/work/al18709/tc_data_%s_flipped_40/extreme_test_X.npy' % dataset,extreme_test_X)
	np.save('/user/work/al18709/tc_data_%s_flipped_40/extreme_test_y.npy' % dataset,extreme_test_y)
	np.save('/user/work/al18709/tc_data_%s_flipped_40/extreme_valid_X.npy' % dataset,extreme_valid_X)
	np.save('/user/work/al18709/tc_data_%s_flipped_40/extreme_valid_y.npy' % dataset,extreme_valid_y)

