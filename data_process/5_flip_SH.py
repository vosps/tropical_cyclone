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
		plot_tc(y[i],'normal')
		
		# era5 doesn't need to be flipped because it's already correct
		if dataset == 'mswep':
			X_flipped = flip(X[i])
			X[i] = X_flipped

		y_flipped = flip(y[i])
		y[i] = y_flipped

	# flip the y values in the hr era5 dataset because they rotate differently
	if dataset == 'era5':
		for i in sh_indices:
			y_flipped = flip(y[i])
			y[i] = y_flipped
		
		
		# plot_tc(y[i],'flipped')
	
	return X,y


dataset = 'era5'
resolution = 40
dataset = 'mswep'
# dataset = 'mswep_extend'
resolution = 100

def save_flipped(grouped_sids):
	# print(sid)
	for sid in grouped_sids:
		if 'NAMED' in sid:
			continue
		if 'extreme_valid' in sid:
			continue
		X = np.load('/user/work/al18709/tc_Xy_extend/X_%s.npy' % sid)
		y = np.load('/user/work/al18709/tc_Xy_extend/y_%s.npy' % sid)
		meta = pd.read_csv('/user/work/al18709/tc_Xy_extend/meta_%s.csv' % sid)
		X,y = find_and_flip(X,y,meta)
		print(X.shape)
		np.save('/user/work/al18709/tc_data_%s_flipped/X_%s.npy' % (dataset,sid),X)
		np.save('/user/work/al18709/tc_data_%s_flipped/y_%s.npy' % (dataset,sid),y)
		meta.to_csv('/user/work/al18709/tc_data_%s_flipped/meta_%s.csv' % (dataset,sid))


def process(filepaths):
	print('doing process...')
	res = save_flipped(filepaths)
	return res



if dataset == 'mswep_extend':

	n_processes = 64

	files = glob.glob('/user/work/al18709/tc_Xy_extend/X_*.npy')
	sids = [file[34:47] for file in files]
	tc_split = np.array(np.array_split(sids, n_processes))
	p = Pool(processes=n_processes)
	pool_results = p.map(process, tc_split)
	p.close()
	p.join()	


elif resolution == 100:
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


# print('valid')
# valid_X,valid_y = find_and_flip(valid_X,valid_y,valid_meta,dataset=dataset)
# np.save('/user/work/al18709/tc_data_%s_flipped_40/valid_X.npy' % dataset,valid_X)
# np.save('/user/work/al18709/tc_data_%s_flipped_40/valid_y.npy' % dataset,valid_y)

# print('train')
# train_X,train_y = find_and_flip(train_X,train_y,train_meta,dataset=dataset)
# np.save('/user/work/al18709/tc_data_%s_flipped_40/train_X.npy' % dataset,train_X)
# np.save('/user/work/al18709/tc_data_%s_flipped_40/train_y.npy' % dataset,train_y)

# print('test')
# test_X,test_y = find_and_flip(test_X,test_y,test_meta,dataset=dataset)
# # np.save('/user/work/al18709/tc_data_%s_flipped_40/test_X.npy' % dataset,test_X)
# # np.save('/user/work/al18709/tc_data_%s_flipped_40/test_y.npy' % dataset,test_y)
# np.save('/user/work/al18709/tc_data_flipped/test_X.npy',test_X)
# np.save('/user/work/al18709/tc_data_flipped/test_y.npy',test_y)

print('extreme test')
print(extreme_test_X.shape)
print(extreme_test_y.shape)
extreme_test_X,extreme_test_y = find_and_flip(extreme_test_X,extreme_test_y,extreme_test_meta,dataset=dataset)
# np.save('/user/work/al18709/tc_data_%s_flipped_40/extreme_test_X.npy' % dataset,extreme_test_X)
# np.save('/user/work/al18709/tc_data_%s_flipped_40/extreme_test_y.npy' % dataset,extreme_test_y)
print(extreme_test_X.shape)
print(extreme_test_y.shape)
np.save('/user/work/al18709/tc_data_flipped/extreme_test_X.npy',extreme_test_X)
np.save('/user/work/al18709/tc_data_flipped/extreme_test_y.npy',extreme_test_y)

# print('extreme valid')
# extreme_valid_X,extreme_valid_y = find_and_flip(extreme_valid_X,extreme_valid_y,extreme_valid_meta,dataset=dataset)
# np.save('/user/work/al18709/tc_data_%s_flipped_40/extreme_valid_X.npy' % dataset,extreme_valid_X)
# np.save('/user/work/al18709/tc_data_%s_flipped_40/extreme_valid_y.npy' % dataset,extreme_valid_y)

exit()

if resolution == 100:
	np.save('/user/work/al18709/tc_data_%s_flipped/valid_X.npy' % dataset,valid_X)
	np.save('/user/work/al18709/tc_data_%s_flipped/valid_y.npy' % dataset,valid_y)
	np.save('/user/work/al18709/tc_data_%s_flipped/train_X.npy' % dataset,train_X)
	np.save('/user/work/al18709/tc_data_%s_flipped/train_y.npy' % dataset,train_y)
	np.save('/user/work/al18709/tc_data_%s_flipped/test_X.npy' % dataset,test_X)
	np.save('/user/work/al18709/tc_data_%s_flipped/test_y.npy' % dataset,test_y)
	np.save('/user/work/al18709/tc_data_%s_flipped/extreme_test_X.npy' % dataset,extreme_test_X)
	np.save('/user/work/al18709/tc_data_%s_flipped/extreme_test_y.npy' % dataset,extreme_test_y)
	np.save('/user/work/al18709/tc_data_%s_flipped/extreme_valid_X.npy' % dataset,extreme_valid_X)
	np.save('/user/work/al18709/tc_data_%s_flipped/extreme_valid_y.npy' % dataset,extreme_valid_y)
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

