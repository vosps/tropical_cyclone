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
# sns.set_style("white")

def flip(tc):
	tc_flipped = np.flip(tc,axis=0)
	return tc_flipped

def plot_tc(tc,title):
	plt.imshow(tc)
	plt.savefig(title)
	plt.clf()


def find_and_flip(X,y,meta):
	print(X.shape)
	print(y.shape)
	print(meta)

	sh_indices = meta[meta['centre_lat'] < 0].index
	nh_indices = meta[meta['centre_lat'] > 0].index
	print(np.sum(meta['centre_lat'] < 0))
	print(np.sum(meta['centre_lat'] > 0))

	# flip the nh tcs so they are rotating anticlockwise (in the raw data array)
	# in mswep they can be plotted correctly with the mswep lats and lons, but the raw data shows them flipped as mswep has descending latitudes
	for i in nh_indices:
		print(i,end='\r')
		plot_tc(y[i],'normal')
		X_flipped = flip(X[i])
		y_flipped = flip(y[i])
		X[i] = X_flipped
		y[i] = y_flipped
		# plot_tc(y[i],'flipped')
	
	return X,y


dataset = 'era5'

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

valid_X,valid_y = find_and_flip(valid_X,valid_y,valid_meta)
train_X,train_y = find_and_flip(train_X,train_y,train_meta)
test_X,test_y = find_and_flip(test_X,test_y,test_meta)
extreme_test_X,extreme_test_y = find_and_flip(extreme_test_X,extreme_test_y,extreme_test_meta)
extreme_valid_X,extreme_valid_y = find_and_flip(extreme_valid_X,extreme_valid_y,extreme_valid_meta)

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

