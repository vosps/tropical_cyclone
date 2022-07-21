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
		plot_tc(y[i],'normal')
		X_flipped = flip(X[i])
		y_flipped = flip(y[i])
		X[i] = X_flipped
		y[i] = y_flipped
		# plot_tc(y[i],'flipped')
	
	return X,y




valid_X = np.load('/user/work/al18709/tc_data_mswep/valid_X.npy')
valid_y = np.load('/user/work/al18709/tc_data_mswep/valid_y.npy')
valid_meta = pd.read_csv('/user/work/al18709/tc_data_mswep/valid_meta.csv')
train_X = np.load('/user/work/al18709/tc_data_mswep/train_X.npy')
train_y = np.load('/user/work/al18709/tc_data_mswep/train_y.npy')
train_meta = pd.read_csv('/user/work/al18709/tc_data_mswep/train_meta.csv')
test_X = np.load('/user/work/al18709/tc_data_mswep/test_X.npy')
test_y = np.load('/user/work/al18709/tc_data_mswep/test_y.npy')
test_meta = pd.read_csv('/user/work/al18709/tc_data_mswep/test_meta.csv')
extreme_test_X = np.load('/user/work/al18709/tc_data_mswep/extreme_test_X.npy')
extreme_test_y = np.load('/user/work/al18709/tc_data_mswep/extreme_test_y.npy')
extreme_test_meta = pd.read_csv('/user/work/al18709/tc_data_mswep/extreme_test_meta.csv')
extreme_valid_X = np.load('/user/work/al18709/tc_data_mswep/extreme_valid_X.npy')
extreme_valid_y = np.load('/user/work/al18709/tc_data_mswep/extreme_valid_y.npy')
extreme_valid_meta = pd.read_csv('/user/work/al18709/tc_data_mswep/extreme_valid_meta.csv')

valid_X,valid_y = find_and_flip(valid_X,valid_y,valid_meta)
train_X,train_y = find_and_flip(train_X,train_y,train_meta)
test_X,test_y = find_and_flip(test_X,test_y,test_meta)
extreme_test_X,extreme_test_y = find_and_flip(extreme_test_X,extreme_test_y,extreme_test_meta)
extreme_valid_X,extreme_valid_y = find_and_flip(extreme_valid_X,extreme_valid_y,extreme_valid_meta)

np.save('/user/work/al18709/tc_data_flipped/valid_X.npy',valid_X)
np.save('/user/work/al18709/tc_data_flipped/valid_y.npy',valid_y)

np.save('/user/work/al18709/tc_data_flipped/train_X.npy',train_X)
np.save('/user/work/al18709/tc_data_flipped/train_y.npy',train_y)

np.save('/user/work/al18709/tc_data_flipped/test_X.npy',test_X)
np.save('/user/work/al18709/tc_data_flipped/test_y.npy',test_y)

np.save('/user/work/al18709/tc_data_flipped/extreme_test_X.npy',extreme_test_X)
np.save('/user/work/al18709/tc_data_flipped/extreme_test_y.npy',extreme_test_y)

np.save('/user/work/al18709/tc_data_flipped/extreme_valid_X.npy',extreme_valid_X)
np.save('/user/work/al18709/tc_data_flipped/extreme_valid_y.npy',extreme_valid_y)

