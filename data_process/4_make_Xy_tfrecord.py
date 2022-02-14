"""
4. Combine netcdf snapshots into an Xy Tfrecord. Each datapoint/image will have the following metadata

	SID
	Storm name
	datetime
	dataset (imerg/mswep)
	lat/lon of centre
	X array of shape : (n_timesteps, 10, 10)
	y array of shape : (n_timesteps,100,100)

saved in : /user/work/al18709/tc_data/

"""
import numpy as np
import tensorflow as tf
import glob

data = np.load('/user/home/al18709/work/tc_Xy/imerg/X_2014068S16169.npy')
print('data shape', data.shape)
time = -1
X = np.zeros((8,10,10))
for array in data:
	print(array.shape)
	time = time + 1
	meta = np.dtype(float, metadata={"dataset": "imerg","time": time})
	array.dtype=meta
	print(array.dtype.metadata)
	X[time,:,:] = array

meta = np.dtype(float, metadata={"dataset": "imerg","time": [0,1,2,3,4,5,6,7,8]})
X.dtype=meta
print(X.dtype.metadata)