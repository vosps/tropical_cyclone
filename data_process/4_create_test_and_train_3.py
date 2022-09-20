"""
	5. Configure the test, train and cross validation sets

	train : 60 % of data
		all data not selected to be in test or cross val
	cross_val : 20 % of data
		random selection of tcs in chunks of 10 timesteps
	test : 20 % of data
		random selection of tcs in chunks of 10 timesteps
	extreme : 
		top the top 100 of most heaviest rainfall tcs based on highest value in X

input: each tc X and y file
output: numpy arrays for each set
	
"""

# TODO: how to select same set of TCs for both datasets? especially when looking at most extreme. Define most extreme by MSWEP

# import modules
import random
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from itertools import groupby
import re
from numpy.random import default_rng
import seaborn as sns
sns.set_style("white")

def get_max_rain(tcs,dataset='imerg'):
	"""
	This function gets the max rain from a list of tc sids
	"""

	max_rains = []
	max_total_rains = []

	# loop through each tc
	for tc in tcs:
		if glob.glob('/user/work/al18709/tc_Xy/X_%s.npy' % tc) == []:
			continue
		X = np.load('/user/work/al18709/tc_Xy/X_%s.npy' % tc,allow_pickle = True)
		
		# isolate extreme TCs
		max_rain = np.max(X)
		total_rain = np.sum(X,axis=(1,2))
		max_total_rain = np.max(total_rain)
		
		# append to lists
		max_rains.append(max_rain)
		max_total_rains.append(max_total_rain)

	return max_rains,max_total_rains

	

def create_set(tcs,dataset='imerg',resolution=100):	
	# initialise arrays
	n_tcs = len(tcs)
	print("n_tcs = ",n_tcs)
	if resolution == 40:
		set_X = np.zeros((1,40,40))
	elif dataset == 'mswep':
		set_X = np.zeros((1,10,10))
	elif dataset == 'era5':
		set_X = np.zeros((1,40,40))
	set_y = np.zeros((1,100,100))
	set_meta = pd.DataFrame()

	# loop through each tc
	for i,tc in enumerate(tcs):
		if glob.glob('/user/work/al18709/tc_Xy/y_%s.npy' % tc) == []: # TODO: this directory doesn't have much in it
			continue
		if resolution == 40:
			y = np.load('/user/work/al18709/tc_Xy_40/y_%s.npy' % tc,allow_pickle = True)
			X = np.load('/user/work/al18709/tc_Xy_40/X_%s.npy' % tc,allow_pickle = True)
		elif dataset == 'mswep':
			y = np.load('/user/work/al18709/tc_Xy/y_%s.npy' % tc,allow_pickle = True)
			X = np.load('/user/work/al18709/tc_Xy/X_%s.npy' % tc,allow_pickle = True)
		else:
			y = np.load('/user/work/al18709/tc_Xy_%s_40/y_%s.npy' % (dataset,tc),allow_pickle = True)
			X = np.load('/user/work/al18709/tc_Xy_%s_40/X_%s.npy' % (dataset,tc),allow_pickle = True)
		meta = pd.read_csv('/user/work/al18709/tc_Xy/meta_%s.csv' % tc)
		print(meta)
		set_X = np.vstack((set_X,X))
		set_y = np.vstack((set_y,y))
		set_meta = set_meta.append(meta)
	print(set_meta)
	return set_X[1:,:,:],set_y[1:,:,:],set_meta

# define which dataset to look at
dataset = 'mswep'
# dataset = 'mswep_extend'
# dataset = 'era5'
resolution = 100

# generate list of sids
tc_dir = '/user/work/al18709/tropical_cyclones/%s/*.nc' % dataset
filepaths = glob.glob(tc_dir)
print('number of filepaths = ',len(filepaths))



# group by tc sid number
if dataset == 'mswep':
	regex = r"/user/work/al18709/tropical_cyclones/mswep/.+?_(.+?)_.*?.nc"
elif dataset == 'mswep_extend':
	regex = r"/user/work/al18709/tropical_cyclones/mswep_extend/.+?_(.+?)_.*?.nc"
elif dataset == 'era5':
	regex = r"/user/work/al18709/tropical_cyclones/era5/.+?_(.+?)_.*?.nc"
keyf = lambda text: (re.findall(regex, text)+ [text])[0]
sids = [gr for gr, items in groupby(sorted(filepaths), key=keyf)]
print('number of sids = ',len(sids))
# pick random 

# find most extreme tcs
tcs = []
max_rains = []



# loop through each tc
for tc in sids:
	if dataset == 'mswep':
		fp = '/user/work/al18709/tc_Xy/X_%s.npy' % tc
	elif dataset == 'mswep_extend':
		fp = '/user/work/al18709/tc_Xy_mswep_extend/X_%s.npy' % tc
	elif dataset == 'era5':
		fp = '/user/work/al18709/tc_Xy_era5_40/X_%s.npy' % tc

	if glob.glob(fp) != []:
		data = np.load(fp,allow_pickle=True)
		# isolate extreme based on mswep
		max_rain = np.max(data)
	
		tcs.append(tc)
		max_rains.append(max_rain)
	
	else:
		continue

	

print(len(max_rains))
print('len sids = ',len(sids))


max_idx = list(np.argpartition(max_rains, -100)[-100:])
extreme_tcs_test = [tcs[i] for i in max_idx]
max_idx = list(np.argpartition(max_rains, -100)[-200:-100])
extreme_tcs_valid = [tcs[i] for i in max_idx]

extreme_tcs = extreme_tcs_test + extreme_tcs_valid

# remove extreme tcs from tc list, sort list so random workds
tcs = sorted(list(set(sids).difference(set(extreme_tcs))))

# set random seed and generate random indices
random.seed(22)
n_tc = len(tcs)
n_20 = int(n_tc*0.2)

# find valid, train and test
valid_tcs = random.sample(tcs,n_20)
tcs = sorted(list(set(tcs).difference(set(valid_tcs))))

test_tcs = random.sample(tcs,n_20)
train_tcs = sorted(list(set(tcs).difference(set(test_tcs))))

# check the numbers
print(len(valid_tcs))
print(len(test_tcs))
print(len(train_tcs))
print(len(extreme_tcs))

# plot histograms
# print("plotting histograms")
# max_rain_train,max_total_rain_train = get_max_rain(train_tcs)
# max_rain_valid,max_total_rain_valid = get_max_rain(valid_tcs)
# max_rain_test,max_total_rain_test = get_max_rain(test_tcs)
# max_rain_extreme_test,max_total_rain_extreme_test = get_max_rain(extreme_tcs_test)
# max_rain_extreme_valid,max_total_rain_extreme_valid = get_max_rain(extreme_tcs_valid)


# print shapes
print(len(valid_tcs))
print(len(train_tcs))
print(len(test_tcs))
print(len(extreme_tcs_test))
print(len(extreme_tcs_valid))


# index the relevant arrays
valid_X,valid_y,valid_meta = create_set(valid_tcs,dataset=dataset,resolution=resolution)
train_X,train_y,train_meta = create_set(train_tcs,dataset=dataset,resolution=resolution)
test_X,test_y,test_meta = create_set(test_tcs,dataset=dataset,resolution=resolution)
extreme_test_X,extreme_test_y,extreme_test_meta = create_set(extreme_tcs_test,dataset=dataset,resolution=resolution)
extreme_valid_X,extreme_valid_y,extreme_valid_meta = create_set(extreme_tcs_valid,dataset=dataset,resolution=resolution)

# print shapes
print(valid_X.shape)
print(valid_y.shape)
print('meta shape',valid_meta.shape)
print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)
print(extreme_test_X.shape)
print(extreme_test_y.shape)
print(extreme_valid_X.shape)
print(extreme_valid_y.shape)
# exit()
if (resolution == 40) or (dataset=='era5'):
	np.save('/user/work/al18709/tc_data_%s_40/valid_X.npy' % dataset,valid_X)
	np.save('/user/work/al18709/tc_data_%s_40/valid_y.npy' % dataset,valid_y)
	valid_meta.to_csv('/user/work/al18709/tc_data_%s_40/valid_meta.csv' % dataset)
	np.save('/user/work/al18709/tc_data_%s_40/train_X.npy' % dataset,train_X)
	np.save('/user/work/al18709/tc_data_%s_40/train_y.npy' % dataset,train_y)
	train_meta.to_csv('/user/work/al18709/tc_data_%s_40/train_meta.csv' % dataset)
	np.save('/user/work/al18709/tc_data_%s_40/test_X.npy' % dataset,test_X)
	np.save('/user/work/al18709/tc_data_%s_40/test_y.npy' % dataset,test_y)
	test_meta.to_csv('/user/work/al18709/tc_data_%s_40/test_meta.csv' % dataset)
	np.save('/user/work/al18709/tc_data_%s_40/extreme_test_X.npy' % dataset,extreme_test_X)
	np.save('/user/work/al18709/tc_data_%s_40/extreme_test_y.npy' % dataset,extreme_test_y)
	extreme_test_meta.to_csv('/user/work/al18709/tc_data_%s_40/extreme_test_meta.csv' % dataset)
	np.save('/user/work/al18709/tc_data_%s_40/extreme_valid_X.npy' % dataset,extreme_valid_X)
	np.save('/user/work/al18709/tc_data_%s_40/extreme_valid_y.npy' % dataset,extreme_valid_y)
	extreme_valid_meta.to_csv('/user/work/al18709/tc_data_%s_40/extreme_valid_meta.csv' % dataset)
else:
	np.save('/user/work/al18709/tc_data_%s/valid_X.npy' % dataset,valid_X)
	np.save('/user/work/al18709/tc_data_%s/valid_y.npy' % dataset,valid_y)
	valid_meta.to_csv('/user/work/al18709/tc_data_%s/valid_meta.csv' % dataset)
	np.save('/user/work/al18709/tc_data_%s/train_X.npy' % dataset,train_X)
	np.save('/user/work/al18709/tc_data_%s/train_y.npy' % dataset,train_y)
	train_meta.to_csv('/user/work/al18709/tc_data_%s/train_meta.csv' % dataset)
	np.save('/user/work/al18709/tc_data_%s/test_X.npy' % dataset,test_X)
	np.save('/user/work/al18709/tc_data_%s/test_y.npy' % dataset,test_y)
	test_meta.to_csv('/user/work/al18709/tc_data_%s/test_meta.csv' % dataset)
	np.save('/user/work/al18709/tc_data_%s/extreme_test_X.npy' % dataset,extreme_test_X)
	np.save('/user/work/al18709/tc_data_%s/extreme_test_y.npy' % dataset,extreme_test_y)
	extreme_test_meta.to_csv('/user/work/al18709/tc_data_%s/extreme_test_meta.csv' % dataset)
	np.save('/user/work/al18709/tc_data_%s/extreme_valid_X.npy' % dataset,extreme_valid_X)
	np.save('/user/work/al18709/tc_data_%s/extreme_valid_y.npy' % dataset,extreme_valid_y)
	extreme_valid_meta.to_csv('/user/work/al18709/tc_data_%s/extreme_valid_meta.csv' % dataset)





