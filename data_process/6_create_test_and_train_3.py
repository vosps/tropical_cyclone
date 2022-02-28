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
		if glob.glob('/user/work/al18709/tc_Xy_%s/X_%s.npy' % (dataset,tc)) == []:
			continue
		# print('/user/work/al18709/tc_Xy/X_%s.npy' % tc)
		X = np.load('/user/work/al18709/tc_Xy_%s/X_%s.npy' % (dataset,tc),allow_pickle = True)
		
		# isolate extreme TCs
		max_rain = np.max(X)
		total_rain = np.sum(X,axis=(1,2))
		max_total_rain = np.max(total_rain)
		
		# append to lists
		max_rains.append(max_rain)
		max_total_rains.append(max_total_rain)
		
	
	return max_rains,max_total_rains


def plot_histogram(ax,max_rains,colour):
	"""
	This function plots a histogram of the set in question
	"""
	# ax = sns.histplot(data=penguins, x="flipper_length_mm", hue="species", element="step")
	return sns.histplot(ax=ax,data=max_rains, stat="density",bins=20, fill=True,color=colour,element='step')
	

def create_set(tcs,datset='imerg'):	
	# initialise arrays
	n_tcs = len(tcs)
	print("n_tcs = ",n_tcs)
	set_X = np.zeros((1,10,10))
	set_y = np.zeros((1,100,100))
	set_meta = pd.DataFrame()

	# loop through each tc
	for i,tc in enumerate(tcs):
		# if glob.glob('/user/work/al18709/tc_Xy/%s/y_%s.npy' % (dataset,tc)) == []: # TODO: this directory doesn't have much in it
		# 	continue
		# y = np.load('/user/work/al18709/tc_Xy/%s/y_%s.npy' % (dataset,tc),allow_pickle = True)
		# X = np.load('/user/work/al18709/tc_Xy/%s/X_%s.npy' % (dataset,tc),allow_pickle = True)
		# meta = pd.read_csv('/user/work/al18709/tc_Xy/%s/meta_%s.npy' % (dataset,tc))
		if glob.glob('/user/work/al18709/tc_Xy/y_%s.npy' % tc) == []: # TODO: this directory doesn't have much in it
			continue
		y = np.load('/user/work/al18709/tc_Xy/y_%s.npy' % tc,allow_pickle = True)
		X = np.load('/user/work/al18709/tc_Xy/X_%s.npy' % tc,allow_pickle = True)
		meta = pd.read_csv('/user/work/al18709/tc_Xy/meta_%s.csv' % tc)
		print(meta)
		set_X = np.vstack((set_X,X))
		set_y = np.vstack((set_y,y))
		set_meta = set_meta.append(meta)
	print(set_meta)
	return set_X[1:,:,:],set_y[1:,:,:],set_meta

# generate list of sids
tc_dir = '/user/work/al18709/tropical_cyclones/mswep/*.nc'
filepaths = glob.glob(tc_dir)
print('number of filepaths = ',len(filepaths))

# group by tc sid number
regex = r"/user/work/al18709/tropical_cyclones/mswep/.+?_(.+?)_.*?.nc"
keyf = lambda text: (re.findall(regex, text)+ [text])[0]
sids = [gr for gr, items in groupby(sorted(filepaths), key=keyf)]
print('number of sids = ',len(sids))
# pick random 

# find most extreme tcs
tcs = []
max_rains = []

# define which dataset to look at
dataset = 'mswep'

# loop through each tc
for tc in sids:
	

	if glob.glob('/user/work/al18709/tc_Xy/X_%s.npy' % tc) != []:
		mswep = np.load('/user/work/al18709/tc_Xy/X_%s.npy' % tc,allow_pickle = True)
		# isolate extreme based on mswep
		max_rain = np.max(mswep)
	
		tcs.append(tc)
		max_rains.append(max_rain)
	
	else:
		continue

		# if glob.glob('/user/work/al18709/tc_Xy/%s/X_%s.npy' % (dataset,tc)) == []:
		# 	continue
		# 	X = np.load('/user/work/al18709/tc_Xy/%s/X_%s.npy' % (dataset,tc),allow_pickle = True)

	

	# append to lists
	

print(len(max_rains))
print('len sids = ',len(sids))


max_idx = list(np.argpartition(max_rains, -100)[-100:])
extreme_tcs_test = [tcs[i] for i in max_idx]
max_idx = list(np.argpartition(max_rains, -100)[-201:-101])
extreme_tcs_valid = [tcs[i] for i in max_idx]

# print(extreme_tcs_test)
# print(extreme_tcs_valid)
extreme_tcs = extreme_tcs_test + extreme_tcs_valid
# print(extreme_tcs)

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
print("plotting histograms")
max_rain_train,max_total_rain_train = get_max_rain(train_tcs)
max_rain_valid,max_total_rain_valid = get_max_rain(valid_tcs)
max_rain_test,max_total_rain_test = get_max_rain(test_tcs)
max_rain_extreme_test,max_total_rain_extreme_test = get_max_rain(extreme_tcs_test)
max_rain_extreme_valid,max_total_rain_extreme_valid = get_max_rain(extreme_tcs_valid)

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
# fig, axs = plt.subplot_mosaic([['a.', 'b.'], ['c.', 'd.']],
#                               constrained_layout=True,
# 							  figsize=(10, 10), 
# 							  sharey=True)


plot_histogram(axes[0,0],max_rain_train,'#dc98a8')
axes[0,0].set_title('Train')
axes[0,0].set_ylim([0, 0.175])
plot_histogram(axes[0,1],max_rain_valid,'#b5a1e2')
axes[0,1].set_title('Validation')
plot_histogram(axes[1,0],max_rain_test,'#80c2de')
axes[1,0].set_title('Test')
plot_histogram(axes[1,1],max_rain_extreme_test,'#85ceb5')
axes[1,1].set_title('Extreme Test')
plot_histogram(axes[0,2],max_rain_extreme_valid,'#85ceb5')
axes[0,2].set_title('Extreme Validation')

plt.savefig('figs/peak_histogram.png',bbox_inches='tight')
plt.clf()

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True,sharex=True)
plot_histogram(axes[0,0],max_total_rain_train,'#dc98a8')
axes[0,0].set_title('Train')
axes[0,0].set_ylim([0, 0.007])
plot_histogram(axes[0,1],max_total_rain_valid,'#b5a1e2')
axes[0,1].set_title('Validation')
plot_histogram(axes[1,0],max_total_rain_test,'#80c2de')
axes[1,0].set_title('Test')
plot_histogram(axes[1,1],max_total_rain_extreme_test,'#85ceb5')
axes[1,1].set_title('Extreme Test')
plot_histogram(axes[0,2],max_total_rain_extreme_valid,'#85ceb5')
axes[0,2].set_title('Extreme Validation')

plt.savefig('figs/total_histogram.png',bbox_inches='tight')



# index the relevant arrays
valid_X,valid_y,valid_meta = create_set(valid_tcs)
train_X,train_y,train_meta = create_set(train_tcs)
test_X,test_y,test_meta = create_set(test_tcs)
extreme_test_X,extreme_test_y,extreme_test_meta = create_set(extreme_tcs_test)
extreme_valid_X,extreme_valid_y,extreme_valid_meta = create_set(extreme_tcs_valid)

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


np.save('/user/work/al18709/tc_data/valid_X.npy',valid_X)
np.save('/user/work/al18709/tc_data/valid_y.npy',valid_y)
valid_meta.to_csv('/user/work/al18709/tc_data/valid_meta.csv')
np.save('/user/work/al18709/tc_data/train_X.npy',train_X)
np.save('/user/work/al18709/tc_data/train_y.npy',train_y)
train_meta.to_csv('/user/work/al18709/tc_data/train_meta.csv')
np.save('/user/work/al18709/tc_data/test_X.npy',test_X)
np.save('/user/work/al18709/tc_data/test_y.npy',test_y)
test_meta.to_csv('/user/work/al18709/tc_data/test_meta.csv')
np.save('/user/work/al18709/tc_data/extreme_test_X.npy',extreme_test_X)
np.save('/user/work/al18709/tc_data/extreme_test_y.npy',extreme_test_y)
extreme_test_meta.to_csv('/user/work/al18709/tc_data/extreme_test_meta.csv')
np.save('/user/work/al18709/tc_data/extreme_valid_X.npy',extreme_valid_X)
np.save('/user/work/al18709/tc_data/extreme_valid_y.npy',extreme_valid_y)
extreme_valid_meta.to_csv('/user/work/al18709/tc_data/extreme_valid_meta.csv')





