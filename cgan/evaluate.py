"""
This script evaluates the model output against a series of metrics and plots them

https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490

"""


import numpy as np
import seaborn as sns
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import colors
import pysteps.verification.spatialscores as spatialscores
from sklearn.metrics import mean_squared_error
from itertools import groupby
from scipy import stats
from netCDF4 import Dataset
import re
import glob
import pysal as ps
from pysal.esda.getisord import G
from itertools import compress
import skgstat as skg
import pandas as pd
sns.set_style("white")

def generate_tcs():
        tc_dir = '/user/work/al18709/tropical_cyclones/*.nc'
        filepaths = glob.glob(tc_dir)
        # group by tc sid number
        regex = r"/user/work/al18709/tropical_cyclones/.+?_(.+?)_.*?.nc"
        keyf = lambda text: (re.findall(regex, text)+ [text])[0]
        sids = [gr for gr, items in groupby(sorted(filepaths), key=keyf)]
        tcs_X,tcs_y = create_set(sids)
        np.save('/user/work/al18709/tc_data/y.npy',tcs_y)

def regrid(array):
        hr_array = np.zeros((100,100))
        for i in range(10):
                for j in range(10):
                        i1 = i*10
                        i2 = (i+1)*10
                        j1 = j*10
                        j2 = (j+1)*10
                        hr_array[i1:i2,j1:j2] = array[i,j]
        return hr_array

def create_set(tcs):	
	# initialise arrays
	n_tcs = len(tcs)
	set_X = np.zeros((1,10,10))
	set_y = np.zeros((1,100,100))

	# loop through each tc
	for i,tc in enumerate(tcs):
		y = np.load('/user/work/al18709/tc_Xy/y_%s.npy' % tc,allow_pickle = True)
		X = np.load('/user/work/al18709/tc_Xy/X_%s.npy' % tc,allow_pickle = True)
		set_X = np.vstack((set_X,X))
		set_y = np.vstack((set_y,y))
	return set_X[1:,:,:],set_y[1:,:,:]


def plot_predictions(real,pred,inputs):
        real[real<=0.1] = np.nan
        pred[pred<=0.1] = np.nan
        # inputs = regrid(inputs[99])
        inputs[inputs<=0.1] = np.nan
        n = 4
        m = 3
        fig, axes = plt.subplots(n, m, figsize=(5*m, 5*n), sharey=True)
        range_ = (-5, 20)
        if mode == 'gcm':
                range_ = (-5,30)

        storms = [102,260,450,799]
        if mode == 'gcm':
                storms = [0,4,2,3,4]
        axes[0,0].set_title('Real')
        axes[0,1].set_title('Pred')
        axes[0,2].set_title('Input')
        for i in range(n):
                j = 0
                storm = storms[i]
                print(storm)
                print(regrid(inputs[storm]))
                axes[i,j].imshow(real[storm], interpolation='nearest', norm=colors.Normalize(*range_), extent=None,cmap='Blues')
                axes[i,j+1].imshow(pred[storm], interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
                axes[i,j+2].imshow(regrid(inputs[storm]), interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
                axes[i,j].set(xticklabels=[])
                axes[i,j].set(yticklabels=[])
                axes[i,j+1].set(xticklabels=[])
                axes[i,j+1].set(yticklabels=[])
                axes[i,j+2].set(xticklabels=[])
                axes[i,j+2].set(yticklabels=[])

        plt.savefig('logs/pred_images_%s.png' % mode,bbox_inches='tight')
        plt.clf()

def plot_histogram(ax,max_rains,colour,binwidth,alpha):
	"""
	This function plots a histogram of the set in question
	"""
	# ax = sns.histplot(data=penguins, x="flipper_length_mm", hue="species", element="step")
	return sns.histplot(ax=ax,data=max_rains, stat="density",binwidth=binwidth, fill=True,color=colour,element='step',alpha=alpha)

def generate_variogram(array):
        all_pairs,all_distances = get_coords(array)
        c = np.var(array.ravel())
        # print('all_pairs',all_pairs)
        distances = [1.0,2.0,5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,50.0,64.0,78.0,92.0,97.0]
        variogram = []
        for h in distances:
                pairs,N = find_pairs(all_pairs,all_distances,h)
                print('pairs',pairs)
                v = calc_variance(pairs,N)
                variogram.append(v)
                sc = 1 - v/c
        print(distances)
        print(variogram)
        return distances,variogram
        

def calc_variance(pairs,N):
        # print(pairs)
        z1_zh = [p[0] - p[1] for p in pairs]
        print('z1 - zh',z1_zh)
        gamma = 0.5*sum(np.square(z1_zh))/N
        return gamma

def get_coords(array):
        all_pairs = []
        all_distances = []
        flat_array = array.ravel()
        lats,lons = range(100),range(100)
        coords = [(i,j) for i in lats for j in lons]
        print(len(coords))
        
        for i in range(10000):
                for j in range(10000):
                        if j==i:
                                continue
                        else:
                                all_pairs.append((flat_array[i],flat_array[j]))
                                x = coords[i]
                                y = coords[j]
                                # print(x)
                                # print(y)
                                
                                dist = np.sqrt( (x[0] - y[0])**2 + (x[1] - y[1])**2 )
                                # print('distance: ',dist)
                                all_distances.append(dist)

        # print(all_pairs)
        # print('all_pairs',all_pairs)

        return all_pairs,all_distances


def find_pairs(all_pairs,all_distances,h):
        print('h',h)
        # print(all_distances)
        filter = np.array(all_distances) == h
        
        pairs = list(compress(all_pairs, filter))
        # pairs = all_pairs[idx]
        # print(pairs)
        return pairs,len(pairs)

def training_loss():
        return []
###
# begin evaluation
###

# set mode
mode = 'validation'
generate_tcs = False

# load datasets
real = np.load('/user/home/al18709/work/cgan_predictions/%s_real.npy' % mode)[0][:,:,:,0]
pred = np.load('/user/home/al18709/work/cgan_predictions/%s_pred.npy' % mode)[0][:,:,:,0]
inputs = np.load('/user/home/al18709/work/cgan_predictions/%s_input.npy' % mode)[0][:,:,:,0]
dtype={'gen_loss': 'float','disc_loss': 'float','training_samples': 'int'}
loss_log = pd.read_csv('/user/home/al18709/work/cgan_results/logs/log.txt', sep=",", header=None,dtype=dtype)[1:]
loss_log.columns = ['training_samples','disc_loss','disc_loss_real','disc_loss_fake','disc_loss_gp','gen_loss']
# loss_log = loss_log.convert_dtypes()
loss_log['training_samples'] = pd.to_numeric(loss_log['training_samples'],errors='coerce')
loss_log['gen_loss'] = pd.to_numeric(loss_log['gen_loss'],errors='coerce')
loss_log['disc_loss'] = pd.to_numeric(loss_log['disc_loss'],errors='coerce')
loss_log['disc_loss_real'] = pd.to_numeric(loss_log['disc_loss_real'],errors='coerce')
loss_log['disc_loss_fake'] = pd.to_numeric(loss_log['disc_loss_fake'],errors='coerce')
loss_log['disc_loss_gp'] = pd.to_numeric(loss_log['disc_loss_gp'],errors='coerce')

plt.plot(loss_log['training_samples'],loss_log['gen_loss'])
plt.plot(loss_log['training_samples'],loss_log['disc_loss_real'])
plt.plot(loss_log['training_samples'],loss_log['disc_loss_fake'])
# plt.plot(loss_log['training_samples'],loss_log['disc_loss'])
plt.savefig('figs/loss_log.png',bbox_inches='tight')
# plt.legend(labels=['generator loss','discriminator loss real','discriminator loss fake','disc loss'])
plt.legend(labels=['generator loss','critic real','critic loss fake'])
print(loss_log)
print(loss_log.dtypes)


# calculate spatial autocorrelation
"""
print('making variogram...')
distances, variogram = generate_variogram(real[0])
plt.plot(distances,variogram,color='#b5a1e2')
distances, variogram = generate_variogram(pred[0])
plt.plot(distances,variogram,color='#dc98a8')
plt.xlabel('Distance')
plt.ylabel('Semivariance')
plt.legend(labels=['real','pred'])
plt.savefig('logs/variogram.png')
plt.clf()
"""

# w = ps.lat2W(real[0].shape[0],real[0].shape[1])
# np.random.seed(12345)
# print(ps.Gamma(real[0],w).g)
# mi = ps.Moran(real[0], w)
# print('real mi = ',mi.I)
# print(mi.p_norm)
# print(V)
# plt = V.plot()
# plt.savefig('logs/variogram.png')

# w = ps.lat2W(pred[0].shape[0],pred[0].shape[1])
# np.random.seed(12345)
# print(ps.Gamma(pred[0],w).g)

# mi = ps.Moran(pred[0], w, two_tailed=False)
# print('pred mi = ',mi.I)
# print(mi.p_norm)
# print("mode = ",mode)
# print("number of storms: ", real.shape[0])
# print(inputs.shape)
# print(pred.shape)
# print(real.shape)

# initiate variables
nstorms,nlats,nlons = real.shape
fss_scores = []
mse_90_scores = []
accumulated_preds = []
accumulated_reals = []
peak_preds = []
peak_reals = []
peak_inputs = []
# actual_valid = np.load('/user/work/al18709/tc_data/extreme_valid_y.npy')
actual_reals = []

# calculate fss scores
print("calculating fss scores... ")
for i in range(nstorms):
        fss = spatialscores.fss(pred[i],real[i],0.1,4) #to not get nan have to filter out low values?
        fss_scores.append(fss)
      
        accumulated_pred = np.sum(pred[i])
        accumulated_real = np.sum(real[i])

        accumulated_preds.append(accumulated_pred/1000)
        accumulated_reals.append(accumulated_real/1000)

        peak_pred = np.max(pred[i])
        peak_real = np.max(real[i])
        peak_input = np.max(inputs[i])
        # actual_peak = np.max(actual_valid[i])

        peak_preds.append(peak_pred)
        peak_reals.append(peak_real)
        peak_inputs.append(peak_input)
        # actual_reals.append(actual_peak)

        # peak_pred
# print([i for i in peak_reals if i >= 110.])
# print([i for i in actual_reals if i >= 110.])
# print([i for i in peak_preds if i >= 110.])
# print(peak_inputs)
print('fss scores: ',np.mean(fss_scores))
print('pred',pred)
print('real',real)
# plot predictions and print score
if mode == 'gcm':
        plot_predictions(pred,pred,inputs)
else:
        plot_predictions(real,pred,inputs)



# plot accumulated histograms
fig, ax = plt.subplots()
plot_histogram(ax,accumulated_reals,'#b5a1e2',1,0.7)
plot_histogram(ax,accumulated_preds,'#dc98a8',1,0.5)
ax.set_xlabel('Accumulated rainfall (m)')
plt.legend(labels=['real','pred'])
plt.savefig('logs/histogram_accumulated_%s.png' % mode)
plt.clf()
ks_accumulated = stats.ks_2samp(accumulated_reals, accumulated_preds)
print('Ks test for accumulated rainfall: ',ks_accumulated)

# plot peak histograms
fig, ax = plt.subplots()
plot_histogram(ax,peak_reals,'#b5a1e2',5,0.7)
plot_histogram(ax,peak_preds,'#dc98a8',5,0.5)
ax.set_xlabel('Peak rainfall (mm)')
plt.legend(labels=['real','pred'])
plt.savefig('logs/histogram_peak_%s.png' % mode)
ks_peak = stats.ks_2samp(peak_reals, peak_preds)
print('ks test for peak rainfall: ',ks_peak)
plt.clf()



"""
# generate list of sids
# if generate_tcs == True:
#         generate_tcs()
# else:
#         tcs_y = np.load('/user/work/al18709/tc_data/y.npy')

# print('tcs_y shape: ',tcs_y.shape)
# nstorms,_,_ = tcs_y.shape
# peaks = []
# for i in range(nstorms):
#         print(i,end='\r')
#         peak = np.max(tcs_y[i])
#         peaks.append(peak)

# save the max peak
# filepaths = glob.glob('/bp1store/geog-tropical/data/Obs/IMERG/half_hourly/final/*.HDF5')
# filepaths = glob.glob('/bp1store/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/*.nc')
# max_peaks = []
# for filepath in filepaths[0:5000]:
#         d = Dataset(filepath, 'r')
#         data = d.variables['precipitation']
#         max_peak = np.max(np.array(data))
#         max_peaks.append(max_peak)
#         d.close()
        
# np.save('/user/work/al18709/tc_data/max_peaks_mswep.npy',max_peaks)

# load the max  
mswep_max = np.load('/user/work/al18709/tc_data/max_peaks_mswep.npy')
mswep_max = mswep_max[mswep_max<=500.0]
imerg_max = np.load('/user/work/al18709/tc_data/max_peaks.npy')
print(max_peaks)



"""
