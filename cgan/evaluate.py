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
sns.set_style("white")

def regrid(array):
        # print(array.shape)
        hr_array = np.zeros((100,100))
        # print(hr_array)
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
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 20), sharey=True)
        range_ = (-5, 20)
        c = axes[0,0].imshow(real[99], interpolation='nearest', norm=colors.Normalize(*range_), extent=None,cmap='Blues')
        axes[0,0].set_title('Real')
        axes[0,1].imshow(pred[99], interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
        axes[0,1].set_title('Pred')
        axes[0,2].imshow(regrid(inputs[99]), interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
        axes[0,2].set_title('Input')
        # plt.colorbar(c)
        axes[1,0].imshow(real[260], interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
        axes[1,1].imshow(pred[260], interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
        axes[1,2].imshow(regrid(inputs[260]), interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
        axes[2,0].imshow(real[450], interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
        axes[2,1].imshow(pred[450], interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
        axes[2,2].imshow(regrid(inputs[450]), interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
        axes[3,0].imshow(real[799], interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
        axes[3,1].imshow(pred[799], interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
        axes[3,2].imshow(regrid(inputs[799]), interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
        for i in range(4):
                for j in range(3):
                        axes[i,j].set(xticklabels=[])
                        axes[i,j].set(yticklabels=[])
        plt.savefig('logs/pred_images_%s.png' % mode,bbox_inches='tight')
        plt.clf()

def plot_histogram(ax,max_rains,colour,binwidth,alpha):
	"""
	This function plots a histogram of the set in question
	"""
	# ax = sns.histplot(data=penguins, x="flipper_length_mm", hue="species", element="step")
	return sns.histplot(ax=ax,data=max_rains, stat="density",binwidth=binwidth, fill=True,color=colour,element='step',alpha=alpha)

mode = 'validation'
generate_tcs = False
print(mode)
print('generate TCs? ', generate_tcs)
real = np.load('/user/home/al18709/work/cgan_predictions/%s_real.npy' % mode)[0][:,:,:,0]
pred = np.load('/user/home/al18709/work/cgan_predictions/%s_pred.npy' % mode)[0][:,:,:,0]
inputs = np.load('/user/home/al18709/work/cgan_predictions/%s_input.npy' % mode)[0][:,:,:,0]

# remove low rainfall
# real[real<=0.1] = np.nan
# pred[pred<=0.1] = np.nan

# initiate variables
nstorms,nlats,nlons = real.shape
fss_scores = []
mse_90_scores = []
accumulated_preds = []
accumulated_reals = []
peak_preds = []
peak_reals = []
peak_inputs = []
actual_valid = np.load('/user/work/al18709/tc_data/extreme_valid_y.npy')
actual_reals = []

# calculate scores
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
        actual_peak = np.max(actual_valid[i])

        peak_preds.append(peak_pred)
        peak_reals.append(peak_real)
        peak_inputs.append(peak_input)
        actual_reals.append(actual_peak)

        # peak_pred
print([i for i in peak_reals if i >= 110.])
print([i for i in actual_reals if i >= 110.])
# print([i for i in peak_preds if i >= 110.])
# print(peak_inputs)


# plot predictions
plot_predictions(real,pred,inputs)
print(np.mean(fss_scores))


# plot accumulated histograms
fig, ax = plt.subplots()
plot_histogram(ax,accumulated_reals,'#b5a1e2',1,0.7)
plot_histogram(ax,accumulated_preds,'#dc98a8',1,0.5)
ax.set_xlabel('Accumulated rainfall (m)')
plt.legend(labels=['real','pred'])
plt.savefig('logs/histogram_accumulated_%s.png' % mode)
plt.clf()
ks_accumulated = stats.ks_2samp(accumulated_reals, accumulated_preds)
print(ks_accumulated)

# plot peak histograms
fig, ax = plt.subplots()
plot_histogram(ax,peak_reals,'#b5a1e2',5,0.7)
plot_histogram(ax,peak_preds,'#dc98a8',5,0.5)
ax.set_xlabel('Peak rainfall (mm)')
plt.legend(labels=['real','pred'])
plt.savefig('logs/histogram_peak_%s.png' % mode)
ks_peak = stats.ks_2samp(peak_reals, peak_preds)
print(ks_peak)
plt.clf()

# generate list of sids
if generate_tcs == True:
        tc_dir = '/user/work/al18709/tropical_cyclones/*.nc'
        filepaths = glob.glob(tc_dir)
        # group by tc sid number
        regex = r"/user/work/al18709/tropical_cyclones/.+?_(.+?)_.*?.nc"
        keyf = lambda text: (re.findall(regex, text)+ [text])[0]
        sids = [gr for gr, items in groupby(sorted(filepaths), key=keyf)]
        tcs_X,tcs_y = create_set(sids)
        print(tcs_y.shape)
        np.save('/user/work/al18709/tc_data/y.npy',tcs_y)
else:
        tcs_y = np.load('/user/work/al18709/tc_data/y.npy')

print(tcs_y.shape)
nstorms,_,_ = tcs_y.shape
peaks = []
for i in range(nstorms):
        print(i,end='\r')
        peak = np.max(tcs_y[i])
        peaks.append(peak)



filepaths = glob.glob('/bp1store/geog-tropical/data/Obs/IMERG/half_hourly/final/*.HDF5')
filepaths = glob.glob('/bp1store/geog-tropical/data/Obs/MSWEP/3hourly/*.nc')

# print(filepaths)
# max_peaks = []
# for filepath in filepaths[0:5000]:
#         # print(filepath)
#         d = Dataset(filepath, 'r')
#         # print(d.variables)
#         data = d.variables['precipitation']
#         # print(data)
#         max_peak = np.max(np.array(data))
#         max_peaks.append(max_peak)
#         d.close()
        
# np.save('/user/work/al18709/tc_data/max_peaks_mswep.npy',max_peaks)      
max_peaks = np.load('/user/work/al18709/tc_data/max_peaks_mswep.npy')
max_peaks = max_peaks[max_peaks<=500.0] 
imerg = np.load('/user/work/al18709/tc_data/max_peaks.npy')

# plot peak histograms
fig, ax = plt.subplots()
# plot_histogram(ax,peak_reals,'#b5a1e2',5,0.7)
# plot_histogram(ax,peak_preds,'#dc98a8',5,0.5)
# plot_histogram(ax,peaks,'#85ceb5',5,0.5)
plot_histogram(ax,imerg,'#80c2de',5,0.5)
plot_histogram(ax,max_peaks,'#85ceb5',5,0.5)
ax.set_xlabel('Peak rainfall (mm)')
# plt.legend(labels=['real','pred','imerg max','mswep max'])
plt.legend(labels=['imerg','mswep'])
plt.savefig('logs/histogram_peak_og_2_%s.png' % mode)
ks_peak = stats.ks_2samp(peak_reals, peak_preds)
print(ks_peak)
plt.clf()

print(np.max(max_peaks))

