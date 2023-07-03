import matplotlib
import seaborn as sns
import metpy
from matplotlib.colors import LinearSegmentedColormap
import metpy.plots.ctables
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.stats as stats
from scipy import interpolate
# from utils.evaluation import find_landfalling_tcs,tc_region,create_xarray,get_storm_coords

def make_cmap(high_vals=False,low_vals=False):
	precip_clevs = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 70, 100, 150]
	if high_vals==True:
		# precip_clevs = [0, 20, 25, 30, 40, 50, 70, 100, 125, 150, 175,200, 225, 250, 300, 350, 400, 500]
		precip_clevs = [0, 20, 40, 60, 80, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 500]
	if low_vals == True:
		precip_clevs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 19, 20, 22, 24, 26, 28, 30]
	precip_cmap = matplotlib.colors.ListedColormap(metpy.plots.ctables.colortables["precipitation"][:len(precip_clevs)-1], 'precipitation')
	precip_norm = matplotlib.colors.BoundaryNorm(precip_clevs, precip_cmap.N)

	tc_colours = [(255/255,255/255,255/255), # no rain
				(169/255, 209/255, 222/255), # drizzle 0-1
				(137/255, 190/255, 214/255), # drizzle 1-2
				(105/255, 160/255, 194/255), # drizzle 2-3
				(93/255, 168/255, 98/255), # drizzle 3-5
				(128/255, 189/255, 100/255), # very light rain 5-7
				(165/255, 196/255, 134/255), # light rain 7-10
				(233/255, 245/255, 105/255), # rain 10-15
				(245/255, 191/255, 105/255), # heavy rain 15-20
				(245/255, 112/255, 105/255), # heavier rain 20-25
				(245/255, 105/255, 149/255), # real heavy rain 25-30
				(240/255, 93/255, 154/255), # intense rain 30-40
				(194/255, 89/255, 188/255), # super intense rain 40-50
				(66/255, 57/255, 230/255), # insane amount of rain 50-70
				(24/255, 17/255, 153/255), # you do not want to be caught in this rain 70-100
				(9/255, 5/255, 87/255), # I can't belive the scle goes up this high 100-150
	]
	if high_vals == False:
		N = 16
	else:
		N=16
	precip_cmap = LinearSegmentedColormap.from_list('tc_colours',tc_colours,N=N) #increasing N makes it smoother
	precip_norm = matplotlib.colors.BoundaryNorm(precip_clevs, precip_cmap.N,extend='max')
	return precip_cmap,precip_norm


def make_anomaly_cmap(high_vals=False):
	precip_clevs = [-20,-15,-10,-5,0,5,10,15,20]
	
	precip_cmap = matplotlib.colors.ListedColormap(metpy.plots.ctables.colortables["precipitation"][:len(precip_clevs)-1], 'precipitation')
	precip_norm = matplotlib.colors.BoundaryNorm(precip_clevs, precip_cmap.N)

	tc_colours = [	(140/255, 112/255, 102/255),
					(171/255, 118/255, 99/255), # no rain
					(196/255, 160/255, 134/255),
					(228/255, 211/255, 187/255),
					
					(241/255, 238/255, 241/255), #0
					
					(155/255, 208/255, 209/255),
					(0/255, 121/255, 162/255),
					(0/255, 81/255, 162/255),
					(1/255, 53/255, 105/255)
				
	]
	if high_vals == False:
		N = 16
	else:
		N=16
	precip_cmap = LinearSegmentedColormap.from_list('tc_colours',tc_colours,N=N) #increasing N makes it smoother
	precip_norm = matplotlib.colors.BoundaryNorm(precip_clevs, precip_cmap.N)
	return precip_cmap,precip_norm

def power_spectrum_dB(img):
	fx = np.fft.fft2(img)
	fx = fx[:img.shape[0]//2, :img.shape[1]//2] # disgard half of it because you can derive one half from the other?
	px = abs(fx)**2 # get the size of the amplitudes
	return 10 * np.log10(px) # is this the rainfall normalisation step?

def power_spectrum_dB_2(img):
	fx = np.fft.fft2(img)
	# fx = fx[:img.shape[0]//2, :img.shape[1]//2] # disgard half of it because you can derive one half from the other?
	px = abs(fx)**2 # get the size of the amplitudes
	return px

def log_spectral_batch(batch1):
	lsd_batch = []
	for i in range(batch1.shape[0]):
		lsd = power_spectrum_dB_2(
				batch1[i, :, :])
		lsd_batch.append(lsd)
	return np.array(lsd_batch)

def plot_power_spectrum(spectral_density,plot=True):
    """
    spectral density should probably be a 100x100 image
    """
	# https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
    npix=100

    # return a one dimensional array containing the wave vectors for the numpy.fft.fftn call, in the correct order.
    kfreq = np.fft.fftfreq(npix) * npix
    
    # convert this to a two dimensional array matching the layout of the two dimensional Fourier image, we can use numpy.meshgrid
    kfreq2D = np.meshgrid(kfreq, kfreq)
    
    # Finally, we are not really interested in the actual wave vectors, but rather in their norm
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    # we no longer need the wave vector norms or Fourier image to be laid out as a two dimensional array, so we will flatten them
    knrm = knrm.flatten()
    fourier_amplitudes = spectral_density
    fourier_amplitudes = fourier_amplitudes.flatten()

    # bin the amplitudes in k space, we need to set up wave number bins
    # Note that the maximum wave number will equal half the pixel size of the image. 
    # This is because half of the Fourier frequencies can be mapped back to negative wave
    #  numbers that have the same norm as their positive counterpart.
    kbins = np.arange(0.5, npix//2+1, 1.)

    # The kbin array will contain the start and end points of all bins; the corresponding k values are the midpoints of these bins
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    # To compute the average Fourier amplitude (squared) in each bin, we can use scipy.stats
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)

    # Remember that we want the total variance within each bin. Right now, we only have the average power. To get the total power, we need to multiply with the volume in each bin (in 2D, this volume is actually a surface area)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    if plot == True:
        plt.loglog(kvals, Abins)
        plt.show
    else:
        return kvals,Abins
    
def calc_power(lsb_pred):
	kvals_list = []
	Abins_list = []

	for lsb in lsb_pred:
		kvals,Abins = plot_power_spectrum(lsb,plot=False)
		kvals_list.append(kvals)
		Abins_list.append(Abins)

	kvals = np.array(kvals_list)
	Abins = np.array(Abins_list)
	return kvals,Abins

def mean_power(kvals,Abins):

	xnew = np.sort(kvals.flatten())
	Y = np.zeros((Abins.shape[0],len(xnew)))
	for i,y in enumerate(Abins):
		x = kvals[i]	
		f = interpolate.interp1d(x, y,bounds_error=False)
		ynew = f(xnew)
		print(ynew.shape)
		print('xnew shape',xnew.shape)
		Y[i] = ynew
	print(xnew.shape)
	print(Y.shape)
	# split up to process the mean
	Y_mean_1 = np.nanmean(Y[0:5000],axis=0)
	Y_mean_2 = np.nanmean(Y[5000:10000],axis=0)
	Y_mean_3 = np.nanmean(Y[10000:],axis=0)
	Y_mean = np.nanmean([Y_mean_1,Y_mean_2,Y_mean_3],axis=0)
	print(Y_mean.shape)
	# kvals = x, Abins = Y_mean
	return(xnew,Y_mean)

def change_units(x):
	# npix * 1/k gives the units of pixels on the x axis
	# then we multiply by 10 because each pixel is 10km, the whole image is 1000km across
	# return 1000/x
	return x/1000

def radMask(index,radius,array):
  a,b = index
  nx,ny = array.shape
  y,x = np.ogrid[-a:nx-a,-b:ny-b]
  mask = x*x + y*y <= radius*radius

  return mask

def calc_ensemble_percentile(ensemble,analysis,percentiles):
	ensemble_p=np.zeros((20,len(percentiles)))
	print(ensemble.shape)
	for i in range(20):
		if analysis == 'Mean' or analysis == 'Mean | extremes':
			p = np.percentile(np.mean(np.mean(ensemble[:,:,:,i],axis=1),axis=1), percentiles)
		elif analysis == 'Peak' or analysis == 'Peak | extremes':
			p = np.percentile(np.max(np.max(ensemble[:,:,:,i],axis=1),axis=1), percentiles)
		elif analysis == 'Total' or analysis == 'Total | extremes':
			p = np.percentile(np.sum(np.sum(ensemble[:,:,:,i],axis=1),axis=1), percentiles)
		elif analysis == 'Core' or analysis == 'Core | extremes':
			mask=radMask((25,25),25,ensemble[0,:,:,0])
			n,_,_,_ = ensemble.shape
			mask = np.expand_dims(mask,axis=0)
			mask = np.repeat(mask,n,axis=0)
			p = np.percentile(np.ma.array(ensemble[:,:,:,i], mask=mask), percentiles)
		ensemble_p[i] = p
	ensemble_mean = np.mean(ensemble_p,axis=0)
	ensemble_std = np.std(ensemble_p,axis=0)
	ensemble_min = np.min(ensemble_p,axis=0)
	ensemble_max = np.max(ensemble_p,axis=0)
	return ensemble_mean,ensemble_min,ensemble_max

def plot_qq(ax,analysis,pred,pred_2,pred_og,real,cnn=False):
	percentiles = np.arange(0,100,0.01)

	if analysis == 'Mean' or analysis == 'Mean | extremes':
		dsrnngan_p,dsrnngan_min,dsrnngan_max = calc_ensemble_percentile(pred,analysis,percentiles)
		pred2_p,pred2_min,pred2_max = calc_ensemble_percentile(pred_2,analysis,percentiles)
		predog_p,predog_min,predog_max = calc_ensemble_percentile(pred_og,analysis,percentiles)
		real_p = np.percentile(np.mean(np.mean(real,axis=1),axis=1), percentiles)

	elif analysis == 'Peak' or analysis == 'Peak | extremes':
		dsrnngan_p,dsrnngan_min,dsrnngan_max = calc_ensemble_percentile(pred,analysis,percentiles)
		pred2_p,pred2_min,pred2_max = calc_ensemble_percentile(pred_2,analysis,percentiles)
		predog_p,predog_min,predog_max = calc_ensemble_percentile(pred_og,analysis,percentiles)
		real_p = np.percentile(np.max(np.max(real,axis=1),axis=1), percentiles)

	elif analysis == 'Total' or analysis == 'Total | extremes':
		dsrnngan_p,dsrnngan_min,dsrnngan_max = calc_ensemble_percentile(pred,analysis,percentiles)
		pred2_p,pred2_min,pred2_max = calc_ensemble_percentile(pred_2,analysis,percentiles)
		predog_p,predog_min,predog_max = calc_ensemble_percentile(pred_og,analysis,percentiles)
		real_p = np.percentile(np.sum(np.sum(real,axis=1),axis=1), percentiles)

	elif analysis == 'Core' or analysis == 'Core | extremes':
		mask=radMask((25,25),25,real[0,:,:,0])
		n,_,_,_ = real.shape
		mask = np.expand_dims(mask,axis=0)
		mask = np.repeat(mask,n,axis=0)
		dsrnngan_p,dsrnngan_min,dsrnngan_max = calc_ensemble_percentile(pred,analysis,percentiles)
		pred2_p,pred2_min,pred2_max = calc_ensemble_percentile(pred_2,analysis,percentiles)
		predog_p,predog_min,predog_max = calc_ensemble_percentile(pred_og,analysis,percentiles)
		real_p = np.percentile(np.ma.array(real, mask=mask), percentiles)
	
	ax.scatter(real_p,dsrnngan_p,color='#729ea1')
	ax.scatter(real_p,pred2_p,color='Green')
	ax.scatter(real_p,predog_p,color='Pink')

	x_y = np.arange(0,1500,1)
	ax.plot(x_y,x_y,'--',color='black')
	ax.fill_between(real_p, dsrnngan_min, dsrnngan_max,color='#729ea1',alpha=0.2)
	ax.fill_between(real_p, pred2_min, pred2_max,color='Green',alpha=0.2)
	ax.fill_between(real_p, predog_min, predog_max,color='Pink',alpha=0.2)
	ax.set_xlabel('Observed rain (mm / h)',fontsize=18)
	ax.set_ylabel('Predicted rain (mm / h)',fontsize=18)
	ax.set_title(analysis + ' rainfall',fontsize=18,fontweight='bold')
	ax.tick_params(axis='both', which='major', labelsize=14)
	if analysis == 'Mean | extremes':
		ax.set_ylim([0,20])
		ax.set_xlim([0,20])
		ax.set_yticks([0,5,10,15,20])
	elif analysis == 'Peak | extremes':
		ax.set_ylim([0,750])
		ax.set_xlim([0,750])
		ax.set_xticks([0,100,200,300,400,500,600,700])
	elif analysis == 'Core | extremes':
		ax.set_ylim([0,300])
		ax.set_xlim([0,300])
	elif analysis == 'Mean':
		ax.set_ylim([0,20])
		ax.set_xlim([0,20])
	elif analysis == 'Peak':
		ax.set_ylim([0,400])
		ax.set_xlim([0,400])
	elif analysis == 'Core':
		ax.set_ylim([0,150])
		ax.set_xlim([0,150])
		ax.set_xticks([0,20,40,60,80,100,120,140])

	print(analysis)
	return ax