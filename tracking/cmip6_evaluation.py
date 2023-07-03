print('asdf', flush=True)
print('importing modules')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
import cartopy.feature as cfeature
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from utils.data import load_tc_data
from utils.plot import change_units,calc_power,log_spectral_batch,plot_qq

# load datasets
print('loading datasets', flush=True)
real,inputs,pred,meta = load_tc_data(set='validation',results='ke_tracks')
miroc_fp = '/user/home/al18709/work/gan_predictions_20/miroc_pred-opt_no_rain_test_run_1.npy'
miroc_corrected_fp = '/user/home/al18709/work/gan_predictions_20/miroc_corrected_pred-opt_no_rain_test_run_1.npy'
miroc = np.load(miroc_fp)
miroc_corrected = np.load(miroc_corrected_fp)
print('datasets loaded!', flush=True)

# plot qq plot
sns.set_context("notebook")
fig, axes = plt.subplots(1,1, figsize=(10, 10))
alpha=0.7
sns.set_style("white")
print('plotting kde plots', flush=True)
bins = np.arange(0,150,1)
counts, bins = np.histogram(real[:,:,:,0].ravel(),bins=bins,density=True)
plt.stairs(counts, bins,color='#cbc3db',alpha=alpha,edgecolor='#6f3bdb')
# sns.kdeplot(real[:,:,:,0].ravel(),fill=True,color='#cbc3db',ax=axes,alpha=alpha,edgecolor='#6f3bdb')
print('1 down...', flush=True)
counts, bins = np.histogram(miroc[:,:,:,0].ravel(),bins=bins,density=True)
# sns.kdeplot(miroc[:,:,:,0].ravel(),fill=True,color='#bcd3e0',ax=axes,alpha=alpha,edgecolor='#1780c2')
plt.stairs(counts, bins,color='#bcd3e0',alpha=alpha,edgecolor='#1780c2')
print('2 down...', flush=True)
# sns.kdeplot(miroc_corrected[:,:,:,0].ravel(),fill=True,color='#b3c9b6',ax=axes,alpha=alpha,edgecolor='#86b58b')
counts, bins = np.histogram(miroc_corrected[:,:,:,0].ravel(),bins=bins,density=True)
plt.stairs(counts, bins,color='#b3c9b6',alpha=alpha,edgecolor='#86b58b')

plt.xlim([-1, 30])
axes.legend(['MSWEP Obs','MIROC6 hist','MIROC 6 hist corrected inputs'],fontsize=20)
axes.set_title('title',fontsize=48)
plt.savefig('hist_plot_cmip6_correction_comparison.png')
print('kde plot saved!', flush=True)

# plot power spectra plot
print('calculating power spectra')
lsb_real = log_spectral_batch(real[:,:,:,0])
lsb_gan = log_spectral_batch(pred[:,:,:,0])
lsb_miroc = log_spectral_batch(miroc[:,:,:,0])
lsb_miroc_qm = log_spectral_batch(miroc_corrected[:,:,:,0])

kvals_real,Abins_real = calc_power(lsb_real)
kvals_gan,Abins_gan = calc_power(lsb_gan)
kvals_miroc,Abins_miroc = calc_power(lsb_miroc)
kvals_miroc_qm,Abins_miroc_qm = calc_power(lsb_miroc_qm)

k_real,p_k_real = kvals_real[0],np.mean(Abins_real,axis=0)
k_gan,p_k_gan = kvals_gan[0],np.mean(Abins_gan,axis=0)
K_miroc,p_k_miroc = kvals_miroc[0],np.mean(Abins_miroc,axis=0)
K_miroc_qm,p_k_miroc_qm = kvals_miroc_qm[0],np.mean(Abins_miroc_qm,axis=0)

k_gan_ensemble = np.zeros((20,50))
p_k_gan_ensemble = np.zeros((20,50))
for i in range(20):
	lsb_gan_i = log_spectral_batch(pred[:,:,:,i])
	kvals_gan_i,Abins_gan_i = calc_power(lsb_gan_i)
	k_gan,p_k_gan = kvals_gan[0],np.mean(Abins_gan,axis=0)
	k_gan_i,p_k_gan_i = kvals_gan[0],np.mean(Abins_gan_i,axis=0)
	k_gan_ensemble[i,:] = k_gan_i
	p_k_gan_ensemble[i,:] = p_k_gan_i

k_gan_min,p_k_gan_min = k_gan_ensemble[0],np.min(p_k_gan_ensemble,axis=0)
k_gan_max,p_k_gan_max = k_gan_ensemble[0],np.max(p_k_gan_ensemble,axis=0)

k_gan_ensemble,p_k_gan_ensemble = k_gan_ensemble[0],np.mean(p_k_gan_ensemble,axis=0)

fig, axes = plt.subplots(1, 1, figsize=(10, 10), sharey=False)
plt.style.use('seaborn-ticks')
plt.rcParams["figure.figsize"] = (10,10)
sns.set_context("notebook")

plt.loglog(change_units(k_gan_ensemble),p_k_gan_ensemble,color='#219ebc')
plt.loglog(change_units(K_miroc),p_k_miroc,color='Green')
plt.loglog(change_units(K_miroc_qm),p_k_miroc_qm,color='Pink')

axes.text(-0.1, 1.05, 'a.', transform=axes.transAxes, size=28, weight='bold')
plt.loglog(change_units(k_real),p_k_real,color='black')
plt.xlabel("$k$ ($km^{-1}$)",fontsize=30)
plt.xlim([0,0.1])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.ylabel("$P(k)$",fontsize=30)
plt.legend(['WGAN_1D','MIROC','MIROC_QM','HR obs'],frameon=True,fontsize=18)
# plt.fill_between(k_vaegan_ensemble, p_k_vaegan_min,p_k_vaegan_max,color='#57cc99',alpha=0.2)
# plt.fill_between(k_gan_ensemble, p_k_gan_min,p_k_gan_max,color='#219ebc',alpha=0.2)
plt.savefig('figure5-power_spectra-ke_tracks_miroc.png',bbox_inches='tight',dpi=600)


# plot qq plots
sns.set_style("ticks")
sns.set_context("notebook")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
ax1 = plot_qq(axes[0],'Mean',pred,miroc,miroc_corrected,real)
ax2 = plot_qq(axes[1],'Peak',pred,miroc,miroc_corrected,real)
ax4 = plot_qq(axes[2],'Core',pred,miroc,miroc_corrected,real)


axes[0].text(-0.1, 1.05, 'a.', transform=axes[0].transAxes, size=24, weight='bold')
axes[1].text(-0.1, 1.05, 'b.', transform=axes[1].transAxes, size=24, weight='bold')
axes[2].text(-0.1, 1.05, 'c.', transform=axes[2].transAxes, size=24, weight='bold')

plt.legend(['WGAN_1D','MIROC6','MIROC6 QM','HR Obs'],frameon=False,fontsize=16)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.35)
plt.savefig('qq_plot_Total_test_new_miroc.png',bbox_inches='tight',dpi=600)
plt.show()