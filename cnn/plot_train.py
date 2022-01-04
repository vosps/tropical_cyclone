"""
Plot the outputs from the train.py script to see if output is sensible

"""
# import modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
sns.set_style("white")

def plot_tc(data,lat_2d,lon_2d,ax):

	cmap ='seismic_r'
	c = ax.pcolor(y_lon2d,y_lat2d,data,vmin=-30,vmax=30,cmap = cmap, transform=ccrs.PlateCarree())
	ax.outline_patch.set_linewidth(0.3)

	return ax

# load in data
X_filepath = 'data/X.npy'
ypred_filepath = 'data/y_pred.npy'
ytrue_filepath = 'data/y_true.npy'
X = np.load(X_filepath)
y_pred = np.load(ypred_filepath)
y_true = np.load(ytrue_filepath)[:,0,:,:]

print(X.shape)
print(y_true.shape)

# begin plotting
fig, axs  = plt.subplots(3, 6,subplot_kw={'projection': ccrs.PlateCarree()})

y_lat2d, y_lon2d = np.linspace(-50, 50, 64),np.linspace(0, 100, 64)
lat_2d, lon_2d = np.meshgrid(y_lat2d,y_lon2d)

# axs[0,0] = plot_tc(X[12005],lat_2d,lon_2d,axs[0,0])
# axs[0,1] = plot_tc(X[12006],lat_2d,lon_2d,axs[0,1])
# axs[0,2] = plot_tc(X[12007],lat_2d,lon_2d,axs[0,2])
# axs[0,3] = plot_tc(X[12008],lat_2d,lon_2d,axs[0,3])
# axs[0,4] = plot_tc(X[12035],lat_2d,lon_2d,axs[0,4])
# axs[0,5] = plot_tc(X[12042],lat_2d,lon_2d,axs[0,5])

axs[0,0] = plot_tc(X[5],lat_2d,lon_2d,axs[0,0])
axs[0,1] = plot_tc(X[6],lat_2d,lon_2d,axs[0,1])
axs[0,2] = plot_tc(X[7],lat_2d,lon_2d,axs[0,2])
axs[0,3] = plot_tc(X[8],lat_2d,lon_2d,axs[0,3])
axs[0,4] = plot_tc(X[35],lat_2d,lon_2d,axs[0,4])
axs[0,5] = plot_tc(X[42],lat_2d,lon_2d,axs[0,5])

# y_lat2d, y_lon2d = np.linspace(-50, 50, 256),np.linspace(0, 100, 256)
y_lat2d, y_lon2d = np.linspace(-50, 50, 128),np.linspace(0, 100, 128)
lat_2d, lon_2d = np.meshgrid(y_lat2d,y_lon2d)

axs[2,0] = plot_tc(y_pred[5],lat_2d,lon_2d,axs[2,0])
axs[2,1] = plot_tc(y_pred[6],lat_2d,lon_2d,axs[2,1])
axs[2,2] = plot_tc(y_pred[7],lat_2d,lon_2d,axs[2,2])
axs[2,3] = plot_tc(y_pred[8],lat_2d,lon_2d,axs[2,3])
axs[2,4] = plot_tc(y_pred[35],lat_2d,lon_2d,axs[2,4])
axs[2,5] = plot_tc(y_pred[42],lat_2d,lon_2d,axs[2,5])

axs[1,0] = plot_tc(y_true[5],lat_2d,lon_2d,axs[1,0])
axs[1,1] = plot_tc(y_true[6],lat_2d,lon_2d,axs[1,1])
axs[1,2] = plot_tc(y_true[7],lat_2d,lon_2d,axs[1,2])
axs[1,3] = plot_tc(y_true[8],lat_2d,lon_2d,axs[1,3])
axs[1,4] = plot_tc(y_true[35],lat_2d,lon_2d,axs[1,4])
axs[1,5] = plot_tc(y_true[42],lat_2d,lon_2d,axs[1,5])



plt.tight_layout()
fig.tight_layout()
# fig.subplots_adjust(hspace=-0.7)
fig.subplots_adjust(hspace=-0.575)

# subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.savefig('predictions_new_new.png',dpi=600,bbox_inches='tight')
plt.clf()










