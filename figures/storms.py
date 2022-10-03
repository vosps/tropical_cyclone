import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pandas as pd
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import warnings
import xarray as xr
from utils.evaluation import create_xarray,get_storm_coords,calculate_crps,tc_region
from utils.data import load_tc_data
from multiprocessing import Pool
from utils.plot import make_cmap
import xesmf as xe
from global_land_mask import globe
import pandas as pd
warnings.filterwarnings("ignore")

sns.set_style("white")
sns.set_palette(sns.color_palette("Paired"))
sns.set_palette(sns.color_palette("Set2"))

precip_cmap,precip_norm = make_cmap(high_vals=True)

def accumulated_rain(storm,meta,real,pred_gan):
	# grab mswep coordinate variables
	fp = '/bp1store/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2000342.00.nc'
	d = Dataset(fp, 'r')
	lat = d.variables['lat'][:] #lat
	lon = d.variables['lon'][:] #lon
	print('lat shape: ',lat.shape)
	print('lon shape: ',lon.shape)
	# calculate lats and lons for storm
	lats,lons = tc_region(meta,storm,lat,lon)
	grid_x, grid_y = np.meshgrid(lons,lats)

	a = np.zeros((grid_y.shape))
	print('a shape',a.shape)
	accumulated_ds = create_xarray(lats,lons,a)
	accumulated_ds_pred = create_xarray(lats,lons,a)
	x,y = grid_y.shape
	accumulated_pred_ensemble = np.zeros((x,y,20))
	for j in range(20):
		for i in storm:
			storm_lats,storm_lons = get_storm_coords(lat,lon,meta,i)
			# define new xarray
			ds = create_xarray(storm_lats,storm_lons,real[i])
			ds_pred = create_xarray(storm_lats,storm_lons,pred_gan_ensemble[:,:,:,j][i])
			# regrid so grids match
			regridder = xe.Regridder(ds, accumulated_ds, "bilinear")
			if j==0:
				ds_out = regridder(ds)
				accumulated_ds = accumulated_ds + ds_out
			ds_pred_out = regridder(ds_pred)	
			# add up rainfall
			accumulated_ds_pred = accumulated_ds_pred + ds_pred_out
			accumulated_pred_ensemble[:,:,j] = accumulated_ds_pred.precipitation.values
		accumulated_ds_pred = create_xarray(lats,lons,a)

		accumulated_ds_pred_ensemble = create_xarray(lats,lons,accumulated_pred_ensemble,ensemble=range(20))

	return accumulated_ds,accumulated_ds_pred_ensemble

# load data
real,inputs,pred_cnn,pred_vaegan,pred_gan,pred_vaegan_ensemble,pred_gan_ensemble,meta = load_tc_data(set='test',results='test')
# real,inputs_x,pred_cnn_x,pred_vaegan_x,pred_gan_x,pred_vaegan_ensemble_x,pred_gan_ensemble,meta = load_tc_data(set='extreme_test',results='test')
tcs = pd.read_csv('/user/work/al18709/ibtracks/tc_files.csv')

# Amphan 2020136N10088 NI 2020138N10086
storm = '2020138N10086' #amphan
# storm = '2019063S18038' #idai
# storm = '1992044S09181' #daman
# storm = '1997279N12263' #pauline
# storm = '2013306N07162' #haiyan
# storm = '2011233N15301' #irene
meta = pd.read_csv('/user/work/al18709/tc_data_mswep_extend_flipped/meta_%s.csv' % storm)
real = np.load('/user/home/al18709/work/gan_predictions_20/storm_real-opt_%s.npy' % storm)[:,:,:,0]
pred_gan_ensemble = np.load('/user/home/al18709/work/gan_predictions_20/storm_pred-opt_%s.npy' % storm)
print(pred_gan_ensemble[:,:,0])
print(pred_gan_ensemble[:,:,1])
sid_2020138N10086 = np.arange(len(real[:,0,0]))
accumulated_ds,accumulated_ds_pred = accumulated_rain(sid_2020138N10086,meta,real,pred_gan_ensemble)


# plot stuff?
levels = [0,10,20,30,40,50,60,70,80,90,100]
centre_lats=None
centre_lons = None
intensity = None
tc_data = pd.read_csv('/user/work/al18709/ibtracks/tc_files.csv')


# land mask
grid_x, grid_y = np.meshgrid(accumulated_ds['lon'].values, accumulated_ds['lat'].values)
land = globe.is_land(grid_y,grid_x)
pred_land = accumulated_ds_pred['precipitation'].values.copy()
real_land = accumulated_ds['precipitation'].values.copy()
pred_land[~land] = 0
real_land[~land] = 0
# plot_accumulated(pred_land[:,:,1],accumulated_ds_pred['lat'].values,accumulated_ds_pred['lon'].values,None,vmin=0,vmax=100,levels = levels,plot='save',centre_lats=centre_lats,centre_lons=centre_lons,intensity=intensity)
print(pred_land.shape)

plt.imshow(pred_land[:,:,1],cmap=precip_cmap)
plt.colorbar()
ax = plt.gca()
ax.invert_yaxis()
plt.savefig('storm_pred_%s.png' % storm)
plt.clf()
plt.imshow(real_land,cmap=precip_cmap)
plt.colorbar()
ax = plt.gca()
ax.invert_yaxis()
plt.savefig('storm_real_%s.png' % storm)
crps = calculate_crps(accumulated_ds.precipitation.values,accumulated_ds_pred.precipitation.values)
crps_land = calculate_crps(real_land,pred_land)
mean_crps = crps.mean()
mean_crps_land = crps_land.mean()


total_basin_rain_real = np.sum(real_land)
print(pred_land.shape)
print(np.mean(pred_land,axis=-1).shape)
total_basin_rain_pred = np.sum(np.mean(pred_land,axis=-1))


normalised_crps_land = mean_crps_land/total_basin_rain_pred
normalised_crps = mean_crps/(np.sum(np.mean(accumulated_ds_pred['precipitation'].values,axis=-1)))

print('crps',mean_crps)
print('land crps',mean_crps_land)
print('total rain pred',total_basin_rain_pred)
print('total rain real',total_basin_rain_real)
print('normalised crps (land and ocean)',normalised_crps)
print('normalised crps (land)',normalised_crps_land)


# Amphan
# crps 3.103285542109189
# land crps 0.8388424807448492
# total rain pred 204658.53051294372
# total rain real 206849.3125
# normalised crps (land and ocean) 2.1656519433893422e-06
# normalised crps (land) 4.098741834227117e-06

# Idai
# crps 3.7767576829732863
# land crps 1.7106260556467234
# total rain pred 339491.80989887577
# total rain real 342965.375
# normalised crps (land and ocean) 3.724961060092826e-06
# normalised crps (land) 5.0387844589130635e-06

# Daman
# crps 2.1946556356686657
# land crps 0.10234705483038087
# total rain pred 25071.697050549847
# total rain real 31829.25
# normalised crps (land and ocean) 1.2107281198981835e-06
# normalised crps (land) 4.082174996930904e-06

# Pauline
# crps 6.025775858235912
# land crps 2.6465250484772422
# total rain pred 231167.10641321534
# total rain real 231140.0625
# normalised crps (land and ocean) 5.078087071627683e-06
# normalised crps (land) 1.144853647017812e-05

# Haiyan
# crps 1.3495214638553337
# land crps 0.34891323838914434
# total rain pred 378025.1632763001
# total rain real 392969.4375
# normalised crps (land and ocean) 5.739120775919245e-07
# normalised crps (land) 9.229894522500933e-07

# Irene
# crps 2.2025059908233575
# land crps 0.9040976852941741
# total rain pred 625263.7890814164
# total rain real 633363.875
# normalised crps (land and ocean) 9.898455286625653e-07
# normalised crps (land) 1.4459460168361842e-06