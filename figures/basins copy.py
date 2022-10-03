import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pandas as pd
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import warnings
import xarray as xr
from utils.evaluation import create_xarray,get_storm_coords,plot_accumulated,find_basin_coords,find_basin_tcs,calculate_crps
from utils.data import load_tc_data
from multiprocessing import Pool
import xesmf as xe
from global_land_mask import globe
import pandas as pd
warnings.filterwarnings("ignore")

sns.set_style("white")
sns.set_palette(sns.color_palette("Paired"))
sns.set_palette(sns.color_palette("Set2"))

# load data
real,inputs,pred_cnn,pred_vaegan,pred_gan,pred_vaegan_ensemble,pred_gan_ensemble,meta = load_tc_data(set='test',results='test')
# real,inputs_x,pred_cnn_x,pred_vaegan_x,pred_gan_x,pred_vaegan_ensemble_x,pred_gan_ensemble,meta = load_tc_data(set='extreme_test',results='test')
tcs = pd.read_csv('/user/work/al18709/ibtracks/tc_files.csv')

# define basin
basin = 'NA'
lats,lons = find_basin_coords(basin)
basin_sids = find_basin_tcs(meta,basin)


# assign sid variable to list of sid indices correspoinnding to storm timesteps
for sid in basin_sids:
	indices = meta.sid[meta.sid == sid].index.tolist()
	exec('sid_%s = indices' % sid)

all_sids = list(dict.fromkeys(meta['sid']))

def computation(sid):
	# initialise accumulated xarray
	lats,lons = find_basin_coords(basin)
	grid_x, grid_y = np.meshgrid(lons, lats)
	a = np.zeros((grid_y.shape))
	accumulated_ds = create_xarray(lats,lons,a)
	accumulated_ds_pred = create_xarray(lats,lons,a)
	x,y = grid_y.shape
	b = np.zeros((x,y,20))
	accumulated_pred_ensemble = np.zeros((x,y,20))

	# grab mswep coordinate variables
	fp = '/bp1store/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2000342.00.nc'
	d = Dataset(fp, 'r')
	lat = d.variables['lat'][:] #lat
	lon = d.variables['lon'][:] #lon
	sid = str(sid[0])

	for i in globals()['sid_%s' % sid]:
		# print('i is',i)	
		storm_lats,storm_lons = get_storm_coords(lat,lon,meta,i)

		ds = create_xarray(storm_lats,storm_lons,real[i])
		# regrid so grids match
		regridder = xe.Regridder(ds, accumulated_ds, "bilinear")
		ds_out = regridder(ds)
		accumulated_ds = accumulated_ds + ds_out

		for j in range(20):
			ds_pred = create_xarray(storm_lats,storm_lons,pred_gan_ensemble[:,:,:,j][i])	
			ds_pred_out = regridder(ds_pred)	
			# add up rainfall
			accumulated_ds_pred = accumulated_ds_pred + ds_pred_out
			accumulated_pred_ensemble[:,:,j] = accumulated_ds_pred.precipitation.values

		accumulated_ds_pred_ensemble = create_xarray(lats,lons,accumulated_pred_ensemble,ensemble=range(20))
		return accumulated_ds,accumulated_ds_pred_ensemble

def process(df):
	# print('doing process...')
	res = df.apply(computation,axis=1)
	return res

if __name__ == '__main__':
	df = pd.DataFrame(basin_sids)
	# df = df[0]
	df_split = np.array_split(df, len(basin_sids))
	p = Pool(processes=len(basin_sids))
	print('pooling processes...')

	pool_results = p.map(process, df_split)
	print('results pooled')
	# result = sum(p)
	p.close()

	p.join()

print(pool_results)

grid_x, grid_y = np.meshgrid(lons,lats)
a = np.zeros((grid_y.shape))
accumulated_ds = create_xarray(lats,lons,a)
accumulated_ds_pred = create_xarray(lats,lons,a)

# print('pool results:')
# print(pool_results)

for i in range(len(pool_results)):
	if pool_results[i][i] == None:
		continue
	accumulated_ds = accumulated_ds + pool_results[i][i][0]
	accumulated_ds_pred = accumulated_ds_pred + pool_results[i][i][1]


# plot stuff?
levels = [0,10,20,30,40,50,60,70,80,90,100]
centre_lats=None
centre_lons = None
intensity = None
tc_data = pd.read_csv('/user/work/al18709/ibtracks/tc_files.csv')
for storm in basin_sids:
	storm_data = tc_data[tc_data['sid']==storm]
	storm_lats = storm_data['lat']
	storm_lons = storm_data['lon']
	print(storm_data.columns)
	print(storm_data['lat'])
	print(storm_data['hour'])

	exit()

# land mask
grid_x, grid_y = np.meshgrid(accumulated_ds['lon'].values, accumulated_ds['lat'].values)
land = globe.is_land(grid_y,grid_x)
# plot_accumulated(land,accumulated_ds_pred['lat'].values,accumulated_ds_pred['lon'].values,vmin=0,vmax=100,levels = levels,plot='save',centre_lats=centre_lats,centre_lons=centre_lons,intensity=intensity)
pred_land = accumulated_ds_pred['precipitation'].values
pred_land[~land] = 0
# print(accumulated_ds_pred.precipitation.values.shape)
# print(pred_land)
plot_accumulated(pred_land[:,:,1],accumulated_ds_pred['lat'].values,accumulated_ds_pred['lon'].values,basin_sids,vmin=0,vmax=100,levels = levels,plot='save',centre_lats=centre_lats,centre_lons=centre_lons,intensity=intensity)

crps = calculate_crps(accumulated_ds.precipitation.values,accumulated_ds_pred.precipitation.values)
mean_crps = crps.mean()
total_basin_rain = np.sum(np.mean(pred_land,axis=-1))
normalised_crps = mean_crps/total_basin_rain
print(crps.mean())
print(total_basin_rain)
print(normalised_crps)
print(len(basin_sids))

# NA - test = 5.094561309165959,  580350.6327211779,8.778419496638296e-06 
# nstorms = 53
# extreme = 1.3553822243221434, 177730.33752527097,7.626060036764548e-06

# NWP - test = 14.259372481267006,590279.1223909395,2.415699952847575e-05
# nstorms = 110
# extreme = 11.515747802961915,646993.4043546547,1.7798864293598677e-05

# NIO - test = 0.6985875320278012,41625.02366503916,1.6782874110761037e-05
# extreme = 0.3799838991402773,54126.58479913847,7.020282187586413e-06



# SPO - test = 


# A - test = 8.578049557717016,708657.9787873459,1.2104639776152321e-05
# extreme = 1.5856964579715616,187389.18838472883,8.462048806764494e-06




