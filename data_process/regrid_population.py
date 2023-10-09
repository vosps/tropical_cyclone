
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import colors
from netCDF4 import Dataset
# import pandas as pd
# import properscoring as ps
# import cartopy.feature as cfeature
# import cartopy.crs as ccrs
# import warnings
import xarray as xr
# from matplotlib import cm
# from utils.evaluation import find_landfalling_tcs,tc_region,create_xarray,get_storm_coords
# from utils.metrics import calculate_crps
# from global_land_mask import globe
# from scipy.interpolate import griddata
# from utils.data import load_tc_data
# from utils.plot import make_cmap
# from utils.metrics import calculate_fid
import xesmf as xe
# import glob


population_file = '/user/home/al18709/work/population/gpw_v4_population_count_rev11_2pt5_min.nc'
pop = xr.load_dataset(population_file)
pop_lats = pop.latitude
pop_lons = pop.longitude
data = pop['Population Count, v4.11 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes'].values[2]

fp = '/bp1/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2000342.00.nc'
d = Dataset(fp, 'r')
rain_lats = d.variables['lat'][:] #lat
rain_lons = d.variables['lon'][:] #lon

grid_in = {"lon": rain_lons, "lat": rain_lats}
grid_out = {"lon": rain_lons, "lat": rain_lats}
print('getting grid ready...')
regridder = xe.Regridder(grid_in, grid_out, "bilinear")
print('regridding')
pop_regrid = regridder(data)
print(pop_regrid.shape)
print(np.nansum(pop_regrid))
print(np.nansum(data))
np.save(pop_regrid,'population_count_regrid_2010_10km.npy')


# nlats,nlons = data.shape
# print(pop_lats.shape)
# print(data.shape)
# grid_X, grid_Y = np.meshgrid(rain_lons,rain_lats)
# pop_regrid = np.zeros(grid_X.shape)
# print(pop_regrid.shape)

# for i in range(nlats):
# 	for j in range(nlons):
# 		# print(i,j)
# 		pop_lat = pop_lats[i]
# 		pop_lon = pop_lons[j]
# 		# print(pop_lat)
# 		# print(pop_lon)
# 		population = data[i,j]
# 		closest_lon = (np.abs(rain_lons-pop_lon)).argmin()
# 		# print(closest_lon)
# 		closest_lat = (np.abs(rain_lats-pop_lat)).argmin()
# 		# print(closest_lat)
# 		pop_regrid[closest_lat,closest_lon] = pop_regrid[closest_lat,closest_lon] + population

# # new_population = xr.dataset({
# # 	'population' : pop_regrid,
# # 	'lats' : rain_lats,
# # 	'lons' : rain_lons
# # })

# new_population = xr.Dataset(
# 					data_vars=dict(
# 							population=(['y','x'], pop_regrid)
# 							),                        
# 					coords=dict(
# 							lat=("y", rain_lats),
# 							lon=("x", rain_lons)
# 					)
# 			)

# new_population.to_netcdf('/user/home/al18709/work/population/gpw_regrid.nc',format='NETCDF4')
