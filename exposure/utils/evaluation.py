from operator import truediv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from global_land_mask import globe
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
import xesmf as xe

sns.set_style("white")

def plot_predictions(real,pred,inputs,plot='save',mode='validation'):
        # real[real<=0.1] = np.nan
        # pred[pred<=0.1] = np.nan
        # inputs = regrid(inputs[99])
        inputs[inputs<=0.1] = np.nan
        n = 4
        m = 3
        if plot == 'save':
                fig, axes = plt.subplots(n, m, figsize=(5*m, 5*n), sharey=True)
        else:
                print('show')
                fig, axes = plt.subplots(n, m, figsize=(2*m, 2*n), sharey=True)
        if mode == 'extreme_valid':
                range_ = (-5, 30)
        else:
                range_ = (-5, 20)
        if mode == 'gcm':
                range_ = (-5,30)
        if mode == 'cmip':
                range_ = (-5,60)

        storms = [102,260,450,799]
        storms = [1200,260,1799,20]
        storms = [32,70,20,60]
        if mode == 'gcm':
                storms = [0,4,2,3,4]
        axes[0,0].set_title('Real',size=24)
        axes[0,1].set_title('Predicted',size=24)
        axes[0,2].set_title('Input',size=24)
        for i in range(n):
                j = 0
                storm = storms[i]
                print(real[storm].max())
                print(pred[storm].max())
                print(np.nanmax(inputs[storm]))
                axes[i,j].imshow(real[storm], interpolation='nearest', norm=colors.Normalize(*range_), extent=None,cmap='Blues')
                axes[i,j+1].imshow(pred[storm], interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
                axes[i,j+2].imshow(regrid(inputs[storm]), interpolation='nearest',norm=colors.Normalize(*range_), extent=None,cmap='Blues')
                axes[i,j].set(xticklabels=[])
                axes[i,j].set(yticklabels=[])
                axes[i,j+1].set(xticklabels=[])
                axes[i,j+1].set(yticklabels=[])
                axes[i,j+2].set(xticklabels=[])
                axes[i,j+2].set(yticklabels=[])

        if plot == 'save':
                plt.savefig('figs/pred_images_%s.png' % mode,bbox_inches='tight')
                plt.clf()
        else:
                plt.show()

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

def plot_anomaly(inputs,cmap,plot='save',vmin=-1,vmax=1,levels = False,mode='validation'):  
        fig, ax = plt.subplots(figsize=(5, 5))  
        ax.set_title('Anomaly')
        im = ax.imshow(inputs, interpolation='nearest', vmin=vmin,vmax=vmax,extent=None,cmap=cmap)
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        if levels:
                plt.colorbar(im, cax=cax,levels = levels) # Similar to fig.colorbar(im, cax = cax)
        else:
                plt.colorbar(im, cax=cax)
        # plt.colorbar(im)

        if plot == 'save':
                plt.savefig('figs/input_images_%s.png' % mode,bbox_inches='tight')
                plt.clf()
        else:
                plt.show()

def plot_histogram(real,pred,binwidth,alpha,type='Mean'):
        """
        This function plots a histogram of the set in question
        """
        # ax = sns.histplot(data=penguins, x="flipper_length_mm", hue="species", element="step")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(ax=ax,data=real, stat="density", fill=True,color='#b5a1e2',element='step',alpha=alpha)
        sns.histplot(ax=ax,data=pred, stat="density", fill=True,color='#dc98a8',element='step',alpha=alpha)
        ax.set_xlabel('Mean or Peak rainfall (mm/h)',size=18)
        ax.set_xlabel('%s Rainfall (mm/h)' % type,size=18)
        ax.set_ylabel('Density',size=18)
        plt.legend(labels=['real','pred'],fontsize=24)
        plt.show()
        # plt.savefig('figs/histogram_accumulated_%s.png' % mode)

def calc_peak(array):
        nstorms,_,_ = array.shape
        peaks = np.zeros((nstorms))
        for i in range(nstorms):
                peaks[i] = np.nanmax(array[i])
        return peaks

def calc_mean(array):
        nstorms,_,_ = array.shape
        means = np.zeros((nstorms))
        for i in range(nstorms):
                means[i] = np.nanmean(array[i])
        return means

def segment_image(im):
        im[im <= 1] = 0
        im = np.where((im > 1) & (im <= 5),1,im)
        im = np.where((im > 5) & (im <= 20),2,im)
        im = np.where((im > 20) & (im <= 50),2,im)
        im = np.where((im > 50),3,im)

        return im

def segment_diff(real_im,pred_im,rain):

        low_to_no_pred = np.where(real_im == rain, pred_im, np.nan)
        correct = (low_to_no_pred == rain).sum()
        no = (low_to_no_pred == 0).sum()
        light = (low_to_no_pred == 1).sum()
        medium = (low_to_no_pred == 2).sum()
        heavy = (low_to_no_pred == 3).sum()

        return correct,no,light,medium,heavy
        
        # seg = np.where(im<1.,np.where(),0)

def plot_accumulated(data,lats,lons,vmin=0,vmax=200,plot='show',cmap='Blues',title='Accumulated Rainfall',levels=[0,50,100,150,200,250,300]):
        """
        Plots the accumulated rainfall of a tropical cyclone while it's at tropical cyclone strength
        """
        lat2d,lon2d = np.meshgrid(lats,lons)
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        c = ax.contourf(lon2d,lat2d,data,vmin=vmin,vmax=vmax,levels=levels,cmap = cmap, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND) # TODO: fix this as it doesn't work
        ax.add_feature(cfeature.COASTLINE)
        ax.outline_patch.set_linewidth(0.3)
        # cbar = plt.colorbar(c)
        # cbar = plt.colorbar(c, shrink=0.54)
        cbar = plt.colorbar(c, shrink=0.68)
        # cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=6,width=0.5)
        # plt.title(title)

        if plot=='show':
                plt.show()
        else:
                plt.savefig('accumulated_rainfall.png')

def find_landfalling_tcs(meta,land=True):
        """
        Grabs all tcs that ever make landfall at tc strength

                inputs : meta csv
        """
        nstorms,_ = meta.shape
        landfall_sids = []
        for i in range(nstorms):
                centre_lat = meta['centre_lat'][i]
                centre_lon = meta['centre_lon'][i]
                if centre_lon > 180:
                        centre_lon = centre_lon - 180
                if land == True:
                        landfall = globe.is_land(centre_lat, centre_lon)
                else:
                        landfall = True
                if landfall:
                        sid = meta['sid'][i]
                        landfall_sids.append(sid)

        # find indices of all landfalling snapshots
        landfall_sids = list(dict.fromkeys(landfall_sids))
        return landfall_sids

def is_near_land(centre_lat, centre_lon):
        lats = [centre_lat + i for i in range(-3,3)]
        lons = [centre_lon + i for i in range(-3,3)]
        for lat in lats:
                for lon in lons:
                        if globe.is_land(lat, lon):
                                return True
        return False



def tc_region(meta,sid_i,lat,lon,era5=False):
        """
        find region which contains all points/ranfinall data of tc track
                inputs
                        meta : csv with metadata
                        sid_i : list of sid indices
                        lat : mswep grid
                        lon : mswep grid
        """
        lat_lower_bounds = []
        lat_upper_bounds = []
        lon_lower_bounds = []
        lon_upper_bounds = []
                
        for i in sid_i:
                lat_lower_bounds.append((np.abs(lat-meta['centre_lat'][i]+5.)).argmin())
                lat_upper_bounds.append((np.abs(lat-meta['centre_lat'][i]-5.)).argmin())
                lon_lower_bounds.append((np.abs(lon-meta['centre_lon'][i]+5.)).argmin())
                lon_upper_bounds.append((np.abs(lon-meta['centre_lon'][i]-5.)).argmin())
        
        lat_lower_bound = min(lat_lower_bounds)
        lat_upper_bound = max(lat_upper_bounds)
        lon_lower_bound = min(lon_lower_bounds)
        lon_upper_bound = max(lon_upper_bounds)

        if era5 == True:
                # lat_lower_bound = lat_lower_bound + 1
                lat_upper_bound = lat_upper_bound - 1

        print('lat lower: ',lat_lower_bound)
        print('lat upper: ',lat_upper_bound)
        print('lon lower: ',lon_lower_bound)
        print('lon upper: ',lon_upper_bound)

        lats = lat[lat_lower_bound:lat_upper_bound]
        # lats = np.flip(lats)
        lons = lon[lon_lower_bound:lon_upper_bound]
        print('lats: ',lats.shape)
        print('lons: ',lons.shape)

        return lats,lons

def create_xarray(lats,lons,data,ensemble=None):
        if ensemble==None:
                accumulated_ds = xr.Dataset(
                        data_vars=dict(
                                precipitation=(["y", "x"], data)),  
                        coords=dict(
                                lon=("x", lons),
                                lat=("y", lats),
                        ))
        else:
                
                accumulated_ds = xr.Dataset(
                        data_vars=dict(
                                precipitation=(["y","x","ens"], data)),                        
                        coords=dict(
                                lon=("x", lons),
                                lat=("y", lats),
                                member=("ens", ensemble)
                        )
                )

        return accumulated_ds

def get_storm_coords(lat,lon,meta,i,era5=False):
        """
        returns lat and longitude of rainfall from one storm
        """

        lat_lower_bound = (np.abs(lat-meta['centre_lat'][i]+5.)).argmin()
        lat_upper_bound = (np.abs(lat-meta['centre_lat'][i]-5.)).argmin()
        lon_lower_bound = (np.abs(lon-meta['centre_lon'][i]+5.)).argmin()
        lon_upper_bound = (np.abs(lon-meta['centre_lon'][i]-5.)).argmin()
        # storm_lats = lat[lat_lower_bound:lat_upper_bound]
        # # storm_lats = np.flip(storm_lats)
        # storm_lons = lon[lon_lower_bound:lon_upper_bound]
        if era5 ==True:
                print(lat_upper_bound)
                lat_upper_bound = lat_upper_bound +1
                print(lat_upper_bound)

        if meta['centre_lon'][i] > 175: 
                diff = lon_upper_bound - lon_lower_bound
                second_upper_bound = 100 - diff
                storm_lats = lat[lat_lower_bound:lat_upper_bound]
                lon1 = lon[lon_lower_bound:lon_upper_bound]
                lon2 = lon[0:second_upper_bound]
                storm_lons = np.concatenate((lon1,lon2))
        elif meta['centre_lon'][i] < -175:
                diff = lon_upper_bound - lon_lower_bound
                second_upper_bound = 100 - diff
                storm_lats = lat[lat_lower_bound:lat_upper_bound]
                lon1 = lon[-second_upper_bound:-1]
                lon2 = lon[lon_lower_bound:lon_upper_bound]
                storm_lons = np.concatenate((lon1,lon2))
        else:
                storm_lats = lat[lat_lower_bound:lat_upper_bound]
                storm_lons = lon[lon_lower_bound:lon_upper_bound]


        return storm_lats,storm_lons

def plot_accumulated(data,lats,lons,basin_sids,vmin=0,vmax=200,plot='show',cmap='Blues',title='Accumulated Rainfall',levels=[0,50,100,150,200,250,300],centre_lats=None,centre_lons=None,intensity=None):
        """
        Plots the accumulated rainfall of a tropical cyclone while it's at tropical cyclone strength
        """
        data = np.where(data<1,np.nan,data)
        lon2d,lat2d = np.meshgrid(lons,lats)
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        # ax.set_extent([-100,-30,5,45], crs=ccrs.PlateCarree())
        c = ax.contourf(lon2d,lat2d,data,vmin=vmin,vmax=vmax,levels=levels,cmap = cmap, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE,linewidth=0.5)
        if centre_lats is not None:
                for i in range(len(centre_lats)):
                        if intensity[i] == 0.0:
                                colour = '#ffb600'
                        elif intensity[i] == 1.0:
                                colour =  '#ff9e00'
                        elif intensity[i] == 2.0:
                                colour = '#ff7900'
                        elif intensity[i] == 3.0:       
                                colour = '#ff6000'
                        elif intensity[i] == 4.0:
                                colour = '#ff4000' 
                        elif intensity[i]==5.0:
                                colour = '#ff2000' 
                        ax.plot(centre_lons[i:i+2],centre_lats[i:i+2],color=colour)

        ax.outline_patch.set_linewidth(0.5)
        cbar = plt.colorbar(c, shrink=0.68)
        cbar.ax.tick_params(labelsize=6,width=0.5)

        tc_data = pd.read_csv('/user/work/al18709/ibtracks/tc_files.csv')

        for storm in basin_sids:
                storm_data = tc_data[tc_data['sid']==storm]
                storm_lats = storm_data['lat']
                storm_lons = storm_data['lon']
                plt.plot(storm_lons,storm_lats,linewidth=0.1,color='Black')
        # ax.set_xlim(-100,-30)
        # ax.set_ylim(5,45)
        ax.set_xlim(np.min(lons),np.max(lons))
        ax.set_ylim(np.min(lats),np.max(lats))
        plt.tight_layout()
        if plot=='show':
                plt.show()
        else:
                plt.savefig('basin_rainfall.png',bbox_inches='tight',dpi=300)

def find_basin_coords(basin):
	# grab mswep coordinate variables
	fp = '/bp1store/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2000342.00.nc'
	ds = xr.open_dataset(fp)
	if basin == 'NA':
		min_lon,min_lat,max_lon,max_lat = -100,5,-30,45
	elif basin == 'NIO':
		min_lon,min_lat,max_lon,max_lat = 40,5,110,30
	elif basin == 'NWP':
			min_lon,min_lat,max_lon,max_lat = 90,5,179,30
	elif basin == 'SPO':
			min_lon,min_lat,max_lon,max_lat = 130,-30,-110,-5
	elif basin == 'A':
			min_lon,min_lat,max_lon,max_lat = 70,-30,150,-5

	mask_lon = (ds.lon >= min_lon) & (ds.lon <= max_lon)
	mask_lat = (ds.lat >= min_lat) & (ds.lat <= max_lat)
	cropped_ds = ds.where(mask_lon & mask_lat, drop=True)

	lats = cropped_ds.lat.values
	lons = cropped_ds.lon.values
	return lats,lons

def find_basin_tcs(meta,basin):
		"""
		Grabs all tcs that ever make landfall at tc strength

				inputs : meta csv
		"""
		nstorms,_ = meta.shape
		basin_sids = []
		if basin == 'NA':
			min_lon,min_lat,max_lon,max_lat = -100,5,-30,45
		elif basin == 'NIO':
			min_lon,min_lat,max_lon,max_lat = 40,5,110,30
		elif basin == 'NWP':
			min_lon,min_lat,max_lon,max_lat = 90,5,179,30
		elif basin == 'SPO':
			min_lon,min_lat,max_lon,max_lat = 130,-30,-110,-5
		elif basin == 'A':
			min_lon,min_lat,max_lon,max_lat = 70,-30,150,-5
		
		for i in range(nstorms):
				centre_lat = meta['centre_lat'][i]
				centre_lon = meta['centre_lon'][i]
				if centre_lon > 180:
						centre_lon = centre_lon - 180
				in_basin = (centre_lat >= min_lat) & (centre_lat <= max_lat) & (centre_lon >= min_lon) & (centre_lon <= max_lon)
				if in_basin:
						sid = meta['sid'][i]
						basin_sids.append(sid)

		# find indices of all basining snapshots
		basin_sids = list(dict.fromkeys(basin_sids))
		return basin_sids

def calculate_crps(observation, forecasts):
	# forecasts = forecasts[...,None]
	fc = forecasts.copy()
	fc.sort(axis=-1)
	obs = observation
	fc_below = fc < obs[..., None]
	crps = np.zeros_like(obs)

	for i in range(fc.shape[-1]):
		below = fc_below[..., i]
		weight = ((i+1)**2 - i**2) / fc.shape[-1]**2
		crps[below] += weight * (obs[below]-fc[..., i][below])

	for i in range(fc.shape[-1] - 1, -1, -1):
		above = ~fc_below[..., i]
		k = fc.shape[-1] - 1 - i
		weight = ((k+1)**2 - k**2) / fc.shape[-1]**2
		crps[above] += weight * (fc[..., i][above] - obs[above])

	return crps

def accumulated_rain(storm,meta,real,pred_gan,inputs,flip=True,era5=False):
	# grab mswep coordinate variables
	fp = '/bp1store/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/2000342.00.nc'

	d = Dataset(fp, 'r')
	lat = d.variables['lat'][:] #lat
	lon = d.variables['lon'][:] #lon
	print('lat shape: ',lat.shape)
	print('lon shape: ',lon.shape)
	# calculate lats and lons for storm
	lats,lons = tc_region(meta,storm,lat,lon)
	# initialise accumulated xarray
	# grid_x, grid_y = np.meshgrid(lats, lons)
	grid_x, grid_y = np.meshgrid(lons,lats)
	# a = np.zeros((grid_x.shape))
	# print('grid_x shape: ',grid_x.shape)
	# print('grid_y.shape: ', grid_y.shape)
	# print('lons shape: ',lons.shape)
	# print('lats shape: ',lats.shape)
	a = np.zeros((grid_y.shape))
	# print('a shape',a.shape)
	accumulated_ds = create_xarray(lats,lons,a)
	accumulated_ds_pred = create_xarray(lats,lons,a)
	# accumulated_ds_input = create_xarray(lats,lons,a)
	# loop through storm time steps o generate accumulated rainfall
	for i in storm:
		storm_lats,storm_lons = get_storm_coords(lat,lon,meta,i)
		# print('storm_lonsshape1',storm_lons.shape)
		# print('storm_latshape1',storm_lats.shape)
		if storm_lats.shape != (100,):
			storm_lats,storm_lons = get_storm_coords(lat,lon,meta,i,era5=era5)
		# print('storm_lonsshape2',storm_lons.shape)
		# print('storm_latshape2',storm_lats.shape)
		ds = create_xarray(storm_lats,storm_lons,real[i])
		ds_pred = create_xarray(storm_lats,storm_lons,pred_gan[i])
		input_lats,input_lons = get_storm_coords(np.arange(-89.5,90,1),np.arange(-179.5,180),meta,i)
		# ds_input = create_xarray(input_lats,input_lons,inputs[i])

		# if flip==True:
		# 	ds.precipitation.values = np.flip(ds.precipitation.values,axis=0)
		# 	ds_pred.precipitation.values = np.flip(ds_pred.precipitation.values,axis=0)

		# regrid so grids match
		regridder = xe.Regridder(ds, accumulated_ds, "bilinear")
		ds_out = regridder(ds)
		ds_pred_out = regridder(ds_pred)

		# regird the inputs
		# regridder = xe.Regridder(ds_input, accumulated_ds, "bilinear")
		# ds_input_out = regridder(ds_input)

		# add up rainfall
		accumulated_ds = accumulated_ds + ds_out
		accumulated_ds_pred = accumulated_ds_pred + ds_pred_out
		# accumulated_ds_input = accumulated_ds_input + ds_input_out

	return accumulated_ds,accumulated_ds_pred


