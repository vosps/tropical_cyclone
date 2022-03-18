import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from global_land_mask import globe
import xarray as xr

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

        storms = [102,260,450,799]
        storms = [1200,260,1799,20]
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
        fig, ax = plt.subplots(figsize=(15, 5))
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
        cbar = plt.colorbar(c)
        # cbar = plt.colorbar(c, shrink=0.54)
        # cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=6,width=0.5)
        plt.title(title)

        if plot=='show':
                plt.show()
        else:
                plt.savefig('figs/accumulated_rainfall')

def find_landfalling_tcs(meta):
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
                landfall = globe.is_land(centre_lat, centre_lon)
                if landfall:
                        sid = meta['sid'][i]
                        landfall_sids.append(sid)

        # find indices of all landfalling snapshots
        landfall_sids = list(dict.fromkeys(landfall_sids))
        return landfall_sids

def tc_region(meta,sid_i,lat,lon):
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

        lats = lat[lat_lower_bound:lat_upper_bound]
        # lats = np.flip(lats)
        lons = lon[lon_lower_bound:lon_upper_bound]

        return lats,lons

def create_xarray(lats,lons,data):
        accumulated_ds = xr.Dataset(
                data_vars=dict(
                        precipitation=(["x", "y"], data)),
                coords=dict(
                        lon=("x", lons),
                        lat=("y", lats),
                ))
        return accumulated_ds

def get_storm_coords(lat,lon,meta,i):
        """
        returns lat and longitude of rainfall from one storm
        """
        lat_lower_bound = (np.abs(lat-meta['centre_lat'][i]+5.)).argmin()
        lat_upper_bound = (np.abs(lat-meta['centre_lat'][i]-5.)).argmin()
        lon_lower_bound = (np.abs(lon-meta['centre_lon'][i]+5.)).argmin()
        lon_upper_bound = (np.abs(lon-meta['centre_lon'][i]-5.)).argmin()
        storm_lats = lat[lat_lower_bound:lat_upper_bound]
        # storm_lats = np.flip(storm_lats)
        storm_lons = lon[lon_lower_bound:lon_upper_bound]
        return storm_lats,storm_lons





