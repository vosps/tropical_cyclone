
""" File for handling data loading and saving. """


import os
import numpy as np
import xarray as xr
from glob import glob
from dask.array.chunk import coarsen

def get_dates(year):
    files = glob(f"/ppdata/NIMROD/{year}/*.nc")
    dates = []
    for f in files:
        dates.append(f[:-3].split('_')[-1])
    return dates

def list_dates(years):
    """inputs: list of years
      outputs: unpacked list of all dates for those years """
    dates = []
    for year in years:
        dates.append(get_dates(year))
        ## unpack list of lists into a list
        ## new list produced for each year and we want a single list
    dates_list = [val for sublist in dates for val in sublist]
    return dates_list

def load_nimrod(date, nimrod_path, hour, log_precip=False):
    year = date[:4]
    data = xr.open_dataset(f"{nimrod_path}{year}/metoffice-c-band-rain-radar_uk_{date}.nc")
    y = np.array(data['unknown'][hour,:,:])
    data.close()
    # The remapping of NIMROD left a few negative numbers
    # So remove those
    y[y<0]=0
    if log_precip:
        return np.log10(1+y)
    else:            
        return y

def load_norm(norm_year=2016):
    import pickle
    with open(f'/ppdata/constants/ERANorm{norm_year}.pkl', 'rb') as f:
        return pickle.load(f)

def load_era(date, era_path, field, hour, log_precip=False, era_norm=False, era_crop=2, norm_year=2016):
    year = date[:4]
    data = xr.open_dataarray(f"{era_path}{year}/{field}{date}.nc")
    norm_yr = load_norm(norm_year)
    if era_crop is None or era_crop == 0:
        y = np.array(data[hour,:,:])
    else:
        y = np.array(data[hour,era_crop:-era_crop,era_crop:-era_crop])        
    data.close()
    if log_precip and field in ['pr','prc','prl']:
        # ERA precip is measure in meters, so multiple up
        return np.log10(1+y*1000)
    elif era_norm:
        return (y-norm_yr[field][0])/norm_yr[field][1]
    else:
        return y

def load_erastack(date, era_path, fields, hour, log_precip=False, era_norm=False, era_crop=2, norm_year=2016):
    field_arrays = []
    for f in fields:
        field_arrays.append(load_era(date, era_path, f, hour, log_precip=log_precip, era_norm=era_norm, era_crop=era_crop, norm_year=norm_year))
    return np.stack(field_arrays,-1)

def load_hires_constants(constants_path, batch_size=1):
    df = xr.load_dataset(f"{constants_path}.nc")
    # Should rewrite this file to have increasing latitudes
    z = np.array(df['Z'])[:,::-1,:]
    # Normalise orography by max
    z = z/z.max()
    # LSM is already 0:1
    lsm = np.array(df['LSM'])[:,::-1,:]
    return np.repeat(np.stack([z,lsm],-1),batch_size,axis=0)

def load_era_nimrod_batch(batch_dates, era_path, nimrod_path, constants_path, era_fields, log_precip=False,
                          constants=False, hour=0, era_norm=False, era_crop=2, norm_year=2016):
    batch_x = [] 
    batch_y = []
    if hour=='random':
        hours = np.random.randint(24,size=[len(batch_dates)])
    elif type(hour)==int:
        hours = len(batch_dates)*[hour]
    
    for i,date in enumerate(batch_dates):
        h = hours[i]
        batch_x.append(load_erastack(date, era_path, era_fields, h, log_precip=log_precip, era_norm=era_norm, era_crop=era_crop, norm_year=norm_year))
        batch_y.append(load_nimrod(date, nimrod_path, h, log_precip=log_precip))
    
    ## load_nimrod returns a (1, 951, 951) tuple, so we expand the dimensions to (1, 951, 951, 1)
    ## this helps TensorFlow match up the dimensions
    ## axis is -1 because we will always add the last dimension
    batch_y = np.expand_dims(np.array(batch_y), axis=-1)
    if type(constants) is bool:
        if (not constants):
            return np.array(batch_x), batch_y
        else:
            return [np.array(batch_x), load_hires_constants(constants_path, len(batch_dates))], batch_y
    elif constants is not None:
        return [np.array(batch_x), constants], batch_y
        

def logprec(y,log_precip=True):
    if log_precip:
        return np.log10(1+y)
    else:
        return y

def repeat_upscale(data,upscale_factor=25):
    assert len(data.shape) == 4
    crop = int(np.floor(upscale_factor/2))
    reshaped_inputs = np.repeat(np.repeat(data,upscale_factor,axis=-2),upscale_factor,axis=-3)
    return reshaped_inputs[:,crop:-crop,crop:-crop,:]

