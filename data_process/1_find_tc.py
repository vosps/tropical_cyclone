"""
1. Find files in IMERG or MSWEP dataset which correspond to tropical cyclone events

This script returns a csv file of all TCs in IMERG or MSWEP files using ibtracks data to locate them

column docs for ibtracks: https://www.ncdc.noaa.gov/ibtracs/pdf/IBTrACS_v04_column_documentation.pdf 

"""
# import modules
import glob
import pandas as pd
from netCDF4 import Dataset
import numpy as np

# open csv file
ibtracks = pd.read_csv('/user/work/al18709/ibtracks/ibtracs.ALL.list.v04r00.csv',
						usecols=['SID','LAT','LON','BASIN','NAME','SEASON', 'NATURE','ISO_TIME','USA_SSHS'],
						parse_dates = ['ISO_TIME'],keep_default_na=False)

# tidy up columns with multiple dtypes
ibtracks = ibtracks.iloc[1: , :]
ibtracks = ibtracks.replace(' ', np.nan)
ibtracks['USA_SSHS'] = pd.to_numeric(ibtracks['USA_SSHS'])
ibtracks['SEASON'] = pd.to_numeric(ibtracks['SEASON'])

# subset storms since 2000
ibtracks = ibtracks[ibtracks['SEASON'] > 2000] #TODO: change so doesn't include 1999

# select tropical storms
ibtracks = ibtracks[ibtracks['NATURE'] == 'TS']
ibtracks = ibtracks[ibtracks['USA_SSHS'] >= 1]

# extract datetime data
ibtracks['ISO_TIME'] = pd.to_datetime(ibtracks['ISO_TIME'])
year_month_day = list(pd.to_datetime(ibtracks['ISO_TIME']).dt.strftime('%Y%m%d'))
day_of_year = ibtracks['ISO_TIME'].dt.dayofyear
for d in list(range(1,10)):

	day_of_year[day_of_year == d] = '00' + str(d)	
for d in list(range(10,100)):
	day_of_year[day_of_year == d] = '0' + str(d)
year_day = [str(y) + str(d) for y,d in zip(list(pd.to_datetime(ibtracks['ISO_TIME']).dt.year),day_of_year)]

# extract time
hours = ibtracks['ISO_TIME'].dt.hour
hours_mswep = ibtracks['ISO_TIME'].dt.hour
mswep_indices = []

for hr in [0,3,6,9]:
	hours_mswep[hours_mswep == hr] = '0' + str(hr)

hours[hours == 0] = 24
hours = hours - 1
hours = hours.astype(str).str.zfill(2)
time_points = [str(h) + '5959' for h in hours]

# generate list of filepaths
filepaths_imerg = ['/bp1store/geog-tropical/data/Obs/IMERG/half_hourly/final/3B-HHR.MS.MRG.3IMERG.%s*%s*.HDF5' % (ymd,h) for ymd,h in zip(year_month_day,time_points)]
filepaths_mswep = ['/bp1store/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/%s.%s.nc' % (yd,h) for yd,h in zip(year_day,hours_mswep)]

# generate lat + lon for centroid of TC
lat = list(ibtracks['LAT'])
lon = list(ibtracks['LON'])

# keep other relevant information for csv file
name = list(ibtracks['NAME'])
basin = list(ibtracks['BASIN']) 
sshs = list(ibtracks['USA_SSHS'])
sids = list(ibtracks['SID'])

# write csv file
tc_files = pd.DataFrame({
						 'filepath_imerg' : filepaths_imerg, 'filepath_mswep' : filepaths_mswep, 'sid' : sids, 'lat' : lat, 'lon' : lon,
						 'name' : name, 'basin' : basin, 'sshs' : sshs, 'hour' : hours
})

# reindex the dataframe and drop old index column
tc_files = tc_files.reset_index(drop=True)
print(tc_files)

tc_files.to_csv('/user/work/al18709/ibtracks/tc_files.csv')
print(tc_files['filepath_imerg'][86])
print(tc_files['filepath_mswep'][86])


