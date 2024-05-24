"""
1. Find files in IMERG or MSWEP dataset which correspond to tropical cyclone events

This script returns a csv file of all TCs in IMERG or MSWEP files using ibtracks data to locate them

column docs for ibtracks: https://www.ncdc.noaa.gov/ibtracs/pdf/IBTrACS_v04_column_documentation.pdf 

"""
# import modules
import glob
import pandas as pd
# from netCDF4 import Dataset
import numpy as np
import sys

variable = sys.argv[1]
print('variable: ', variable)
if '/' in variable:
	var = variable.split('/')[0]
else: 
	var = variable

print('hello')

# # open csv file
# ibtracks = pd.read_csv('/user/work/al18709/ibtracks/ibtracs.ALL.list.v04r00.csv',
# 						usecols=['SID','LAT','LON','BASIN','NAME','SEASON', 'NATURE','ISO_TIME','USA_SSHS'],
# 						parse_dates = ['ISO_TIME'],keep_default_na=False)

# # tidy up columns with multiple dtypes
# ibtracks = ibtracks.iloc[1: , :]
# ibtracks = ibtracks.replace(' ', np.nan)
# ibtracks['USA_SSHS'] = pd.to_numeric(ibtracks['USA_SSHS'])
# ibtracks['SEASON'] = pd.to_numeric(ibtracks['SEASON'])

# # subset storms since 2000
# # ibtracks = ibtracks[ibtracks['SEASON'] > 2000] #TODO: change so doesn't include 1999
# ibtracks = ibtracks[ibtracks['SEASON'] >= 1979] # changed for mswep data
# ibtracks = ibtracks[ibtracks['SEASON'] <= 2014] 

# # select tropical storms that reach TC strength
# ibtracks = ibtracks[ibtracks['NATURE'] == 'TS'] #TODO: uncomment this usually

# # ibtracks = ibtracks[ibtracks['USA_SSHS'] >= 1] # TODO: change
# # select only storms that strengthen to TC strength at any point
# TCs = ibtracks[ibtracks['USA_SSHS'] >= 0]['SID']
# TCs = TCs.drop_duplicates()
# print('number of TCs: ', len(TCs))
# # reference ibtracks with TCs
# ibtracks = pd.merge(ibtracks, 
#                       TCs, 
#                       on ='SID', 
#                       how ='inner')
# # TODO: add another filter so you only collect storms that are tropical so -1+ we're not interested in sub or extra tropical storms
# ibtracks = ibtracks[ibtracks['USA_SSHS'] >= 0]
# # if no filter on track strength then we have 134,400 images
# if >= -1 then we have 131,377 images
# if >=0 then we have 99,741 images
# if >= 1 then we have ~50,000 images

tracks = pd.read_csv('/user/work/al18709/ibtracks/ibtracs.ALL.list.v04r00.csv',
						usecols=['SID','LAT','LON','BASIN','NAME','SEASON', 'NATURE','ISO_TIME','USA_SSHS'],
						parse_dates = ['ISO_TIME'],keep_default_na=False)

# tidy up columns with multiple dtypes
tracks = tracks.iloc[1: , :]
tracks = tracks.replace(' ', np.nan)
tracks['USA_SSHS'] = pd.to_numeric(tracks['USA_SSHS'])
tracks['SEASON'] = pd.to_numeric(tracks['SEASON'])

tracks = tracks[tracks['SEASON'] >= 2000]
tracks = tracks[tracks['SEASON'] <= 2014]
# tracks = tracks[tracks['SEASON'] == 2000]
print(tracks)
# tracks = tracks[tracks['ISO_TIME'].dt.dayofyear >= 153]
# tracks = tracks[tracks['ISO_TIME'].dt.dayofyear <= 158]
tracks = tracks[tracks['NATURE'] == 'TS']

TCs = tracks[tracks['USA_SSHS'] >= 0]['SID']
TCs = TCs.drop_duplicates()
print('number of TCs: ', len(TCs))
# reference tracks with TCs
tracks = pd.merge(tracks, 
                      TCs, 
                      on ='SID', 
                      how ='inner')
# tracks = tracks[tracks['USA_SSHS'] >= 1]
tracks = tracks[tracks['USA_SSHS'] >= 0]
print(tracks)
print(tracks['SID'].drop_duplicates())

ibtracks = tracks

print(len(ibtracks))
print('actual number of TCs',len(ibtracks['SID'].drop_duplicates()))



# extract datetime data
ibtracks['ISO_TIME'] = pd.to_datetime(ibtracks['ISO_TIME'])
year_month_day = list(pd.to_datetime(ibtracks['ISO_TIME']).dt.strftime('%Y%m%d'))
year_month = list(pd.to_datetime(ibtracks['ISO_TIME']).dt.strftime('%Y%m'))
years = list(pd.to_datetime(ibtracks['ISO_TIME']).dt.strftime('%Y'))
months = list(pd.to_datetime(ibtracks['ISO_TIME']).dt.strftime('%m'))
days = list(pd.to_datetime(ibtracks['ISO_TIME']).dt.strftime('%d'))
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
hours_era5 = hours
hours[hours == 0] = 24
hours = hours - 1
hours = hours.astype(str).str.zfill(2)
time_points = [str(h) + '5959' for h in hours]

# generate list of filepaths
filepaths_imerg = ['/bp1/geog-tropical/data/Obs/IMERG/half_hourly/final/3B-HHR.MS.MRG.3IMERG.%s*%s*.HDF5' % (ymd,h) for ymd,h in zip(year_month_day,time_points)]
filepaths_mswep = ['/bp1/geog-tropical/data/Obs/MSWEP/3hourly_invertlat/%s.%s.nc' % (yd,h) for yd,h in zip(year_day,hours_mswep)]
filepaths_era5 = ['/bp1/geog-tropical/data/ERA-5/hour/precipitation_invertlat_em/ERA5_precipitation_3hrly_%s.nc' % ym for ym in year_month]
if variable == "t":
	filepaths_var = ['/user/home/al18709/work/topography/topography_10km_nn.nc'] * len(filepaths_mswep)
else:
	filepaths_var = [f'/bp1/geog-tropical/data/ERA-5/hour/{variable}_invertlat/ERA5_{var}_3hourly_1deg_{ym}.nc' for ym in year_month]

# generate lat + lon for centroid of TC
lat = list(ibtracks['LAT'])
lon = list(ibtracks['LON'])

# keep other relevant information for csv file
name = list(ibtracks['NAME'])
basin = list(ibtracks['BASIN']) 
sshs = list(ibtracks['USA_SSHS'])
sids = list(ibtracks['SID'])
hours_era5[hours_era5 == 24] = '00'

# write csv file
tc_files = pd.DataFrame({
						 'filepath_imerg' : filepaths_imerg, 
						 'filepath_mswep' : filepaths_mswep, 
						 'filepath_era5' : filepaths_era5,
						 'filepath_var' : filepaths_var,
						 'sid' : sids, 
						 'lat' : lat, 
						 'lon' : lon,
						 'name' : name, 
						 'basin' : basin, 
						 'sshs' : sshs,
						 'year' : years,
						 'month' : months,
						 'day' : days,
						 'hour' : hours_era5
})

# reindex the dataframe and drop old index column
tc_files = tc_files.reset_index(drop=True)
print(tc_files)

tc_files.to_csv('/user/work/al18709/ibtracks/tc_files_tcs_and_storms.csv')
# tc_files.to_csv('/user/work/al18709/ibtracks/tc_files_all.csv')
print(tc_files['filepath_imerg'][86])
print(tc_files['filepath_mswep'][86])
print(tc_files['filepath_var'][86])
print(tc_files.shape)

