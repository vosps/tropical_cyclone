# This script gets ERA5 10m u and v compponent of wind
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
#
# This script utilises cdsapi: https://pypi.org/project/cdsapi/ 
# To access netcdf data use xarray: https://docs.xarray.dev/en/stable/ 

import os
import cdsapi
import subprocess


# generate numbers in the form of yearmonth to help loop through downloads
def generate_yrmonths():
	years = range(2018,2024) # change the years you want to download here
	print(list(years))
	months = ['01','02','03','04','05','06','07','08','09','10','11','12']
	yrmonths = [ int("%s%s" % (year,month)) for year in years for month in months]
	return yrmonths

# define useful variables
outdir = '/path/to/where/you/want/to/save/data/'

yrmonths = generate_yrmonths()
time = ['00:00', '01:00', '02:00',
        '03:00', '04:00', '05:00',
        '06:00', '07:00', '08:00',
        '09:00', '10:00', '11:00',
        '12:00', '13:00', '14:00',
        '15:00', '16:00', '17:00',
        '18:00', '19:00', '20:00',
        '21:00', '22:00', '23:00',
		]
variables = ['10m_u_component_of_wind', '10m_v_component_of_wind']

# initiate client
c = cdsapi.Client()

for variable in variables:
	for yrmonth in yrmonths:
		year = str(yrmonth)[:4]
		mon  = str(yrmonth)[4:6]
		print('Downloading:',year,mon)
		outfile = f'ERA5_UK_hourly_{variable}_{year}_{mon}.nc'

		request = {
					'product_type': 'reanalysis',
					'format': 'netcdf',
					'variable': variable,
					'year': year,
					'month': mon,
					'day': [
						'01', '02', '03',
						'04', '05', '06',
						'07', '08', '09',
						'10', '11', '12',
						'13', '14', '15',
						'16', '17', '18',
						'19', '20', '21',
						'22', '23', '24',
						'25', '26', '27',
						'28', '29', '30',
						'31',
					],		
					'time': time,
					'area': [60.9, -8.74, 49.81, 1.84,
			],}

		print(request)
		c.retrieve('reanalysis-era5-single-levels',request,outdir)
		print('Downloaded!')

		



		