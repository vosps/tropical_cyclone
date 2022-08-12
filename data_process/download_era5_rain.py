# This script gets ERA5 temperature data that is missing from the /badc store
#
# Uses conda environment 'ecmwf-cds', within jaspy
# Setup by:
# export PATH=/apps/contrib/jaspy/miniconda_envs/jaspy3.7/m3-4.6.14/bin:$PATH
# source activate jaspy3.7-m3-4.6.14-r20190627
# Author: Peter Uhe 21/01/2020
# Changes made: Emily Vosper 28/07/2021
# cdo tutorial northern hemisphere https://code.mpimet.mpg.de/projects/cdo/wiki/Tutorial
# remap https://nicojourdain.github.io/students_dir/students_netcdf_cdo/
import os,glob
import cdsapi
import subprocess
# import utils 


regrid = 'True'
variable = 'precipitation'
level = 'surface'
variable_name = 'total_precipitation'

# change dataset depending on single or pressure levels
dataset = 'reanalysis-era5-single-levels'

outdir = '/bp1store/geog-tropical/data/ERA-5/hour/precipitation/'
# tmpdir = '/bp1store/geog-tropical/data/ERA-5/tmp'
print(level)
print(variable)
print(variable_name)


def generate_yrmonths():
	# 1979 - 2020
	years = range(1980,2022)
	print(list(years))
	months = ['01','02','03','04','05','06','07','08','09','10','11','12']
	yrmonths = [ int("%s%s" % (year,month)) for year in years for month in months]
	# return yrmonths[:-1]
	return yrmonths

# List of months is produced from running the script ~pfu599/src/ERA5_process/check_months.py
yrmonths = generate_yrmonths()
time = [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ],

print(yrmonths)
c = cdsapi.Client()

for yrmonth in yrmonths:
	year = str(yrmonth)[:4]
	mon  = str(yrmonth)[4:6]

	# Check if the files (for tas, tasmin, tasmax) are already there:
	# outfiles = glob.glob(os.path.join(outdir,'ERA5_tas*_day_'+year+mon+'.nc'))
	# if len(outfiles) == 3:
	# 	print('Files exist, skipping')
	# 	continue

	print('Processing:',year,mon)

	outfile =  os.path.join(outdir,'ERA5_'+variable+'_3hrly_'+year+mon+'.nc')
	# tmpfile_day =  os.path.join(tmpdir,'ERA5_'+variable+'_day_'+year+mon+'.nc')

	request = {
				'product_type': 'reanalysis',
				'format': 'netcdf',
				'variable': variable_name,
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
				'time': [
					'00:00', '03:00', '06:00',
					'09:00', '12:00', '15:00',
					'18:00', '21:00',
				],
				}

	print(request)
	# download single hourly dataset
	c.retrieve(dataset,request,outfile)
	print('Downloaded',outfile)

	
print('Finished')

# c = cdsapi.Client()
# directory = '/bp1store/geog-tropical/data/ERA-5/hour/precipitation/'
# request = {
#         'product_type': 'reanalysis',
#         'format': 'netcdf',
#         'variable': 'total_precipitation',
#         'year': [
#             '1979', '1980', '1981',
#             '1982', '1983', '1984',
#             '1985', '1986', '1987',
#             '1988', '1989', '1990',
#             '1991', '1992', '1993',
#             '1994', '1995', '1996',
#             '1997', '1998', '1999',
            
#         ],
#         'month': [
#             '01', '02', '03',
#             '04', '05', '06',
#             '07', '08', '09',
#             '10', '11', '12',
#         ],
#         'day': [
#             '01', '02', '03',
#             '04', '05', '06',
#             '07', '08', '09',
#             '10', '11', '12',
#             '13', '14', '15',
#             '16', '17', '18',
#             '19', '20', '21',
#             '22', '23', '24',
#             '25', '26', '27',
#             '28', '29', '30',
#             '31',
#         ],
#         'time': [
#             '00:00', '03:00', '06:00',
#             '09:00', '12:00', '15:00',
#             '18:00', '21:00',
#         ],
#     }

# c.retrieve('reanalysis-era5-single-levels',request,directory)

	