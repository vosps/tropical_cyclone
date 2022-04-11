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
variable = 'u'
level = '200'
variable_name = 'u_component_of_wind'

# change dataset depending on single or pressure levels
dataset = 'reanalysis-era5-single-levels'
dataset = 'reanalysis-era5-pressure-levels'

outdir = '/bp1store/geog-tropical/data/ERA-5/hour'
tmpdir = '/bp1store/geog-tropical/data/ERA-5/tmp'
print(level)
print(variable)
print(variable_name)

if not os.path.exists(tmpdir):
    os.mkdir(tmpdir)

def generate_yrmonths():
	# 1979 - 2020
	years = range(1979,2022)
	print(list(years))
	months = ['01','02','03','04','05','06','07','08','09','10','11','12']
	yrmonths = [ int("%s%s" % (year,month)) for year in years for month in months]
	# return yrmonths[:-1]
	return yrmonths

# List of months is produced from running the script ~pfu599/src/ERA5_process/check_months.py
yrmonths = generate_yrmonths()
# time = [
# 		'00:00', '01:00', '02:00',
# 		'03:00', '04:00', '05:00',
# 		'06:00', '07:00', '08:00',
# 		'09:00', '10:00', '11:00',
# 		'12:00', '13:00', '14:00',
# 		'15:00', '16:00', '17:00',
# 		'18:00', '19:00', '20:00',
# 		'21:00', '22:00', '23:00',
# 				]
time = [
		'00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00',
		]
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

	tmpfile =  os.path.join(tmpdir,'ERA5_'+variable+level+'_hrly_'+year+mon+'.nc')
	tmpfile_day =  os.path.join(tmpdir,'ERA5_'+variable+'_day_'+year+mon+'.nc')

	request = {
				'product_type': 'reanalysis',
				'format': 'netcdf',
				'variable': variable_name,
				# 'pressure_level': level,
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
				'time': time
				,}

	print(request)
	# download single hourly dataset
	c.retrieve(dataset,request,tmpfile)
	print('Downloaded',tmpfile)

	# convert to daily mean and save to "ftas" outfile location
	if regrid == False:
		ftas = os.path.join(outdir,variable,'ERA5_'+variable+'_day_'+year+mon+'.nc') # normal resolution
		cdo_cmd = ['cdo','-O','-b','F32','daymean',tmpfile,ftas] # normal resolution
		print(' '.join(cdo_cmd))
		ret = subprocess.call(cdo_cmd)
		if not ret==0:
			raise Exception('Error with cdo command')
	
	# save as above and then regrid to 1 degree resolution
	else:
		ftas = os.path.join(outdir,variable,level,'ERA5_'+variable+'_3hourly_1deg_'+year+mon+'.nc') # 1 deg res
		cdo_cmd = ['cdo','remapbil,r360x180',tmpfile,ftas]
		print(' '.join(cdo_cmd))
		ret = subprocess.call(cdo_cmd)
		if not ret==0:
			raise Exception('Error with cdo command')

	# delete tmpfile
	os.remove(tmpfile)
	print('Finished')
			