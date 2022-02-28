# This script gets ERA5 temperature data that is missing from the /badc store
#
# Uses conda environment 'ecmwf-cds', within jaspy
# Setup by:
# export PATH=/apps/contrib/jaspy/miniconda_envs/jaspy3.7/m3-4.6.14/bin:$PATH
# source activate jaspy3.7-m3-4.6.14-r20190627
# Peter Uhe
# 2020/01/21
# cdo tutorial northern hemisphere https://code.mpimet.mpg.de/projects/cdo/wiki/Tutorial
# remap https://nicojourdain.github.io/students_dir/students_netcdf_cdo/
import os,glob
import cdsapi
import subprocess
# import utils
import time
variable = 'surface_pressure'
dataset = 'reanalysis-era5-single-levels'
outdir = '/bp1store/geog-tropical/data/ERA-5/day'
# download to /tmp Blue Pebble store as quicker
tmpdir = '/tmp'
print(variable)
if not os.path.exists(tmpdir):
	os.mkdir(tmpdir)
# List of months is produced from running the script ~pfu599/src/ERA5_process/check_months.py
def generate_yrmonths():
		# years = range(1979,2021)
		years = range(2021,2023)
		months = ['01','02','03','04','05','06','07','08','09','10','11','12']
		yrmonths = [ int("%s%s" % (year,month)) for year in years for month in months]
		return yrmonths

yrmonths = generate_yrmonths()
yrmonths = yrmonths[3:]
# c = cdsapi.Client()
c = cdsapi.Client(timeout=600,quiet=False,debug=True)
for yrmonth in yrmonths:
	year = str(yrmonth)[:4]
	mon  = str(yrmonth)[4:6]
	print('Processing:',year,mon)
	tmpfile =  os.path.join(tmpdir,'ERA5_'+variable+'_hrly_'+year+mon+'.nc')
	#tmpfile_day =  os.path.join(tmpdir,'ERA5_'+variable+'_day_'+year+mon+'.nc')
	request = {
				'product_type': 'reanalysis',
				'format': 'netcdf',
				'variable': 'surface_pressure',
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
					'00:00', '01:00', '02:00',
					'03:00', '04:00', '05:00',
					'06:00', '07:00', '08:00',
					'09:00', '10:00', '11:00',
					'12:00', '13:00', '14:00',
					'15:00', '16:00', '17:00',
					'18:00', '19:00', '20:00',
					'21:00', '22:00', '23:00',
				],}
	print(request)
	start = time.time()
	c.retrieve(dataset,request,tmpfile)
	end = time.time()
	print ("Time to download file (secs):", end - start)
	print('Downloaded',tmpfile)
	ftas = os.path.join(outdir,variable,'ERA5_'+variable+'_day_'+year+mon+'.nc')
	cdo_cmd = ['cdo','-O','-b','F32','daymean',tmpfile,ftas]
	print(' '.join(cdo_cmd))
	ret = subprocess.call(cdo_cmd)
	if not ret==0:
		raise Exception('Error with cdo command')
	#cdo_cmd = ['cdo','remapbil,r360x180',tmpfile_day,ftas]
	#print(' '.join(cdo_cmd))
	#ret = subprocess.call(cdo_cmd)
	#if not ret==0:
	#   raise Exception('Error with cdo command')
	#os.remove(tmpfile_day)
	os.remove(tmpfile)
	print('Finished')