
import subprocess,os
from pathlib import Path

def regrid(fp):
	new_fp = fp[:-3] + '_regrid.nc'
	cdo_cmd = ['cdo','remapbil,mygrid',fp,new_fp]
	print(' '.join(cdo_cmd))
	# ret = subprocess.call(cdo_cmd)
	ret = subprocess.run(cdo_cmd)
	# if not ret==0:
	# 	raise Exception('Error with cdo command')
	return new_fp

def handle_regrid(fp):
	if Path(fp[:-3] + '_regrid.nc').is_file():
		print(fp[:-3] + '_regrid.nc')
		regrid_fp = fp[:-3] + '_regrid.nc'				
		print('file already regridded')
	else:
		print('regridding file: ',fp)
		regrid_fp = regrid(fp)
		print('regridded ', fp)

	return regrid_fp

model = 'NESM3'
scenario = 'historical'
rainfall_dir = f'/bp1/geog-tropical/data/CMIP6/CMIP6-rain/{model}/pr/{scenario}/'

for f in os.listdir(rainfall_dir):
	rainfall_fp = rainfall_dir + f
	regrid_rainfall_fp = handle_regrid(rainfall_fp)
