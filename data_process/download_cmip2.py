"""
download highresmip data from jasmin
"""

import subprocess

bp_dir = '/user/home/al18709/work/CMIP6/HighResMIP/CMCC-CM2-VHR4/historical/pr/'
primavera_directory = '/badc/cmip6/data/PRIMAVERA/HighResMIP/CMCC/CMCC-CM2-VHR4/hist-1950/r1i1p1f1/Prim6hr/pr/gn/v20180705/'

def generate_yrmonths():
	# 1979 - 2020
	years = range(1979,2023)
	print(list(years))
	months = ['01','02','03','04','05','06','07','08','09','10','11','12']
	yrmonths = [ "%s%s" % (year,month) for year in years for month in months]
	return yrmonths

yrmonths=generate_yrmonths()

for ym in yrmonths:
	filename = f'pr_Prim6hr_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_{ym}010000-{ym}??1800.nc'
	filename2 = f'pr_Prim6hr_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_{ym}-{ym}.nc'

	jasmin_filename = f'jasmin-sci:{primavera_directory}{filename}'
	bp_filename = f'{bp_dir}{filename2}'

	cmd = ['scp',jasmin_filename,bp_filename]

	ret = subprocess.call(cmd)
	if not ret==0:
		raise Exception('Error with command')



# scp jasmin-sci:/badc/cmip6/data/PRIMAVERA/HighResMIP/CMCC/CMCC-CM2-VHR4/hist-1950/r1i1p1f1/Prim6hr/pr/gn/v20180705/pr_Prim6hr_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_197901010000-197901311800.nc pr_Prim6hr_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_197901010000-197901311800.nc
# scp jasmin-sci:/home/users/vosps/MIROC6/historical/NH/*_newtime2.nc NH
# scp jasmin-sci:/home/users/vosps/MIROC6/historical/SH/*_newtime2.nc SH