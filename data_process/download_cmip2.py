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
# scp jasmin-sci:/badc/cmip6/data/PRIMAVERA/HighResMIP/MPI-M/MPI-ESM1-2-HR/hist-1950/r1i1p1f1/Prim6hr/pr/gn/latest/*.nc /pr

# scp jasmin-sci:/gws/nopw/j04/cmip6_track/CMIP6-TRACK/MPI/MPI-ESM1-2-HR/historical/TC/NH/*.nc

# scp jasmin-sci:/home/users/vosps/MIROC6/historical/NH/*_newtime2.nc NH
# scp jasmin-sci:/home/users/vosps/MIROC6/historical/SH/*_newtime2.nc SH

# scp jasmin-sci:/badc/cmip6/data/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/6hrPlev/pr/gn/latest/pr_6hrPlev_MIROC6_historical_r1i1p1f1_gn_198201010300-198212312100.nc pr_6hrPlev_MIROC6_historical_r1i1p1f1_gn_198201010300-198212312100.nc


# scp jasmin-sci:/badc/cmip6/data/PRIMAVERA/HighResMIP/CMCC/CMCC-CM2-VHR4/highres-future/r1i1p1f1/Prim6hr/pr/gn/latest/*.nc ssp585

# /bp1/geog-tropical/data/CMIP6/HighResMIP/MRI/MRI-AGCM3-2-H/highresSST-future/r1i1p1f1/fx/areacella/gn/latest

# scp jasmin-sci:/badc/cmip6/data/PRIMAVERA/HighResMIP/CMCC/CMCC-CM2-HR4/hist-1950/r1i1p1f1/Prim6hr/pr/gn/latest/*.nc /hist-1950/r1i1p1f1/Prim6hr/pr/gn/latest
# scp jasmin-sci:/badc/cmip6/data/PRIMAVERA/HighResMIP/CMCC/CMCC-CM2-HR4/highres-future/r1i1p1f1/Prim6hr/pr/gn/latest/*.nc latest/

# scp jasmin-sci:/badc/cmip6/data/PRIMAVERA/HighResMIP/MPI-M/MPI-ESM1-2-HR/hist-1950/r1i1p1f1/Prim6hr/pr/gn/latest/*.nc latest/
# scp jasmin-sci:/badc/cmip6/data/PRIMAVERA/HighResMIP/MPI-M/MPI-ESM1-2-HR/highres-future/r1i1p1f1/Prim6hr/pr/gn/latest/*.nc latest/

# scp jasmin-sci:/badc/cmip6/data/CMIP6/CMIP/NCC/NorESM2-LM/historical/r1i1p1f1/6hrPlev/pr/gn/latest/*.nc historical
# scp jasmin-sci:/badc/cmip6/data/CMIP6/ScenarioMIP/NCC/NorESM2-LM/ssp585/r1i1p1f1/6hrPlev/pr/gn/latest/*.nc ssp585

# scp jasmin-sci:/badc/cmip6/data/CMIP6/CMIP/AS-RCEC/TaiESM1/historical/r1i1p1f1/6hrPlev/pr/gn/latest/*.nc historical/
# scp jasmin-sci:/badc/cmip6/data/CMIP6/ScenarioMIP/AS-RCEC/TaiESM1/ssp585/r1i1p1f1/6hrPlev/pr/gn/latest/*.nc ssp585/


# scp jasmin-sci:/home/users/vosps/MPI-ESM1-2-LR/historical/NH/*newtime2.nc NH
# scp jasmin-sci:/home/users/vosps/MPI-ESM1-2-LR/historical/SH/*newtime2.nc SH

