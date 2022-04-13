"""
download cmip6 data
"""
import subprocess


url = 'http://esgf-data2.diasjp.net/thredds/fileServer/esg_dataroot/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/E1hr/pr/gn/v20191114/pr_E1hr_MIROC6_historical_r1i1p1f1_gn_197901010030-197912312330.nc'

urls = ['http://esgf-data2.diasjp.net/thredds/fileServer/esg_dataroot/CMIP6/CMIP/MIROC/MIROC6/historical/r1i1p1f1/E1hr/pr/gn/v20191114/pr_E1hr_MIROC6_historical_r1i1p1f1_gn_%s01010030-%s12312330.nc' % (year,year) for year in range(2010,2021)]
fps = ['/user/home/al18709/work/CMIP6/MIROC6/hour/pr/pr_E1hr_MIROC6_historical_r1i1p1f1_gn_%s01010030-%s12312330.nc' % (year,year) for year in range(2010,2021)]

urls = ['http://esgf-data3.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/HighResMIP/EC-Earth-Consortium/EC-Earth3P/hist-1950/r1i1p2f1/3hr/pr/gr/v20190314/pr_3hr_EC-Earth3P_hist-1950_r1i1p2f1_gr_%s01010000-%s12312100.nc' % (year,year) for year in range(2010,2021)]
fps = ['/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/pr_3hr_EC-Earth3P_hist-1950_r1i1p2f1_gr_%s01010000-%s12312100.nc' % (year,year) for year in range(2004,2021)]

# for i,url in enumerate(urls):
# 	fp = fps[i]
# 	wget_cmd = ['wget','-O',fp,url] # normal resolution
# 	print(' '.join(wget_cmd))
# 	ret = subprocess.call(wget_cmd)
# 	if not ret==0:
# 		raise Exception('Error with wget command')

tracks = 'https://dap.ceda.ac.uk/badc/highresmip-derived/data/storm_tracks/TRACK/EC-Earth-Consortium/EC-Earth3P/hist-1950/r1i1p2f1/tropical/v4/TC-NH_TRACK_EC-Earth3P_hist-1950_r1i1p2f1_gr_19500101-20141231.nc'
fp = '/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/TC-NH_TRACK_EC-Earth3P_hist-1950_r1i1p2f1_gr_19500101-20141231.nc'

wget_cmd = ['wget','-O',fp,tracks] # normal resolution
print(' '.join(wget_cmd))
ret = subprocess.call(wget_cmd)
if not ret==0:
	raise Exception('Error with wget command')