
# import gdal
from osgeo import gdal

# inputfile = '/user/home/al18709/work/population/ppp_2020_1km_Aggregated.tif'
# outputfile = '/user/home/al18709/work/population/ppp_2020_1km_Aggregated.nc'
inputfile = '/user/home/al18709/work/population/SSP2_2085.tif'
outputfile = '/user/home/al18709/work/population/SSP2_2085.nc'
#Do not change this line, the following command will convert the geoTIFF to a netCDF
ds = gdal.Translate(outputfile, inputfile, format='NetCDF')


# regrid

# cdo remapnn,mygrid /user/home/al18709/work/population/SSP2_2085_final.nc /user/home/al18709/work/population/SSP2_2085_10km_final.nc

# cdo remapnn,mygrid /user/home/al18709/work/population/ppp_2020_1km_Aggregated_final.nc /user/home/al18709/work/population/ppp_2020_10km_Aggregated_final.nc