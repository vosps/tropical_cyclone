import gdal

inputfile = '/user/home/al18709/work/population/ppp_2020_1km_Aggregated.tif'
outputfile = '/user/home/al18709/work/population/ppp_2020_1km_Aggregated.nc'
#Do not change this line, the following command will convert the geoTIFF to a netCDF
ds = gdal.Translate(outputfile, inputfile, format='NetCDF')