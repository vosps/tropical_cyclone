import numpy as np
from netCDF4 import Dataset

def flip(tc):
	tc_flipped = np.flip(tc,axis=0)
	return tc_flipped

def find_and_flip(centre_lat,centre_lon,data):
	print(centre_lat)
	sh_indices = (centre_lat < 0)
	nh_indices = (centre_lat > 0)
	print(nh_indices)

	# flip the nh tcs so they are rotating anticlockwise (in the raw data array)
	# in mswep they can be plotted correctly with the mswep lats and lons, but the raw data shows them flipped as mswep has descending latitudes
	if nh_indices:
		topography_flipped = flip(data)

	else:
		topography_flipped = data
	
	return topography_flipped


def fix_dimensions(lat,lon,data):
	print('fixing dimensions...')
	if len(lon) == 99:
		print('lon too long')
		lon = np.append(lon,lon[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,np.array([data[:,-1]]).T),axis=1)
	if len(lat) == 99:
		print('lat too long')
		lat = np.append(lat,lat[-1])
		print(data[-1,:].shape)
		print(data[:,-1].shape)
		print(data.shape)
		data = np.concatenate((data,[data[-1,:]]),axis=0)
	if len(lon) == 101:
		lon=lon[:-1]
		data = data[:,:-1]
	if len(lat) == 101:
		lat = lat[:-1]
		data = data[:-1,:]
		
	print(lon.shape,lat.shape,data.shape)
	return lat,lon,data

def generate_topography(centre_lat,centre_lon,i):

	# open file
	filepath = '/user/work/al18709/topography/topography_10km_nn.nc'
	d = Dataset(filepath, 'r')
	lat = d.variables['lat'][:] #lat
	lon = d.variables['lon'][:] #lon
		
	# clip to location
	lat_lower_bound = (np.abs(lat-centre_lat+5.)).argmin()
	lat_upper_bound = (np.abs(lat-centre_lat-5.)).argmin()
	lon_lower_bound = (np.abs(lon-centre_lon+5.)).argmin()
	lon_upper_bound = (np.abs(lon-centre_lon-5.)).argmin()

	if centre_lon > 175: 
		print('goes over centre')
		diff = lon_upper_bound - lon_lower_bound
		second_upper_bound = 100 - diff

		data1 = d.variables['z'][lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
		data2 = d.variables['z'][lat_lower_bound:lat_upper_bound,0:second_upper_bound]
		lat = lat[lat_lower_bound:lat_upper_bound]
		lon1 = lon[lon_lower_bound:lon_upper_bound]
		lon2 = lon[0:second_upper_bound]
		data = np.concatenate((data1,data2),axis=1)
		lon = np.concatenate((lon1,lon2))
	elif centre_lon < -175:
		diff = lon_upper_bound - lon_lower_bound
		second_upper_bound = 100 - diff
		data1 = d.variables['z'][lat_lower_bound:lat_upper_bound,-second_upper_bound:-1]
		data2 = d.variables['z'][lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
		lat = lat[lat_lower_bound:lat_upper_bound]
		lon1 = lon[-second_upper_bound:-1]
		lon2 = lon[lon_lower_bound:lon_upper_bound]
		
		data = np.concatenate((data1,data2),axis=1)
		lon = np.concatenate((lon1,lon2))
	else:
		data = d.variables['z'][lat_lower_bound:lat_upper_bound,lon_lower_bound:lon_upper_bound]
		lat = lat[lat_lower_bound:lat_upper_bound]
		lon = lon[lon_lower_bound:lon_upper_bound]

	d.close()
	if (len(lon) != 100) or (len(lat) != 100): # TODO: figure out why this happens
		print('dimensions do not match')
		lat,lon,data = fix_dimensions(lat,lon,data)
	
	data = find_and_flip(centre_lat,centre_lon,data)

	print('centre_lat',centre_lat)
	print('centre_lon',centre_lon)
	print(data.shape)
	tc_topography[i,:,:] = data


# load data
data = np.load('/user/work/al18709/tc_data_flipped/KE_tracks/ke_miroc6-ssp585.npy')
centre_lats = data[:,4]
centre_lons = data[:,5]
tc_topography = np.zeros((data.shape[0],100,100))

# 2. extract TC
for i in range(data.shape[0]):
	centre_lat = centre_lats[i]
	centre_lon = centre_lons[i]
	generate_topography(centre_lat,centre_lon,i)


np.save('/user/work/al18709/tc_data_flipped_t/miroc_ssp585_topography.npy',tc_topography)
