
import subprocess
import os
import datetime


# wget -r 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2001/ /bp1/geog-tropical/data/Obs/IMERG-V07/'


# file = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2001/001/3B-HHR.MS.MRG.3IMERG.20010101-S000000-E002959.0000.V07A.HDF5'

def generate_days(year):
	if year in [2004,2008,2012,2016]:
		days = ["{0:03}".format(i) for i in range(1,367)]
	elif year == 2000:
		days = ["{0:03}".format(i) for i in range(153,366)]
	elif year == 2020:
		days = ["{0:03}".format(i) for i in range(1,305)]
	elif year == 2005:
		days = ["{0:03}".format(i) for i in range(1,191)]
	elif year == 2015:
		days = ["{0:03}".format(i) for i in range(262,366)]
	else:
		days = ["{0:03}".format(i) for i in range(1,366)]
	return days

def which_month(year,doy):
	date = datetime.date(year, 1, 1) #Will give 1996-01-01
	delta = datetime.timedelta(int(doy) - 1) #str(delta) will be '31 days, 0:00:00'
	newdate = date + delta
	month = "{0:02}".format(newdate.month)
	day = "{0:02}".format(newdate.day)
	return month,day
	# 2000 = 153 - 366
	# 2004 leap
	# 2008 leap
	# 2012 leap
	# 2016 leap
	# 2020 001 - 305

years = range(2015,2016)
times3a = ["{0:04}".format(i) for i in range(0,1440,30)][::2]
times3b = ["{0:04}".format(i) for i in range(30,1440,30)][::2]
times2a = ["{0:06}".format(i) for i in  range(2959,245959,10000)]
times2b = ["{0:06}".format(i) for i in  range(5959,245959,10000)]
times1b = ["{0:06}".format(i) for i in  range(3000,243000,10000)]
times1a = ["{0:06}".format(i) for i in  range(0,233000,10000)]

print(times1a)
print(times1b)
print(times2a)
print(times2b)
# print(times3)

# years = range(2000,2001)
for year in years:
	days = generate_days(year)
	# days = ['153']
	for day in days:
		# print(days)
		# file = f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/{year}/{day}/3B-HHR.MS.MRG.3IMERG.20010101-S000000-E002959.0000.V07A.HDF5'
		dir = f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/{year}/{day}/'
		dir = f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.06/{year}/{day}/'
		# https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.06/2000/153/
		save_dir = f'/bp1/geog-tropical/data/Obs/IMERG-V07/{year}/{day}/'
		save_dir = f'/bp1/geog-tropical/data/Obs/IMERG-V06/{year}/{day}/'
		os.makedirs(save_dir)
		# wget_cmd = ['wget', '-r', '-np', '-A', "*.HDF5", dir,save_dir]
		m,d = which_month(year,day)
		# for i in 
		print(len(times1a))
		print(len(times2a))
		timesA = [f'{year}{m}{d}-S{times1a[i]}-E{times2a[i]}.{times3a[i]}' for i in range(len(times1a))]
		timesB = [f'{year}{m}{d}-S{times1b[i]}-E{times2b[i]}.{times3b[i]}' for i in range (len(times1b))]
		print(timesA)
		print(timesB)
		times = timesA+timesB
		print('times',times)
		for time in times:
			file = f'3B-HHR.MS.MRG.3IMERG.{time}.V07A.HDF5'
			file = f'3B-HHR.MS.MRG.3IMERG.{time}.V06B.HDF5'
			# 3B-HHR.MS.MRG.3IMERG.20000601-S000000-E002959.0000.V06B.HDF5
			wget_cmd = ['wget','-np','--no-glob','-P',save_dir,dir+file]
			subprocess.run(wget_cmd)
			print(dir+file,'saved!')