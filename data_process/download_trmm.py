import os,subprocess,datetime

def which_month(year,doy):
	date = datetime.date(year, 1, 1) #Will give 1996-01-01
	delta = datetime.timedelta(int(doy) - 1) #str(delta) will be '31 days, 0:00:00'
	newdate = date + delta
	month = "{0:02}".format(newdate.month)
	day = "{0:02}".format(newdate.day)
	return month,day

years = range(1999,2020)
doys = ["{0:03}".format(i) for i in range(1,366)]
hours = ["{0:02}".format(i) for i in range(0,24,3)]

for year in years:
	for doy in doys:
		os.makedirs(f'/bp1/geog-tropical/data/Obs/TRMM/{year}/{doy}/')
		for hour in hours:
			month,day = which_month(year,doy)
			if hour == '00':
				print(day)
				day = "{0:02}".format(int(day)+1)
				print(day)
			filename = f'3B42.{year}{month}{day}.{hour}.7.HDF'

			cmd = ['scp', f'jasmin-sci:/badc/trmm/data/TRMM_3B42/{year}/{doy}/{filename}', f'/bp1/geog-tropical/data/Obs/TRMM/{year}/{doy}/']
			scp jasmin-sci:/badc/trmm/data/TRMM_3B42/*.HDF /bp1/geog-tropical/data/Obs/TRMM/
			print('downloading ',cmd)
			# scp jasmin-sci:/badc/trmm/data/TRMM_3B42/1999/001/3B42.19990101.09.7.HDF /bp1/geog-tropical/data/Obs/TRMM/1999/001/]
			# /bp1/geog-tropical/data/Obs/TRMM/1999/001/3B42.19990101.09.7.HDF
			# /bp1/geog-tropical/data/Obs/TRMM/1999/001
			subprocess.run(cmd)