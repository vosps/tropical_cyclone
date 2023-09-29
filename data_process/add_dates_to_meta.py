
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib import colors
# import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
from utils.data import load_tc_data
# from utils.plot import make_cmap
import cftime as cf


def lookup_ibtracs(row,tracks):
	tracks = pd.read_csv('/user/work/al18709/ibtracks/ibtracs.ALL.list.v04r00.csv',
						usecols=['SID','LAT','LON','BASIN','NAME','SEASON', 'NATURE','ISO_TIME','USA_SSHS'],
						parse_dates = ['ISO_TIME'],keep_default_na=False)
	boolean = (tracks.SID == row.sid) & (tracks.LAT == row.centre_lat) & (tracks.LON == row.centre_lon)
	time =  pd.to_datetime(tracks.loc[boolean].ISO_TIME)
	if list(time) == []:
		print('no',flush=True)
		date = cf.datetime(calendar='gregorian',
						year=1978,
						month=1,
						day=1,
						hour=0
						)
		return date
	else:
		time =  time.iloc[0]
		date = cf.datetime(calendar='gregorian',
						year=time.year,
						month=time.month,
						day=time.day,
						hour=time.hour
						)
	return date

def find_dates_ibtracs(meta):
		tracks = pd.read_csv('/user/work/al18709/ibtracks/ibtracs.ALL.list.v04r00.csv',
							usecols=['SID','LAT','LON','BASIN','NAME','SEASON', 'NATURE','ISO_TIME','USA_SSHS'],
							parse_dates = ['ISO_TIME'],keep_default_na=False)
		dates = meta.apply(lookup_ibtracs, tracks= tracks, axis=1)
		# dates = meta.apply(lookup_ibtracs, axis=1)
		return dates
	

# load current 1D dataset
real,inputs,pred,meta = load_tc_data(set='validation',results='ke_tracks')
dates = find_dates_ibtracs(meta)
meta['date'] = dates
meta.to_csv('/user/work/al18709/tc_data_mswep_40/scalar_wgan_valid_meta_with_dates.csv')

# load original 2D WGAN
real_2,inputs_2,pred_2,meta_2,imput_og,pred_og,meta_og = load_tc_data(set='validation',results='kh_tracks')
meta_2['date'] = find_dates_ibtracs(meta_2)
# meta_og['date'] = find_dates_ibtracs(meta_og,'gregorian')
real_og_x,_,_,_,_,_,pred_og_x,meta_og_x = load_tc_data(set='extreme_test',results='test')
meta_og_x['date'] = find_dates_ibtracs(meta_og_x)
meta_og = pd.read_csv('/user/work/al18709/tc_data_mswep_40/valid_meta.csv')
meta_og['date'] = find_dates_ibtracs(meta_og)
meta_og.to_csv('/user/work/al18709/tc_data_mswep_40/original_wgan_valid_meta_with_dates.csv')
# meta_valid = pd.read_csv('/user/work/al18709/tc_data_flipped/valid_meta.csv')

# TODO everything in gregorian calendar... double check this