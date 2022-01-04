"""
This script plots a gif from the X and y data to check everything alligns well
"""


# TODO: adapt as X and y are grouped by sid number
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import seaborn as sns
# import cartopy.crs as ccrs
import numpy as np
sns.set_style("white")

# get data
y = np.load('/user/work/al18709/tc_data/y.npy',allow_pickle = True)
X = np.load('/user/work/al18709/tc_data/X.npy',allow_pickle = True)

print(X.shape)
print(y.shape)


time = range(0,40)

X_lat2d,X_lon2d = np.linspace(-50, 50, 10),np.linspace(-50, 50, 10)
y_lat2d, y_lon2d = np.linspace(-50, 50, 100),np.linspace(0, 100, 100)

# plot stuff
fig=plt.figure(1, figsize=(20, 10))
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='tc_gif', artist='Matplotlib',
		comment='Movie support!')
writer = FFMpegWriter(fps=2, metadata=metadata)
f_movie = 'tc.mp4'

print('making gif...')
with writer.saving(fig, f_movie,dpi=300):
	for t in time:
		ax = plt.subplot2grid((1,2), (0,0))
		cmap ='seismic_r'
		c = ax.pcolor(X_lon2d,X_lat2d,X[t],vmin=-30,vmax=30,cmap = cmap)
		# plt.contourf(data, 60,vmin=-40, vmax=40)
		ax.set_title('X')
		# plt.subplots_adjust(wspace=0.0, right=0.7)

		ax = plt.subplot2grid((1,2), (0,1))
		c = ax.pcolor(y_lon2d,y_lat2d,y[t],vmin=-30,vmax=30,cmap = cmap)
		# plt.contourf(data, 60,vmin=-40, vmax=40)
		ax.set_title('y')
		# plt.subplots_adjust(wspace=0.0, right=0.7)

		# save frame
		plt.savefig('figs/tc_mswep.png',bbox_inches='tight')
		# exit()

		writer.grab_frame()
		plt.clf()
		

print('gif saved!')