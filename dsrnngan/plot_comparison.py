import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colorbar, colors, gridspec
import numpy as np
import rainfarm
import data
import models
import train
import plots

batch_size = 16
test_years = 2019
noise_channels = 4
filters_gen = 128
gen_gan_weights = '/ppdata/lucy-cGAN/logs/IFS/gen_128_disc_512/noise_4/lr1e-5/gen_weights-IFS-0313600.h5'
gen_det_mse_weights = '/ppdata/lucy-cGAN/logs/IFS/filters_128/softplus/det/lr_1e-4/gen_det_weights-ERA-0313600.h5'
application = "IFS"
downsample = False
weights = None

plots.plot_comparison(test_years, 
                filters_gen, 
                noise_channels, 
                gen_gan_weights, 
                gen_det_mse_weights, 
                downsample=downsample,
                weights=weights,
                batch_size, 
                application)

plt.savefig("/ppdata/lucy-cGAN/figures/comparison.pdf", bbox_inches='tight')
plt.close()
