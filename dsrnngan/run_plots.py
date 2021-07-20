import matplotlib
matplotlib.use("Agg")
import plots
import netCDF4
import numpy as np
import pandas as pd

log_path ='/ppdata/lucy-cGAN/logs/EASY/GAN/noise_2/g128_d512'
application = 'IFS'

rank_metrics_files_1 = ["{}/ranks-noise-124800.npz".format(log_path), "{}/ranks-noise-198400.npz".format(log_path)]
rank_metrics_files_2 = ["{}/ranks-noise-240000.npz".format(log_path), "{}/ranks-noise-384000.npz".format(log_path)]
labels_1 = ['124800', '198400']
labels_2 = ['240000', '384000']
name_1 = 'noise-early'
name_2 = 'noise-late'

plots.plot_rank_histogram_all(rank_metrics_files_1, labels_1, log_path, name_1)
plots.plot_rank_histogram_all(rank_metrics_files_2, labels_2, log_path, name_2)
