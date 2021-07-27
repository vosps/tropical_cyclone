import matplotlib
matplotlib.use("Agg")
import plots
import netCDF4
import numpy as np
import pandas as pd

log_path ='/ppdata/lucy-cGAN/logs/IFS/gen_128_disc_512/noise_4/lr1e-5'
application = 'IFS'

rank_metrics_files_1 = ["{}/ranks-198400.npz".format(log_path), "{}/ranks-noise-198400.npz".format(log_path)]
rank_metrics_files_2 = ["{}/ranks-240000.npz".format(log_path), "{}/ranks-noise-240000.npz".format(log_path)]
labels_1 = ['no-noise-198400', 'noise-198400']
labels_2 = ['no-noise-240000', 'noise-240000']
name_1 = 'comparison-early'
name_2 = 'comparison-late'

plots.plot_rank_histogram_all(rank_metrics_files_1, labels_1, log_path, name_1)
plots.plot_rank_histogram_all(rank_metrics_files_2, labels_2, log_path, name_2)
