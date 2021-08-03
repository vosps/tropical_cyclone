import matplotlib
matplotlib.use("Agg")
import plots
import netCDF4
import numpy as np
import pandas as pd

log_path ='/ppdata/lucy-cGAN/logs/IFS/gen_128_disc_512/noise_4/lr1e-5'
application = 'IFS'
added_noise = True

if added_noise == True:
    rank_metrics_files_1 = ["{}/ranks-noise-124800.npz".format(log_path), "{}/ranks-noise-198400.npz".format(log_path)]
    rank_metrics_files_2 = ["{}/ranks-noise-240000.npz".format(log_path), "{}/ranks-noise-320000.npz".format(log_path)]
    labels_1 = ['noise-124800', 'noise-198400']
    labels_2 = ['noise-240000', 'noise-320000']
    name_1 = 'noise-early'
    name_2 = 'noise-late'
elif added_noise == False:
    rank_metrics_files_1 = ["{}/ranks-no-noise-124800.npz".format(log_path), "{}/ranks-no-noise-198400.npz".format(log_path)]
    rank_metrics_files_2 = ["{}/ranks-no-noise-240000.npz".format(log_path), "{}/ranks-no-noise-384000.npz".format(log_path)]                                     
    labels_1 = ['no-noise-124800', 'no-noise-198400']
    labels_2 = ['no-noise-240000', 'no-noise-384000']
    name_1 = 'no-noise-early'
    name_2 = 'no-noise-late'
plots.plot_rank_histogram_all(rank_metrics_files_1, labels_1, log_path, name_1)
plots.plot_rank_histogram_all(rank_metrics_files_2, labels_2, log_path, name_2)
