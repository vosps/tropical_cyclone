import matplotlib
matplotlib.use("Agg")
import plots
import netCDF4
import numpy as np
import pandas as pd

log_path = "/ppdata/lucy-cGAN/logs/IFS/filters_128/softplus/gan/noise_8/g_1e-5_d_2e-5"
application = 'IFS'

rank_metrics_files_1 = ["{}/ranks-{}-9600.npz".format(log_path, application), "{}/ranks-{}-124800.npz".format(log_path, application)]
rank_metrics_files_2 = ["{}/ranks-{}-240000.npz".format(log_path, application), "{}/ranks-{}-313600.npz".format(log_path, application)]
labels_1 = ['9600', '124800']
labels_2 = ['240000', '313600']
name_1 = 'early'
name_2 = 'late'

plots.plot_rank_histogram_all(rank_metrics_files_1, labels_1, log_path, name_1)
plots.plot_rank_histogram_all(rank_metrics_files_2, labels_2, log_path, name_2)
