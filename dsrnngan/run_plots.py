import plots
import netCDF4
import numpy as np
import pandas as pd

log_path = "/ppdata/lucy-cGAN/jupyter"
application = 'ERA'
eval_fn = "{}/eval-{}.txt".format(log_path, application)
qual_fn = "{}/qual-{}.txt".format(log_path, application)

metrics_fn = eval_fn
print(metrics_fn)

plots.plot_rank_metrics_by_samples(metrics_fn)
