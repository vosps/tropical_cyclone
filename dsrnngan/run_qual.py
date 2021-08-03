import matplotlib
matplotlib.use("Agg")
import numpy as np
import eval

mode = "ensemble"
#mode = "deterministic"
#train_years = [2016, 2017, 2018]
train_years = 2018
#val_years = [2016, 2017, 2018]
val_years = 2019
application = "IFS"
batch_size = 16
num_batches = 64
filters_gen = 128
filters_disc = 512
lr_disc = 1e-5
lr_gen = 1e-5
#downsample = True
downsample = False
constant_fields = 2
noise_channels = 4
#weights = None

weights = np.arange(6,2,-1)
weights = weights / weights.sum()

if downsample == True:
    input_channels = 1 
elif  downsample == False:
    input_channels = 9

log_path = "/ppdata/lucy-cGAN/logs/IFS/gen_128_disc_512/noise_4/weights_6_2"
out_fn = "{}/qual-{}.txt".format(log_path, application)

eval.quality_metrics_by_time(mode, 
                             train_years, 
                             val_years, 
                             application, 
                             out_fn, 
                             weights_dir=log_path, 
                             check_every=1, 
                             downsample=downsample,
                             weights=weights,
                             batch_size=batch_size, 
                             num_batches=num_batches, 
                             filters_gen=filters_gen, 
                             filters_disc=filters_disc, 
                             input_channels=input_channels,
                             constant_fields=constant_fields,
                             noise_channels=noise_channels, 
                             lr_disc=lr_disc, 
                             lr_gen=lr_gen)

#log_path = "/ppdata/lucy-cGAN/jupyter"
#weights_fn="gen_weights-ERA-0012800.h5"

#eval.quality_metrics_table(mode, weights_fn, train_years, val_years, application, batch_size=batch_size, num_batches=num_batches, filters=filters, lr_disc=lr_disc, lr_gen=lr_gen, lr=lr)  
