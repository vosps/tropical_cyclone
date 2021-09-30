import matplotlib
matplotlib.use("Agg")
import numpy as np
import eval

mode = "ensemble"
val_years = 2019
application = "IFS"
batch_size = 16
num_batches = 64
filters_gen = 256
filters_disc = 512
lr_disc = 1e-5
lr_gen = 1e-5
downsample = False
constant_fields = 2
noise_channels = 4
add_noise = True
load_full_image = True
#weights = [0.87, 0.06, 0.03, 0.03]
weights = [0.4, 0.3, 0.2, 0.1]
#weights = np.arange(36,2,-11)
#weights = weights / weights.sum()
#weights  = None

if downsample == True:
    input_channels = 1 
elif  downsample == False:
    input_channels = 9

log_path = "/ppdata/lucy-cGAN/logs/IFS/gen_256_disc_512/noise_4/weights_4x"

if add_noise == False and load_full_image==False:
    out_fn = "{}/qual-{}_no_noise__{}.txt".format(log_path, application, str(val_years))
elif add_noise ==True and load_full_image==False:
    out_fn = "{}/qual-{}_noise__{}.txt".format(log_path, application, str(val_years))
if add_noise == False and load_full_image==True:
    out_fn = "{}/qual-{}_no_noise_full_image__{}.txt".format(log_path, application, str(val_years))
elif add_noise ==True and load_full_image==True:
    out_fn = "{}/qual-{}_noise_full_image__{}.txt".format(log_path, application, str(val_years))


eval.quality_metrics_by_time(mode=mode, 
                             val_years=val_years, 
                             application=application, 
                             out_fn=out_fn, 
                             weights_dir=log_path, 
                             check_every=1, 
                             downsample=downsample,
                             load_full_image=load_full_image,
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

