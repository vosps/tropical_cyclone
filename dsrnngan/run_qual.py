import matplotlib
matplotlib.use("Agg")
import evaluation

mode = "GAN"
val_years = 2019
batch_size = 16
num_batches = 64
filters_gen = 256
filters_disc = 512
lr_disc = 1e-5
lr_gen = 1e-5
downsample = False
latent_variables = 1
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
    out_fn = "{}/qual_no_noise__{}.txt".format(log_path, str(val_years))
elif add_noise ==True and load_full_image==False:
    out_fn = "{}/qual_noise__{}.txt".format(log_path, str(val_years))
if add_noise == False and load_full_image==True:
    out_fn = "{}/qual_no_noise_full_image__{}.txt".format(log_path, str(val_years))
elif add_noise ==True and load_full_image==True:
    out_fn = "{}/qual_noise_full_image__{}.txt".format(log_path, str(val_years))


evaluation.quality_metrics_by_time(mode=mode, 
                                   val_years=val_years, 
                                   out_fn=out_fn, 
                                   weights_dir=log_path, 
                                   downsample=downsample,
                                   weights=weights,
                                   load_full_image=load_full_image,
                                   batch_size=batch_size, 
                                   num_batches=num_batches, 
                                   filters_gen=filters_gen, 
                                   filters_disc=filters_disc, 
                                   input_channels=input_channels,
                                   latent_variables=latent_variables,
                                   noise_channels=noise_channels)