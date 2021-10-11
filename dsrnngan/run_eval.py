
import matplotlib
matplotlib.use("Agg")
import numpy as np
import evaluation
import plots

mode = "ensemble"
val_years = 2019
application = "IFS"
batch_size = 16
num_batches = 64
filters_gen = 64
filters_disc = 512
lr_disc = 1e-5
lr_gen = 1e-5
downsample = False
constant_fields = 2
#model_number = '0124800'
model_number = None
noise_channels = 4
add_noise = True
load_full_image = True
#weights = np.arange(36,2,-11)
#weights = weights / weights.sum()
#weights = [0.87, 0.06, 0.03, 0.03]
weights = [0.4, 0.3, 0.2, 0.1]

if downsample == True:
    input_channels = 1 
elif  downsample == False:
    input_channels = 9

if mode == "ensemble":
    log_path = "/ppdata/lucy-cGAN/logs/IFS/gen_64_disc_512/noise_4/weights_4x"
    rank_samples = 100
elif mode == "deterministic":
    log_path = "/ppdata/lucy-cGAN/logs/IFS/filters_128/softplus/det/lr_1e-4/"
    rank_samples = 1

if add_noise == False and load_full_image==False:
    out_fn = "{}/eval-{}_no_noise__{}.txt".format(log_path, application, str(val_years))
elif add_noise ==True and load_full_image==False:
    out_fn = "{}/eval-{}_noise__{}.txt".format(log_path, application, str(val_years))
if add_noise == False and load_full_image==True:
    out_fn = "{}/eval-{}_no_noise_full_image__{}.txt".format(log_path, application, str(val_years))
elif add_noise ==True and load_full_image==True:
    out_fn = "{}/eval-{}_noise_full_image__{}.txt".format(log_path, application, str(val_years))

eval.rank_metrics_by_time(mode, 
                          val_years, 
                          application, 
                          out_fn, 
                          weights_dir=log_path, 
                          check_every=1, 
                          N_range=None, 
                          downsample=downsample,
                          weights=weights,
                          add_noise=add_noise,
                          load_full_image=load_full_image,
                          model_number=model_number,
                          batch_size=batch_size, 
                          num_batches=num_batches, 
                          filters_gen=filters_gen, 
                          filters_disc=filters_disc, 
                          input_channels=input_channels,
                          constant_fields=constant_fields,
                          noise_channels=noise_channels,  
                          rank_samples=rank_samples, 
                          lr_disc=lr_disc, 
                          lr_gen=lr_gen)

## plot rank histograms
labels_1 = ['124800', '198400']
labels_2 = ['240000', '320000']
if add_noise == True and load_full_image == False:
    name_1 = 'noise-early'
    name_2 = 'noise-late'
    rank_metrics_files_1 = ["{}/ranks-noise-124800.npz".format(log_path), "{}/ranks-noise-198400.npz".format(log_path)]
    rank_metrics_files_2 = ["{}/ranks-noise-240000.npz".format(log_path), "{}/ranks-noise-320000.npz".format(log_path)]

elif add_noise == False and load_full_image == False:
    name_1 = 'no-noise-early'
    name_2 = 'no-noise-late'
    rank_metrics_files_1 = ["{}/ranks-124800.npz".format(log_path), "{}/ranks-198400.npz".format(log_path)]
    rank_metrics_files_2 = ["{}/ranks-240000.npz".format(log_path), "{}/ranks-320000.npz".format(log_path)]

elif add_noise == True and load_full_image == True:
    name_1 = 'full_image-noise-early'
    name_2 = 'full_image-noise-late'
    rank_metrics_files_1 = ["{}/ranks-full_image-noise-124800.npz".format(log_path), "{}/ranks-full_image-noise-198400.npz".format(log_path)]
    rank_metrics_files_2 = ["{}/ranks-full_image-noise-240000.npz".format(log_path), "{}/ranks-full_image-noise-320000.npz".format(log_path)]

elif add_noise == False and load_full_image == True:
    name_1 = 'full_image-no-noise-early'
    name_2 = 'full_image-no-noise-late'
    rank_metrics_files_1 = ["{}/ranks-full_image-124800.npz".format(log_path), "{}/ranks-full_image-198400.npz".format(log_path)]
    rank_metrics_files_2 = ["{}/ranks-full_image-240000.npz".format(log_path), "{}/ranks-full_image-320000.npz".format(log_path)]

plots.plot_rank_histogram_all(rank_metrics_files_1, labels_1, log_path, name_1)
plots.plot_rank_histogram_all(rank_metrics_files_2, labels_2, log_path, name_2)
