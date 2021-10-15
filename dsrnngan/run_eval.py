import os
import yaml
import matplotlib
matplotlib.use("Agg")
import evaluation
import plots

# input parameters
log_folder = '/ppdata/lucy-cGAN/logs/test/GAN'
val_years = 2019
load_full_image = False
model_numbers = [124800]

model_weights_root = os.path.join(log_folder, "models")
config_path = os.path.join(log_folder, 'setup_params.yaml')
with open(config_path, 'r') as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
batch_size = setup_params["TRAIN"]["batch_size"]
problem_type = setup_params["GENERAL"]["problem_type"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]
num_batches = setup_params["EVAL"]["num_batches"]
add_noise = setup_params["EVAL"]["add_postprocessing_noise"]
noise_factor = setup_params["EVAL"]["postprocessing_noise_factor"]

if problem_type == 'normal':
    input_channels = 1 
    downsample = False
elif  problem_type == 'superresolution':
    input_channels = 9
    downsample = True

if mode in ["GAN", "VAEGAN"]:
    rank_samples = 100
elif mode == "det":
    rank_samples = 1

if load_full_image:
    batch_size = 1

if add_noise == False and load_full_image==False:
    out_fn = "{}/eval_no_noise__{}.txt".format(log_folder, str(val_years))
elif add_noise ==True and load_full_image==False:
    out_fn = "{}/eval_noise__{}.txt".format(log_folder, str(val_years))
if add_noise == False and load_full_image==True:
    out_fn = "{}/eval_no_noise_full_image__{}.txt".format(log_folder, str(val_years))
elif add_noise ==True and load_full_image==True:
    out_fn = "{}/eval_noise_full_image__{}.txt".format(log_folder,str(val_years))

evaluation.rank_metrics_by_time(mode=mode,
                                arch=arch,
                                val_years=val_years, 
                                log_fname=out_fn, 
                                weights_dir=log_folder, 
                                downsample=downsample,
                                add_noise=add_noise,
                                noise_factor=noise_factor,
                                load_full_image=load_full_image,
                                model_numbers=model_numbers,
                                batch_size=batch_size, 
                                num_batches=num_batches, 
                                filters_gen=filters_gen, 
                                filters_disc=filters_disc, 
                                input_channels=input_channels,
                                latent_variables=latent_variables,
                                noise_channels=noise_channels,  
                                rank_samples=rank_samples)

## plot rank histograms
labels_1 = ['124800', '198400']
labels_2 = ['240000', '320000']
if add_noise == True and load_full_image == False:
    name_1 = 'noise-early'
    name_2 = 'noise-late'
    rank_metrics_files_1 = ["{}/ranks-noise-124800.npz".format(log_folder), "{}/ranks-noise-198400.npz".format(log_folder)]
    rank_metrics_files_2 = ["{}/ranks-noise-240000.npz".format(log_folder), "{}/ranks-noise-320000.npz".format(log_folder)]

elif add_noise == False and load_full_image == False:
    name_1 = 'no-noise-early'
    name_2 = 'no-noise-late'
    rank_metrics_files_1 = ["{}/ranks-124800.npz".format(log_folder), "{}/ranks-198400.npz".format(log_folder)]
    rank_metrics_files_2 = ["{}/ranks-240000.npz".format(log_folder), "{}/ranks-320000.npz".format(log_folder)]

elif add_noise == True and load_full_image == True:
    name_1 = 'full_image-noise-early'
    name_2 = 'full_image-noise-late'
    rank_metrics_files_1 = ["{}/ranks-full_image-noise-124800.npz".format(log_folder), "{}/ranks-full_image-noise-198400.npz".format(log_folder)]
    rank_metrics_files_2 = ["{}/ranks-full_image-noise-240000.npz".format(log_folder), "{}/ranks-full_image-noise-320000.npz".format(log_folder)]

elif add_noise == False and load_full_image == True:
    name_1 = 'full_image-no-noise-early'
    name_2 = 'full_image-no-noise-late'
    rank_metrics_files_1 = ["{}/ranks-full_image-124800.npz".format(log_folder), "{}/ranks-full_image-198400.npz".format(log_folder)]
    rank_metrics_files_2 = ["{}/ranks-full_image-240000.npz".format(log_folder), "{}/ranks-full_image-320000.npz".format(log_folder)]

plots.plot_rank_histogram_all(rank_metrics_files_1, labels_1, log_folder, name_1)
plots.plot_rank_histogram_all(rank_metrics_files_2, labels_2, log_folder, name_2)
