import os
import yaml
import matplotlib
matplotlib.use("Agg")
import evaluation

# input parameters
log_folder = '/ppdata/lucy-cGAN/logs/test/GAN'
val_years = 2019
load_full_image = False
model_numbers = [3200, 9600]

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
    input_channels = 9
    downsample = False
elif  problem_type == 'superresolution':
    input_channels = 1
    downsample = True

if load_full_image:
    batch_size = 1
    
if add_noise == False and load_full_image==False:
    out_fn = "{}/qual_no_noise__{}.txt".format(log_folder, str(val_years))
elif add_noise ==True and load_full_image==False:
    out_fn = "{}/qual_noise__{}.txt".format(log_folder, str(val_years))
if add_noise == False and load_full_image==True:
    out_fn = "{}/qual_no_noise_full_image__{}.txt".format(log_folder, str(val_years))
elif add_noise ==True and load_full_image==True:
    out_fn = "{}/qual_noise_full_image__{}.txt".format(log_folder, str(val_years))


evaluation.quality_metrics_by_time(mode=mode, 
                                   arch=arch,
                                   val_years=val_years, 
                                   log_fname=out_fn, 
                                   weights_dir=model_weights_root, 
                                   downsample=downsample,
                                   load_full_image=load_full_image,
                                   model_numbers=model_numbers,
                                   batch_size=batch_size, 
                                   num_batches=num_batches, 
                                   filters_gen=filters_gen, 
                                   filters_disc=filters_disc, 
                                   input_channels=input_channels,
                                   latent_variables=latent_variables,
                                   noise_channels=noise_channels)
