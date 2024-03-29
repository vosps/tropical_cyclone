import os
import yaml
import matplotlib
matplotlib.use("Agg")
import evaluation
import plots

# input parameters
log_folder = '/ppdata/andrew-output/VAEGAN-CL20fc-kl5lv10-neworog'
val_years = 2020
load_full_image = True

model_weights_root = os.path.join(log_folder, "models")
config_path = os.path.join(log_folder, 'setup_params.yaml')
with open(config_path, 'r') as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
padding = setup_params["MODEL"]["padding"]
batch_size = 1 # setup_params["TRAIN"]["batch_size"]
# steps_per_checkpoint = setup_params["TRAIN"]["steps_per_checkpoint"]
problem_type = setup_params["GENERAL"]["problem_type"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]
num_batches = 512  # setup_params["EVAL"]["num_batches"]
add_noise = setup_params["EVAL"]["add_postprocessing_noise"]
noise_factor = setup_params["EVAL"]["postprocessing_noise_factor"]
noise_factor = float(noise_factor)
max_pooling = setup_params["EVAL"]["max_pooling"]
avg_pooling = setup_params["EVAL"]["avg_pooling"]

model_numbers = [160000]

if problem_type == 'normal':
    input_channels = 9 
    downsample = False
elif  problem_type == 'superresolution':
    input_channels = 1
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
                                weights_dir=model_weights_root, 
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
                                padding=padding,
                                rank_samples=rank_samples,
                                max_pooling=max_pooling,
                                avg_pooling=avg_pooling)
