import roc
import os
import yaml

# input parameters
log_folder = '/ppdata/lucy-cGAN/logs/IFS/GAN/weights_natural'
model_numbers = [128000, 192000, 256000, 320000]
plot_ecpoint = True
predict_year = 2019
predict_full_image = True
ensemble_members = 100

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
problem_type = setup_params["GENERAL"]["problem_type"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]

if predict_full_image:
    batch_size = 2
    num_images = 10
else:
    batch_size = 16
    num_images = 50

if problem_type == "normal":
    downsample = False
    plot_input_title = 'IFS'
    input_channels = 9
    noise_channels = 4
elif problem_type == "superresolution":
    downsample = True
    plot_input_title = "Superresolution"
    input_channels = 1
    noise_channels = 2
else:
    raise Exception("no such problem type, try again!")


roc.plot_roc_curves(mode=mode,
                    arch=arch,
                    log_folder=log_folder,
                    model_numbers=model_numbers,
                    weights_dir=model_weights_root,
                    problem_type=problem_type,
                    filters_gen=filters_gen,
                    filters_disc=filters_disc,
                    noise_channels=noise_channels,
                    latent_variables=latent_variables,
                    padding=padding,
                    predict_year=predict_year,
                    predict_full_image=predict_full_image,
                    ensemble_members=ensemble_members,
                    plot_ecpoint=plot_ecpoint)
