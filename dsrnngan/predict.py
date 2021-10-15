import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors, gridspec
from noise import NoiseGenerator
from setupmodel import setup_model
import ecpoint
import data
import models
import benchmarks
import argparse
from tfrecords_generator_ifs import create_fixed_dataset
from data_generator_ifs import DataGenerator as DataGeneratorFull
from data import get_dates
from plots import plot_img

parser = argparse.ArgumentParser()
parser.add_argument('--log_folder', type=str, 
                    help="directory where model weights are saved", 
                    default='/ppdata/lucy-cGAN/logs/IFS/gen_128_disc_512/noise_4/lr1e-5')
parser.add_argument('--model_number', type=str, 
                    help="model iteration to load", default='0313600')
parser.add_argument('--predict_year', type=int,
                    help="year to predict on", default=2019)
parser.add_argument('--num_predictions', type=int,
                    help="number of images to predict on", default=5)
parser.add_argument('--num_samples', type=int,
                    help="size of prediction ensemble", default=3)
parser.set_defaults(predict_full_image=False)
parser.set_defaults(include_Lanczos=False)
parser.set_defaults(include_RainFARM=False)
parser.set_defaults(include_deterministic=False)
parser.set_defaults(include_ecPoint=False)
parser.add_argument('--predict_full_image', dest='predict_full_image', action='store_true',
                    help="Predict on full images")
parser.add_argument('--include_Lanczos', dest='include_Lanczos', action='store_true',
                    help="Include Lanczos benchmark")
parser.add_argument('--include_RainFARM', dest='include_RainFARM', action='store_true',
                    help="Include RainFARM benchmark")
parser.add_argument('--include_deterministic', dest='include_deterministic', action='store_true',
                    help="Include deterministic model for comparison")
parser.add_argument('--include_ecPoint', dest='include_ecPoint', action='store_true',
                    help="Include ecPoint benchmark")
parser.add_argument('--plot_ranks_full', dest='plot_ranks_full', action='store_true',
                    help="Plot rank histograms for full images")
parser.add_argument('--plot_roc_small', dest='plot_roc_small', action='store_true',
                    help="Plot ROC and AUC curves for small images")
parser.add_argument('--plot_roc_full', dest='plot_roc_full', action='store_true',
                    help="Plot ROC and AUC curves for full images")   
args = parser.parse_args()

log_folder = args.log_folder
model_number = args.model_number
predict_year = args.predict_year
num_predictions = args.num_predictions
num_samples = args.num_samples

config_path = os.path.join(log_folder, 'setup_params.yaml')
with open(config_path, 'r') as f:
    try:
        setup_params = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
        
mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
problem_type = setup_params["GENERAL"]["problem_type"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]
batch_size = setup_params["TRAIN"]["batch_size"]
val_years = setup_params["VAL"]["val_years"]
val_size = setup_params["VAL"]["val_size"]

weights_fn = os.path.join(log_folder, 'models', 'gen_weights-{}.h5'.format(model_number))
dates = get_dates(predict_year)

if problem_type == "normal":
    downsample = False
    plot_input_title = 'IFS'
    input_channels = 9
elif problem_type == "superresolution":
    downsample = True
    plot_input_title = 'Downsampled'
    input_channels = 1 # superresolution problem doesn't have all 9 IFS fields
    if args.include_RainFARM or args.nclude_ecPoint or args.include_Lanczos:
        raise Exception("Cannot include ecPoint/Lanczos/RainFARM results for downsampled problem")

## initialise model
model = setup_model(mode=mode,
                    arch=arch,
                    input_channels=input_channels,
                    filters_gen=filters_gen, 
                    filters_disc=filters_disc,
                    noise_channels=noise_channels, 
                    latent_variables=latent_variables)
gen = model.gen
gen.load_weights(weights_fn)

## load appropriate dataset
if args.predict_full_image:
    plot_label = 'large'
    all_ifs_fields = ['tp','cp' ,'sp' ,'tisr','cape','tclw','tcwv','u700','v700']
    data_predict = DataGeneratorFull(dates=dates, 
                                     ifs_fields=all_ifs_fields,
                                     batch_size=batch_size, 
                                     log_precip=True, 
                                     crop=True,
                                     shuffle=False,
                                     constants=True,
                                     hour=2,
                                     ifs_norm=True,
                                     downsample=downsample)

else:
    include_ecPoint = False
    plot_label = 'small'
    data_predict = create_fixed_dataset(predict_year, 
                                        batch_size=batch_size, 
                                        downsample=downsample)
## dataset for benchmarks
data_ecpoint = DataGeneratorFull(dates=dates,
                                 ifs_fields=ecpoint.ifs_fields,
                                 batch_size=batch_size,
                                 log_precip=False,
                                 crop=True,
                                 shuffle=False,
                                 hour=2,
                                 ifs_norm=False,
                                 downsample=downsample)    
if args.include_deterministic:
    if problem_type == 'superresolution':
        filters_det = 256
        gen_det_weights = '/ppdata/lucy-cGAN/logs/EASY/deterministic/filters_256/gen_det_weights-IFS-0400000.h5'
    elif problem_type == 'normal':
        filters_det = 128
        gen_det_weights = '/ppdata/lucy-cGAN/logs/IFS/filters_128/softplus/det/lr_1e-4/gen_det_weights-ERA-0400000.h5'
    gen_det = models.generator(mode='det', input_channels=input_channels, filters_gen=filters_det)
    gen_det.load_weights(gen_det_weights)

pred = []
seq_real=[]
seq_cond=[]
seq_const=[]
seq_lanczos = []
seq_rainfarm = []
seq_det = []
seq_ecpoint = []
dummy = np.zeros((1, 940, 940))
data_predict_iter = iter(data_predict)    
for i in range(num_predictions):
    (inputs,outputs) = next(data_predict_iter)
    noise_shape = inputs['generator_input'][0,...,0].shape + (noise_channels,)
    inputs['noise_input'] = NoiseGenerator(noise_shape, batch_size=batch_size)
    ## store denormalised inputs, outputs, predictions
    seq_const.append(data.denormalise(inputs['constants']))
    seq_cond.append(data.denormalise(inputs['generator_input']))    
    ## make sure ground truth image has correct dimensions
    if args.predict_full_image :
        sample = np.expand_dims(np.array(outputs['generator_output']), axis=-1)
        seq_real.append(data.denormalise(sample))
    else:
        seq_real.append(data.denormalise(outputs['generator_output']))
    if args.include_deterministic:
        seq_det.append(data.denormalise(gen_det.predict(inputs)))
    else:
        seq_det.append(dummy)
    if mode == 'det':
        num_samples = 1 #can't generate an ensemble with deterministic method
        pred.append(data.denormalise(gen.predict(inputs)))
    else:
        pred_ensemble = []
        if mode == 'GAN':
            noise_shape = inputs['generator_input'][0,...,0].shape + (noise_channels,)
        elif mode == 'VAEGAN':
            noise_shape = inputs['generator_input'][0,...,0].shape + (latent_variables,)
        noise = NoiseGenerator(noise_shape, batch_size=batch_size)
        if mode == 'VAEGAN':
            # call encoder once
            mean, logvar = gen.encoder([inputs['generator_input'], inputs['constants']])       
        for j in range(num_samples):
            inputs['noise_input'] = noise()
            if mode == 'GAN':
                gan_inputs = [inputs['generator_input'], inputs['constants'], inputs['noise_input']]
                pred_ensemble.append(data.denormalise(gen.predict(gan_inputs)))
            elif mode == 'VAEGAN':
                dec_inputs = [mean, logvar, inputs['noise_input'], inputs['constants']]
                pred_ensemble.append(data.denormalise(gen.decoder.predict(dec_inputs)))
        pred_ensemble = np.array(pred_ensemble)
        pred.append(pred_ensemble)

data_ecpoint_iter = iter(data_ecpoint)
for i in range(num_predictions):
    (inp,outp) = next(data_ecpoint_iter)        
    ## ecPoint prediction
    if args.include_ecPoint:
        seq_ecpoint.append(np.mean(benchmarks.ecpointPDFmodel(inp['generator_input']),axis=-1))
    else:
        seq_ecpoint.append(dummy)
    if args.include_RainFARM:
        seq_rainfarm.append(benchmarks.rainfarmmodel(inp['generator_input'][...,1]))
    else:
        seq_rainfarm.append(dummy)
    if args.include_Lanczos:
        seq_lanczos.append(benchmarks.lanczosmodel(inp['generator_input'][...,1]))
    else:
        seq_lanczos.append(dummy)

## plot input conditions and prediction example
## batch 0
IFS = seq_cond[0][0,...,0]
constant_0 = seq_const[0][0,...,0]
constant_1 = seq_const[0][0,...,1]
NIMROD = seq_real[0][0,...,0]
pred_0_0 = pred[0][0][0,...,0]
(vmin, vmax) = (0,2)
fig, ax = plt.subplots(1,5, figsize=(15,10))
ax[2].imshow(IFS, vmin=vmin, vmax=vmax)
ax[2].set_title(plot_input_title)
ax[1].imshow(constant_0, vmin=vmin, vmax=vmax)
ax[1].set_title('Orography')
ax[0].imshow(constant_1, vmin=vmin, vmax=vmax)
ax[0].set_title('Land-sea mask')
ax[3].imshow(NIMROD, vmin=vmin, vmax=vmax)
ax[3].set_title('NIMROD')
ax[4].imshow(pred_0_0, vmin=vmin, vmax=vmax)
ax[4].set_title('Prediction')
for ax in ax.flat:
    ax.tick_params(left=False, bottom=False,labelleft=False, labelbottom=False)
    ax.invert_yaxis()

plt.savefig("{}/prediction-and-input-{}-{}.pdf".format(log_folder, 
                                                       problem_type,
                                                       plot_label), bbox_inches='tight')
plt.close()


## generate labels for plots
labels = [plot_input_title, "NIMROD"]
for i in range(num_samples):
    labels.append(f"{mode} pred {i+1}")
if args.include_RainFARM:
    labels.append("RainFARM")
if args.include_ecPoint:
    labels.append("ecPoint mean")
if args.include_deterministic:
    labels.append("Deterministic")
if args.include_Lanczos:
    labels.append("Lanczos")

    
## plot a range of prediction examples for different downscaling methods    
sequences = []
for i in range(num_predictions):
    tmp = {}
    tmp['NIMROD'] = seq_real[i][0,...,0]
    tmp[plot_input_title] = seq_cond[i][0,...,0]
    tmp['Lanczos'] = seq_lanczos[i][0,...]
    tmp['RainFARM'] = seq_rainfarm[i][0,...]
    tmp['Deterministic'] = seq_det[i][0,...,0]
    tmp['ecPoint mean'] = seq_ecpoint[i][0,...]
    for j in range(num_samples):
        tmp[f"{mode} pred {j+1}"] = pred[i][j][0,...,0]
    sequences.append(tmp)
    
num_cols = num_predictions
num_rows = len(labels)+1
plt.figure(figsize=(1.5*num_cols,1.5*num_rows))
value_range = (0.1,5)
gs = gridspec.GridSpec(num_rows*num_rows,num_rows*num_cols,wspace=0.5,hspace=0.5)

for k in range(num_predictions):
    for i in range(len(labels)):
        plt.subplot(gs[(num_rows*i):(num_rows+num_rows*i),(num_rows*k):(num_rows+num_rows*k)])
        plot_img(sequences[k][labels[i]], value_range=value_range)
        if k==0:
            plt.ylabel(labels[i])
plt.suptitle('Example predictions for different input conditions')
##colorbar
units = "Rain rate [mm h$^{-1}$]"
cb_tick_loc = np.array([0.1, 0.5, 1, 2, 5, 10])
cb_tick_labels = [0.1, 0.5, 1, 2, 5, 10]
cax = plt.subplot(gs[-1,1:-1]).axes
cb = colorbar.ColorbarBase(cax, norm=colors.Normalize(*value_range), orientation='horizontal')
cb.set_ticks(cb_tick_loc)
cb.set_ticklabels(cb_tick_labels)
cax.tick_params(labelsize=16)
cb.set_label(units, size=16)
            
plt.savefig("{}/predictions-{}-{}.pdf".format(log_folder, 
                                              problem_type,
                                              plot_label), bbox_inches='tight')
plt.close()
