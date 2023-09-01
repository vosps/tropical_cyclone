import os
import yaml
import gc
import copy
import numpy as np
import matplotlib as mpl
#mpl.use('svg')
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors, gridspec
from matplotlib.backends.backend_pdf import PdfPages
# import cartopy
# import cartopy.crs as ccrs
from noise import NoiseGenerator
from setupmodel import setup_model
# import ecpoint
import data
import models
import benchmarks
import argparse
from tfrecords_generator_ifs import create_fixed_dataset
from data_generator_ifs import DataGenerator as DataGeneratorFull
# from data import get_dates, all_ifs_fields
from data import all_ifs_fields
from plots import plot_img_log_coastlines, truncate_colourmap
from data import ifs_norm
from rapsd import plot_spectrum1d, rapsd

# plotting parameters
value_range_precip = (0.1,30)
value_range_lsm = (0,1.2)
value_range_orog = (-0.05,1)
cmap = copy.copy(mpl.cm.get_cmap("viridis"))
cmap.set_under('white')
linewidth = 0.4
extent = [-7.5, 2, 49.5, 59] # (lon, lat)
alpha = 0.8
dpi = 200

# colorbar
units = "Rain rate [mm h$^{-1}$]"
cb_tick_loc = np.array([0.1, 0.5, 1, 2, 5, 10, 30, 50])
cb_tick_labels = [0.1, 0.5, 1, 2, 5, 10, 30, 50]

# colormap for LSM -- removes the white end
cmap_lsm = plt.get_cmap('terrain')
cmap_lsm = truncate_colourmap(cmap_lsm, 0, 0.8)

parser = argparse.ArgumentParser()
parser.add_argument('--log_folder', type=str, 
                    help="directory where model weights are saved")
parser.add_argument('--model_number', type=str, 
                    help="model iteration to load", default='0313600')
parser.add_argument('--predict_year', type=int,
                    help="year to predict on", default=2019)
parser.add_argument('--num_samples', type=int,
                    help="number of images to generate predictions for", default=5)
parser.add_argument('--pred_ensemble_size', type=int,
                    help="size of prediction ensemble", default=3)
parser.set_defaults(predict_full_image=False)
parser.set_defaults(plot_rapsd=False)
parser.set_defaults(include_Lanczos=False)
parser.set_defaults(include_RainFARM=False)
parser.set_defaults(include_deterministic=False)
parser.set_defaults(include_ecPoint=False)
parser.add_argument('--predict_full_image', dest='predict_full_image', action='store_true',
                    help="Predict on full images")
parser.add_argument('--plot_rapsd', dest='plot_rapsd', action='store_true',
                    help="Plot Radially Averaged Power Spectral Density")
parser.add_argument('--include_Lanczos', dest='include_Lanczos', action='store_true',
                    help="Include Lanczos benchmark")
parser.add_argument('--include_RainFARM', dest='include_RainFARM', action='store_true',
                    help="Include RainFARM benchmark")
parser.add_argument('--include_deterministic', dest='include_deterministic', action='store_true',
                    help="Include deterministic model for comparison")
parser.add_argument('--include_ecPoint', dest='include_ecPoint', action='store_true',
                    help="Include ecPoint benchmark")  
args = parser.parse_args()

log_folder = args.log_folder
model_number = args.model_number
predict_year = args.predict_year
num_samples = args.num_samples
pred_ensemble_size = args.pred_ensemble_size

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
batch_size = setup_params["TRAIN"]["batch_size"]
val_years = setup_params["VAL"]["val_years"]
lr_gen = setup_params["GENERATOR"]["learning_rate_gen"]
lr_gen = float(lr_gen)

if args.predict_full_image:
    batch_size = 1

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
    if args.include_RainFARM or args.include_ecPoint or args.include_Lanczos:
        raise Exception("Cannot include ecPoint/Lanczos/RainFARM results for downsampled problem")

## initialise model
model = setup_model(mode=mode,
                    arch=arch,
                    input_channels=input_channels,
                    filters_gen=filters_gen, 
                    filters_disc=filters_disc,
                    noise_channels=noise_channels, 
                    latent_variables=latent_variables,
                    padding=padding,
                    lr_gen=lr_gen)
gen = model.gen
gen.load_weights(weights_fn)

## load appropriate dataset
if args.predict_full_image:
    plot_label = 'large'
    data_predict = DataGeneratorFull(dates=dates, 
                                     ifs_fields=all_ifs_fields,
                                     batch_size=batch_size, 
                                     log_precip=True, 
                                     crop=True,
                                     shuffle=True,
                                     constants=True,
                                     hour='random',
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
                                 shuffle=True,
                                 hour='random',
                                 ifs_norm=False,
                                 downsample=downsample)    
if args.include_deterministic:
    if problem_type == 'superresolution':
        filters_det = 256
        gen_det_weights = '/ppdata/lucy-cGAN/logs/det/easy/models/gen_weights-0160000.h5'
    elif problem_type == 'normal':
        filters_det = 128
        gen_det_weights = '/ppdata/lucy-cGAN/logs/det/baseline/models/gen_weights-0160000.h5'
    gen_det = models.generator(mode='det', arch='normal', input_channels=input_channels, filters_gen=filters_det)
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
for i in range(num_samples):
    (inputs,outputs) = next(data_predict_iter)
    if problem_type == 'superresolution': # superresolution problem has only one input field
        inputs['lo_res_inputs'] = np.expand_dims(inputs['lo_res_inputs'][...,0], axis=-1)
    ## store denormalised inputs, outputs, predictions
    seq_const.append(inputs['hi_res_inputs'])
    input_conditions = inputs['lo_res_inputs'].copy()
    
    ##  denormalise precip inputs
    input_conditions[...,0:2] = data.denormalise(inputs['lo_res_inputs'][...,0:2])
    if problem_type != 'superresolution': # superresolution problem has only one input field
        ##  denormalise wind inputs
        input_conditions[...,-2] = inputs['lo_res_inputs'][...,-2]*ifs_norm['u700'][1] + ifs_norm['u700'][0]
        input_conditions[...,-1] = inputs['lo_res_inputs'][...,-1]*ifs_norm['v700'][1] + ifs_norm['v700'][0]
    seq_cond.append(input_conditions)    
    
    ## make sure ground truth image has correct dimensions
    if args.predict_full_image :
        sample = np.expand_dims(np.array(outputs['output']), axis=-1)
        seq_real.append(data.denormalise(sample))
    else:
        seq_real.append(data.denormalise(outputs['output']))

    if args.include_deterministic: # NB this is using det as a comparison
        seq_det.append(data.denormalise(gen_det.predict(inputs)))
    else:
        seq_det.append(dummy)

    pred_ensemble = []
    if mode == 'det': # this is plotting det as a model
        pred_ensemble_size = 1 # can't generate an ensemble with deterministic method
        pred_ensemble.append(data.denormalise(gen.predict(inputs))) #pretend it's an ensemble so dims match
        pred_ensemble = np.array(pred_ensemble)
        pred.append(pred_ensemble)
    else:
        if mode == 'GAN':
            noise_shape = inputs['lo_res_inputs'][0,...,0].shape + (noise_channels,)
            noise_hr_shape = inputs['hi_res_inputs'][0,...,0].shape + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            noise_hr_gen = NoiseGenerator(noise_hr_shape, batch_size=batch_size)
        elif mode == 'VAEGAN':
            noise_shape = inputs['lo_res_inputs'][0,...,0].shape + (latent_variables,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        if mode == 'VAEGAN':
            # call encoder once
            # mean, logvar = gen.encoder([inputs['lo_res_inputs'], inputs['hi_res_inputs']])
            mean, logvar = gen.encoder([inputs['lo_res_inputs']])         
        for j in range(pred_ensemble_size):
            inputs['noise_input'] = noise_gen()
            inputs['noise_hr_inputs'] = noise_hr_gen()
            if mode == 'GAN':
                gan_inputs = [inputs['lo_res_inputs'], inputs['hi_res_inputs'], inputs['noise_input'], inputs['noise_hr_input']]
                pred_ensemble.append(data.denormalise(gen.predict(gan_inputs)))
                print(f"sample number {i+1}")
                print(f"max predicted value is {np.amax(data.denormalise(gen.predict(gan_inputs)))}")
            elif mode == 'VAEGAN':
                dec_inputs = [mean, logvar, inputs['noise_input'], inputs['hi_res_inputs']]
                pred_ensemble.append(data.denormalise(gen.decoder.predict(dec_inputs)))
        pred_ensemble = np.array(pred_ensemble)
        pred.append(pred_ensemble)

data_ecpoint_iter = iter(data_ecpoint)
for i in range(num_samples):
    (inp,outp) = next(data_ecpoint_iter)        
    ## ecPoint prediction
    if args.include_ecPoint:
        seq_ecpoint.append(np.mean(benchmarks.ecpointPDFmodel(inp['lo_res_inputs']),axis=-1))
    else:
        seq_ecpoint.append(dummy)
    if args.include_RainFARM:
        seq_rainfarm.append(benchmarks.rainfarmmodel(inp['lo_res_inputs'][...,1]))
    else:
        seq_rainfarm.append(dummy)
    if args.include_Lanczos:
        seq_lanczos.append(benchmarks.lanczosmodel(inp['lo_res_inputs'][...,1]))
    else:
        seq_lanczos.append(dummy)

# plot input conditions and prediction example
# len(seq) = num_predictions (iterations through data generator)
# seq[0].shape = [NHWC], C=9 for cond, C=2 for const, C=1 for real, pred
# list entry[0] - sample image 0
IFS_total = seq_cond[0][0,...,0] #total precip
if problem_type != 'superresolution':
    IFS_conv = seq_cond[0][0,...,1] #convective precip
    IFS_u700 = seq_cond[0][0,...,-2] #u700
    IFS_v700 = seq_cond[0][0,...,-1] #v700
constant_0 = seq_const[0][0,...,0] #orog
constant_1 = seq_const[0][0,...,1] #lsm
NIMROD = seq_real[0][0,...,0]
pred_0_0 = pred[0][0][0,...,0] # [sample_images][pred_ensemble_size][NHWC]
pred_mean = pred[0][:,0,...,0].mean(axis=0) # mean of ensemble members for img 0

##colorbar
cb = list(range(0,6))
    
plt.figure(figsize=(8,7), dpi=200)
gs = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
if problem_type != 'superresolution':
    ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
else:
    ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[0, 2])
ax4 = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax5 = plt.subplot(gs[1, 1], projection=ccrs.PlateCarree())
ax6 = plt.subplot(gs[1, 2], projection=ccrs.PlateCarree())
ax = [ax1, ax2, ax3, ax4, ax5, ax6]

IFS_total_ax = ax[0].imshow(IFS_total, norm=colors.LogNorm(*value_range_precip), cmap=cmap, origin='lower', extent=extent, 
                            transform=ccrs.PlateCarree(), alpha=alpha)
ax[0].set_title(plot_input_title)
ax[0].coastlines(resolution='10m', color='black', linewidth=linewidth)
cb[2] = plt.colorbar(IFS_total_ax, ax=ax[0], norm=colors.LogNorm(*value_range_precip), orientation='horizontal',
                    fraction=0.035, pad=0.04)

if problem_type != 'superresolution':
    IFS_conv_ax = ax[1].imshow(IFS_conv, norm=colors.LogNorm(*value_range_precip), cmap=cmap, origin='lower', extent=extent, 
                               transform=ccrs.PlateCarree(), alpha=alpha)
    ax[1].set_title('IFS - convective precip')
    ax[1].coastlines(resolution='10m', color='black', linewidth=linewidth)
    cb[1] = plt.colorbar(IFS_conv_ax, ax=ax[1], norm=colors.LogNorm(*value_range_precip), orientation='horizontal',
                        fraction=0.035, pad=0.04)
else:
    LSM = ax[1].imshow(constant_1, cmap=cmap_lsm, origin='lower', alpha=alpha)
    ax[1].set_title('Land sea mask')
    cb[1] = plt.colorbar(LSM, ax=ax[1], orientation='horizontal', fraction=0.035, pad=0.04)

OROG = ax[2].imshow(constant_0, cmap="terrain", origin='lower', alpha=alpha)
ax[2].set_title('Orography')
cb[0] = plt.colorbar(OROG, ax=ax[2], norm=colors.Normalize(*value_range_orog), orientation='horizontal',
                    fraction=0.04, pad=0.04)

TRUTH = ax[3].imshow(NIMROD, norm=colors.LogNorm(*value_range_precip), cmap=cmap, origin='lower', extent=extent, 
                     transform=ccrs.PlateCarree(), alpha=alpha)
ax[3].set_title('NIMROD - ground truth')
ax[3].coastlines(resolution='10m', color='black', linewidth=linewidth)
cb[3] = plt.colorbar(TRUTH, ax=ax[3], norm=colors.LogNorm(*value_range_precip), orientation='horizontal',
                    fraction=0.035, pad=0.04)

PRED = ax[4].imshow(pred_0_0, norm=colors.LogNorm(*value_range_precip), cmap=cmap, origin='lower', extent=extent, 
                    transform=ccrs.PlateCarree(), alpha=alpha)
ax[4].set_title('GAN - example prediction')
ax[4].coastlines(resolution='10m', color='black', linewidth=linewidth)
cb[4] = plt.colorbar(PRED, ax=ax[4], norm=colors.LogNorm(*value_range_precip), orientation='horizontal',
                    fraction=0.035, pad=0.04)

PRED_mean = ax[5].imshow(pred_mean, norm=colors.LogNorm(*value_range_precip), cmap=cmap, origin='lower', extent=extent, 
                         transform=ccrs.PlateCarree(), alpha=0.8)
ax[5].set_title('GAN - mean prediction')
ax[5].coastlines(resolution='10m', color='black', linewidth=linewidth)
cb[5] = plt.colorbar(PRED_mean, ax=ax[5], norm=colors.LogNorm(*value_range_precip), orientation='horizontal',
                    fraction=0.035, pad=0.04)

for ax in ax:
    ax.tick_params(left=False, bottom=False,labelleft=False, labelbottom=False)
    
if problem_type != "superresolution":
    for cb in cb[1:]:
        cb.set_ticks(cb_tick_loc)
        cb.set_ticklabels(cb_tick_labels)
        ax.tick_params(labelsize=8)
        cb.set_label(units, size=8)
else:
    for cb in cb[2:]:
        cb.set_ticks(cb_tick_loc)
        cb.set_ticklabels(cb_tick_labels)
        ax.tick_params(labelsize=8)
        cb.set_label(units, size=8)


plt.savefig("{}/prediction-and-inputs-{}-{}-{}.png".format(log_folder, #cannot save as pdf - will produce artefacts
                                                          model_number,
                                                          problem_type,
                                                          plot_label), bbox_inches='tight')
plt.close()


## generate labels for plots
labels = [plot_input_title, "TRUTH"]
for i in range(pred_ensemble_size):
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
for i in range(num_samples):
    tmp = {}
    tmp['TRUTH'] = seq_real[i][0,...,0]
    tmp[plot_input_title] = seq_cond[i][0,...,0]
    tmp['Lanczos'] = seq_lanczos[i][0,...]
    tmp['RainFARM'] = seq_rainfarm[i][0,...]
    tmp['Deterministic'] = seq_det[i][0,...,0]
    tmp['ecPoint mean'] = seq_ecpoint[i][0,...]
    for j in range(pred_ensemble_size):
        tmp[f"{mode} pred {j+1}"] = pred[i][j][0,...,0]
    sequences.append(tmp)
    
num_cols = num_samples
num_rows = len(labels)+1
plt.figure(figsize=(1.5*num_cols,1.5*num_rows), dpi=300)

gs = gridspec.GridSpec(num_rows*num_rows,num_rows*num_cols,wspace=0.5,hspace=0.5)

for k in range(num_samples):
    for i in range(len(labels)):
        plt.subplot(gs[(num_rows*i):(num_rows+num_rows*i),(num_rows*k):(num_rows+num_rows*k)], 
                    projection=ccrs.PlateCarree())
        ax = plt.gca()
        ax.coastlines(resolution='10m', color='black', linewidth=linewidth)
        plot_img_log_coastlines(sequences[k][labels[i]], value_range_precip=value_range_precip, 
                                cmap=cmap, extent=extent, alpha=alpha)
        if i == 0:
            plt.title(k+1)

        if k==0:
            ax.set_ylabel(labels[i]) # cartopy takes over the xlabel and ylabel
            ax.set_yticks([]) # this weird hack restores them. WHY?!?!
            
plt.suptitle('Example predictions for different input conditions')

cax = plt.subplot(gs[-1,1:-1]).axes
cb = colorbar.ColorbarBase(cax, norm=colors.LogNorm(*value_range_precip), orientation='horizontal')
cb.set_ticks(cb_tick_loc)
cb.set_ticklabels(cb_tick_labels)
cax.tick_params(labelsize=12)
cb.set_label(units, size=12)
            
plt.savefig("{}/predictions-{}-{}-{}.png".format(log_folder, #cannot save as pdf - will produce artefacts
                                                 model_number,
                                                 problem_type,
                                                 plot_label), bbox_inches='tight')
plt.close()
gc.collect()

if args.plot_rapsd:
    colours = ['plum', 'palevioletred', 'lightslategrey', 'coral', 'lightblue', 'darkseagreen', 'mediumturquoise', 'purple', 'navy']
    plot_scales = [512, 256, 128, 64, 32, 16, 8, 4]
    # create a PdfPages object to save multiple plots to same pdf
    pdf = PdfPages("{}/RAPSD-{}-{}-{}.pdf".format(log_folder,
                                                  model_number,
                                                  problem_type,
                                                  plot_label))

    for k in range(num_samples):
        fig, ax = plt.subplots()
        for i in range(len(labels)):
            if labels[i] == 'IFS':
                ## skip the input data b/c the resolution is different
                pass
            else:
                R_1, freq_1 = rapsd(sequences[k][labels[i]], fft_method=np.fft, return_freq=True)
                # Plot the observed power spectrum and the model
                plot_spectrum1d(freq_1,
                                R_1,
                                x_units="km",
                                y_units="dBR",
                                color=colours[i],
                                ax=ax,
                                label=labels[i],
                                wavelength_ticks=plot_scales)


                plt.legend()
            
        ax.set_title(f"Radially averaged log-power spectrum - {k+1}")
        # save the current figure
        pdf.savefig(fig)
    plt.close()
    pdf.close()
