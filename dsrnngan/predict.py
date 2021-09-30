import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors, gridspec
from noise import noise_generator
import train
import ecpoint
import data
import models
import benchmarks
import argparse
from tfrecords_generator_ifs import create_fixed_dataset
from data_generator_ifs import DataGenerator as DataGeneratorFull
from data import get_dates
from plots import plot_img, plot_img_log

parser = argparse.ArgumentParser()
parser.add_argument('--load_weights_root', type=str, 
                    help="directory where model weights are saved", 
                    default='/ppdata/lucy-cGAN/logs/IFS/gen_128_disc_512/noise_4/lr1e-5')
parser.add_argument('--model_number', type=str, 
                    help="model iteration to load", default='0313600')
parser.add_argument('--noise_channels', type=int,
                    help="Number of noise channels passed to generator", default=4)
parser.add_argument('--constant_fields', type=int,
                    help="Number of constant fields passed to generator and discriminator", default=2)
parser.add_argument('--input_channels', type=int,
                    help="Dimensions of input condition passed to generator and discriminator", default=9)
parser.add_argument('--problem_type', type=str, 
                    help="normal (IFS>NIMROD), easy (NIMROD>NIMROD)", default="normal")
parser.add_argument('--predict_full_image', type=bool,
                    help="False (small images used for training), True (full image)", default=False)
parser.add_argument('--include_Lanczos', type=bool,
                    help="True or False)", default=False)
parser.add_argument('--include_RainFARM', type=bool,
                    help="True or False)", default=False)
parser.add_argument('--include_deterministic', type=bool,
                    help="True or False)", default=False)
parser.add_argument('--include_ecPoint', type=bool,
                    help="True or False)", default=False)
parser.add_argument('--predict_year', type=int,
                    help="year to predict on", default=2019)
parser.add_argument('--num_predictions', type=int,
                    help="number of images to predict on", default=5)
parser.add_argument('--num_samples', type=int,
                    help="size of prediction ensemble", default=3)
parser.add_argument('--batch_size', type=int,
                    help="Batch size", default=1)
parser.add_argument('--filters_gen', type=int, default=128,
        help="Number of filters used in generator")
parser.add_argument('--filters_disc', type=int, default=512,
        help="Number of filters used in discriminator")
parser.add_argument('--learning_rate_disc', type=float, default=1e-5,
        help="Learning rate used for discriminator optimizer")
parser.add_argument('--learning_rate_gen', type=float, default=1e-5,
        help="Learning rate used for generator optimizer")
args = parser.parse_args()

## load parameters
load_weights_root = args.load_weights_root
model_number = args.model_number
noise_channels = args.noise_channels
input_channels = args.input_channels
constant_fields = args.constant_fields
problem_type = args.problem_type
predict_full_image = args.predict_full_image
include_Lanczos = args.include_Lanczos
include_RainFARM = args.include_RainFARM
include_deterministic = args.include_deterministic
include_ecPoint = args.include_ecPoint
predict_year = args.predict_year
num_predictions = args.num_predictions
num_samples = args.num_samples
#batch_size default is 1 to avoid issues loading full image data
batch_size = args.batch_size
filters_disc = args.filters_disc
filters_gen = args.filters_gen
lr_disc = args.learning_rate_disc
lr_gen = args.learning_rate_gen

weights_fn = load_weights_root + '/' + 'gen_weights-IFS-{}.h5'.format(model_number)
print(weights_fn)

#weights = np.arange(6,2,-1)                                                                                                
#weights = weights / weights.sum() 
weights = None
dates = get_dates(predict_year)

if problem_type == "normal":
    downsample = False
    plot_input_title = 'IFS'
    input_channels = input_channels
elif problem_type == "easy":
    downsample = True
    plot_input_title = 'Downsampled'
    input_channels = 1
    if include_RainFARM or include_ecPoint or include_Lanczos:
        raise Exception("Cannot include ecPoint/Lanczos/RainFARM results for downsampled problem")

## initialise GAN
(wgan) = train.setup_gan(train_years=None, 
                         val_years=None, 
                         val_size=None, 
                         downsample=downsample, 
                         weights=weights,
                         input_channels=input_channels,
                         constant_fields=constant_fields,
                         batch_size=batch_size,
                         filters_gen=filters_gen, 
                         filters_disc=filters_disc,
                         noise_channels=noise_channels, 
                         lr_disc=lr_disc, 
                         lr_gen=lr_gen)

## load GAN model
gen = wgan.gen
gen.load_weights(weights_fn)

##load det model
## current best deterministic model
if problem_type == 'easy':
    filters_det = 256
    gen_det_weights = '/ppdata/lucy-cGAN/logs/EASY/deterministic/filters_256/gen_det_weights-IFS-0400000.h5'
elif problem_type == 'normal':
    filters_det = 128
    gen_det_weights = '/ppdata/lucy-cGAN/logs/IFS/filters_128/softplus/det/lr_1e-4/gen_det_weights-ERA-0400000.h5'
gen_det = models.generator_deterministic(input_channels=input_channels, filters_gen=filters_det)
gen_det.load_weights(gen_det_weights)

## load appropriate dataset
if predict_full_image:
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

if not predict_full_image:
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

pred = []
seq_real=[]
seq_cond=[]
seq_const=[]
seq_lanczos = []
seq_rainfarm = []
seq_det = []
seq_ecpoint = []

data_predict_iter = iter(data_predict)    
for i in range(num_predictions):
    (inputs,outputs) = next(data_predict_iter)
    ## retrieve noise dimensions from input condition
    noise_dim = inputs['generator_input'][0,...,0].shape + (noise_channels,)
    inputs['noise_input'] = noise_generator(noise_dim, batch_size=batch_size)
    ## store denormalised inputs, outputs, predictions
    seq_const.append(data.denormalise(inputs['constants']))
    seq_cond.append(data.denormalise(inputs['generator_input']))    
    ## make sure ground truth image has correct dimensions
    if predict_full_image == True:
        sample = np.expand_dims(np.array(outputs['generator_output']), axis=-1)
        seq_real.append(data.denormalise(sample))
    elif predict_full_image == False:
        seq_real.append(data.denormalise(outputs['generator_output']))
    seq_det.append(data.denormalise(gen_det.predict(inputs)))
    ## Generate ensemble members
    pred_ensemble = []
    for j in range(num_samples):
        ## retrieve noise dimensions from input condition
        noise_dim = inputs['generator_input'][0,...,0].shape + (noise_channels,) 
        inputs['noise_input'] = noise_generator(noise_dim, batch_size=batch_size)
        pred_ensemble.append(data.denormalise(gen.predict(inputs)))
    pred_ensemble = np.array(pred_ensemble)
    pred.append(pred_ensemble)
    
data_ecpoint_iter = iter(data_ecpoint)
dummy = np.zeros((1, 940, 940))
for i in range(num_predictions):
    (inp,outp) = next(data_ecpoint_iter)        
    ## ecPoint prediction
    if include_ecPoint:
        seq_ecpoint.append(np.mean(benchmarks.ecpointPDFmodel(inp['generator_input']),axis=-1))
    if not include_ecPoint:
        seq_ecpoint.append(dummy)
    if include_RainFARM:
        seq_rainfarm.append(benchmarks.rainfarmmodel(inp['generator_input'][...,1]))
    if not include_RainFARM:
        seq_rainfarm.append(dummy)
    if include_Lanczos:
        seq_lanczos.append(benchmarks.lanczosmodel(inp['generator_input'][...,1]))
    if not include_Lanczos:
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

plt.savefig("{}/prediction-and-input-{}-{}.pdf".format(load_weights_root, 
                                                       problem_type,
                                                       plot_label), bbox_inches='tight')
plt.close()


## generate labels for plots
labels = [plot_input_title, "NIMROD"]
for i in range(num_samples):
    labels.append(f"GAN pred {i+1}")
if include_RainFARM:
    labels.append("RainFARM")
if include_ecPoint:
    labels.append("ecPoint mean")
if include_deterministic:
    labels.append("Deterministic")
if include_Lanczos:
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
        tmp[f"GAN pred {j+1}"] = pred[i][j][0,...,0]
    sequences.append(tmp)
    
num_cols = num_predictions
num_rows = len(labels)+1
plt.figure(figsize=(1.5*num_cols,1.5*num_rows))
value_range = (0.1,10)
gs = gridspec.GridSpec(num_rows*num_rows,num_rows*num_cols,wspace=0.5,hspace=0.5)

for k in range(num_predictions):
    for i in range(len(labels)):
        plt.subplot(gs[(num_rows*i):(num_rows+num_rows*i),(num_rows*k):(num_rows+num_rows*k)])
        plot_img_log(sequences[k][labels[i]], value_range=value_range)
        if k==0:
            plt.ylabel(labels[i])
plt.suptitle('Example predictions for different input conditions')
##colorbar
units = "Rain rate [mm h$^{-1}$]"
cb_tick_loc = np.array([0.1, 1, 2, 5, 10])
cb_tick_labels = [0.1, 1, 2, 5, 10]
cax = plt.subplot(gs[-1,1:-1]).axes
cb = colorbar.ColorbarBase(cax, norm=colors.LogNorm(*value_range), orientation='horizontal')
cb.set_ticks(cb_tick_loc)
cb.set_ticklabels(cb_tick_labels)
cax.tick_params(labelsize=16)
cb.set_label(units, size=16)
            
plt.savefig("{}/predictions-{}-{}.pdf".format(load_weights_root, 
                                              problem_type,
                                              plot_label), bbox_inches='tight')
plt.close()
