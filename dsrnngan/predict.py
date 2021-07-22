import numpy as np
import matplotlib.pyplot as plt
import train
import argparse
from tfrecords_generator_ifs import create_fixed_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--load_weights_root', type=str, 
                    help="directory where model weights are saved", 
                    default='/ppdata/lucy-cGAN/logs/IFS/gen_128_disc_512/noise_4/lr1e-5')
parser.add_argument('--model_number', type=str, 
                    help="model iteration to load", default='0313600')
parser.add_argument('--train_years', type=int, nargs='+', default=[2016, 2017, 2018],
                        help="Training years")
parser.add_argument('--val_years', type=int, nargs='+', default=2019,
                        help="Validation years -- cannot pass a list if using create_fixed_dataset")
parser.add_argument('--val_size', type=int, default=8,
                        help='Num val examples')
parser.add_argument('--noise_channels', type=int,
                    help="Number of channels of noise passed to generator", default=4)
parser.add_argument('--constant_fields', type=int,
                    help="Number of constant fields passed to generator and discriminator", default=2)
parser.add_argument('--input_channels', type=int,
                    help="Dimensions of input condition passed to generator and discriminator", default=9)
parser.add_argument('--problem_type', type=str, 
                    help="normal (IFS>NIMROD), easy (NIMROD>NIMROD)", default="normal")
parser.add_argument('--predict_year', type=int,
                    help="year to predict on", default=2019)
parser.add_argument('--batch_size', type=int,
                    help="Batch size", default=16)
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
batch_size = args.batch_size
predict_year = args.predict_year
train_years = args.train_years
val_years = args.val_years
val_size = args.val_size
filters_disc = args.filters_disc
filters_gen = args.filters_gen
lr_disc = args.learning_rate_disc
lr_gen = args.learning_rate_gen

noise_dim = (10,10) + (noise_channels,)
weights_fn = load_weights_root + '/' + 'gen_weights-IFS-{}.h5'.format(model_number)
print(weights_fn)

if problem_type == "normal":
    downsample = False
    plot_input_title = 'IFS'
    input_channels = input_channels
elif problem_type == "easy":
    downsample = True
    plot_input_title = 'NIMROD - downscaled'
    input_channels = 1
else:
    raise Exception("no such problem type, try again!")

## initialise GAN
(wgan, _, _, _, _, _) = train.setup_gan(train_years, 
                                        val_years, 
                                        val_size=val_size, 
                                        downsample=downsample, 
                                        input_channels=input_channels,
                                        constant_fields=constant_fields,
                                        batch_size=batch_size, 
                                        filters_gen=filters_gen, 
                                        filters_disc=filters_disc,
                                        noise_channels=noise_channels, 
                                        lr_disc=lr_disc, 
                                        lr_gen=lr_gen)

## load model
gen = wgan.gen
gen.load_weights(weights_fn)


## load dataset
data_predict = create_fixed_dataset(predict_year, batch_size=batch_size, downsample=downsample)

## create noise
def noise_generator(shape, batch_size, random_seed=None, mean=0.0, std=1.0):
    rng = np.random.RandomState(seed=random_seed)
    shape = (batch_size, ) + shape
    n = rng.randn(*shape).astype(np.float32)
    if std != 1.0:
        n *= std
    if mean != 0.0:
        n += mean
    return n

## make 50 sets of predictions
pred = {}
inputs = {}
for i,batch in enumerate(data_predict):
    if i == 50:
        break
    inputs['generator_input'] = batch[0]
    inputs['constants'] = batch[1]
    outputs = batch[2]
    inputs['noise_input'] = noise_generator(noise_dim, batch_size=batch_size)
    pred[i] = np.array(gen.predict_on_batch(inputs))

## turn results into arrays
input_img = np.array(inputs['generator_input'])
const = np.array(inputs['constants'])
gt = np.array(outputs)


## plot results 
## batch 0
IFS_0 = input_img[0][:,:,0]
NIMROD_0 = np.squeeze(gt[0], axis=-1)
pred_0_1 = np.squeeze(pred[0][0], axis = -1)
pred_0_10 = np.squeeze(pred[9][0], axis = -1)
pred_0_50 = np.squeeze(pred[49][0], axis = -1)

## batch 1
IFS_1 = input_img[1][:,:,0]
NIMROD_1 = np.squeeze(gt[1], axis=-1)
pred_1_1 = np.squeeze(pred[0][1], axis = -1)
pred_1_10 = np.squeeze(pred[9][1], axis = -1)
pred_1_50 = np.squeeze(pred[49][1], axis = -1)

## batch 2
IFS_2 = input_img[2][:,:,0]
NIMROD_2 = np.squeeze(gt[2], axis=-1)
pred_2_1 = np.squeeze(pred[0][2], axis = -1)
pred_2_10 = np.squeeze(pred[9][2], axis = -1)
pred_2_50 = np.squeeze(pred[49][2], axis = -1)

## batch 3                                                                                                                       
IFS_3 = input_img[3][:,:,0]
NIMROD_3 = np.squeeze(gt[3], axis=-1)
pred_3_1 = np.squeeze(pred[0][3], axis = -1)
pred_3_10 = np.squeeze(pred[9][3], axis = -1)
pred_3_50 = np.squeeze(pred[49][3], axis = -1)

fig, axs = plt.subplots(4, 5, figsize=(12,7), subplot_kw=dict(sharex=True, sharey=True))
axs[0, 0].set_title(plot_input_title)
axs[0, 0].set_ylabel('batch 0')
axs[0, 0].imshow(IFS_0)
axs[0, 1].set_title('NIMROD')
axs[0, 1].imshow(NIMROD_0)
axs[0, 2].set_title('Pred 1')
axs[0, 2].imshow(pred_0_1)
axs[0, 3].set_title('Pred 10')
axs[0, 3].imshow(pred_0_10)
axs[0, 4].set_title('Pred 50')
axs[0, 4].imshow(pred_0_50)
axs[1, 0].set_ylabel('batch 1')
axs[1, 0].imshow(IFS_1)
axs[1, 1].imshow(NIMROD_1)
axs[1, 2].imshow(pred_1_1)
axs[1, 3].imshow(pred_1_10)
axs[1, 4].imshow(pred_1_50)
axs[2, 0].set_ylabel('batch 2')
axs[2, 0].imshow(IFS_2)
axs[2, 1].imshow(NIMROD_2)
axs[2, 2].imshow(pred_2_1) 
axs[2, 3].imshow(pred_2_10)
axs[2, 4].imshow(pred_2_50)
axs[3, 0].set_ylabel('batch 3')
axs[3, 0].imshow(IFS_3)
axs[3, 1].imshow(NIMROD_3)
axs[3, 2].imshow(pred_3_1)
axs[3, 3].imshow(pred_3_10)
axs[3, 4].imshow(pred_3_50)

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.savefig("{}/predictions.pdf".format(load_weights_root), bbox_inches='tight')
plt.close()

## plot prediction and inputs
## batch 0
input_condition = input_img[0][:,:,0]
constant_0 = const[0][:,:,0]
constant_1 = const[0][:,:,1]
NIMROD = np.squeeze(gt[0], axis=-1)
pred_0_0 = np.squeeze(pred[49][0], axis = -1)

fig, ax = plt.subplots(1,5, figsize=(9,3))
ax[0].imshow(input_condition, vmin=0, vmax=1)
ax[0].set_title(plot_input_title)
ax[1].imshow(constant_0, vmin=0, vmax=1)
ax[1].set_title('constant_0')
ax[2].imshow(constant_1, vmin=0, vmax=1)
ax[2].set_title('constant_1')
ax[3].imshow(NIMROD, vmin=0, vmax=1)
ax[3].set_title('NIMROD')
ax[4].imshow(pred_0_0, vmin=0, vmax=1)
ax[4].set_title('Prediction')
for ax in ax.flat:
    ax.label_outer()

# Hide x labels and tick labels for top plots and y ticks for right plots.                                                       
for ax in axs.flat:
    ax.label_outer()

plt.savefig("{}/prediction-and-input.pdf".format(load_weights_root), bbox_inches='tight')
plt.close()
