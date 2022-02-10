from tensorflow.python.keras.utils import generic_utils
import gc
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import ecpoint
import data
import crps
import benchmarks
import evaluation
from data import get_dates
from data_generator_ifs import DataGenerator as DataGeneratorFull
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--log_folder', type=str, 
                    help="directory to store results")
parser.add_argument('--eval_year', type=int,
                    help="year to predict on", default=2019)
parser.add_argument('--ensemble_members', type=int,
                    help="number of ensemble members", default=100)
parser.add_argument('--num_batches', type=int,
                    help="number of images to predict on", default=256)
parser.add_argument('--ecpoint_model', type=str,
                    help="type of ecpoint model to calculate ranks for", default='part-corr')
args = parser.parse_args()

eval_year = args.eval_year
ensemble_members = args.ensemble_members
num_batches = args.num_batches
log_folder = args.log_folder
ecpoint_model = args.ecpoint_model

# input parameters
out_fn = "{}/eval_noise_full_image_ecpoint_{}.txt".format(log_folder,str(eval_year))
load_full_image = True
add_noise = True
noise_factor = 1e-3
rank_samples = 100
denormalise_data = True
show_progress = True
normalize_ranks = True
batch_size = 1 # memory issues

evaluation.log_line(out_fn, "Method CRPS mean std")

# setup data
dates=get_dates(eval_year)
data_ecpoint = DataGeneratorFull(dates=dates,
                                    ifs_fields=ecpoint.ifs_fields,
                                    batch_size=batch_size,
                                    log_precip=False,
                                    crop=True,
                                    shuffle=True,
                                    constants=True,
                                    hour='random',
                                    ifs_norm=False)

ranks = []
batch_gen_iter  = iter(data_ecpoint)

if show_progress:
    # Initialize progbar and batch counter
    progbar = generic_utils.Progbar(num_batches)

for k in range(num_batches):
    # load truth images
    if load_full_image:
        inputs, outputs = next(batch_gen_iter)
        cond = inputs['lo_res_inputs']
        const = inputs['hi_res_inputs']
        sample_truth = outputs['output']
        sample_truth = np.expand_dims(np.array(sample_truth), axis=-1) # must be 4D tensor for pooling NHWC
    else:
        cond, const, sample = next(batch_gen_iter)
        sample_truth = sample.numpy()
    if denormalise_data:
        sample_truth = data.denormalise(sample_truth)
    if add_noise:
        noise_dim_1, noise_dim_2 = sample_truth[0, ..., 0].shape
        noise = np.random.rand(batch_size, noise_dim_1, noise_dim_2, 1)*noise_factor
        sample_truth += noise
        
    # generate predictions
    samples_ecpoint = []
    
    if ecpoint_model == 'ecpoint_no-corr': # pred_ensemble will be batch_size x H x W x ens
        sample_ecpoint = benchmarks.ecpointmodel(inputs['lo_res_inputs'],
                                                 ensemble_size=ensemble_members,
                                                 data_format="channels_last")
    elif ecpoint_model == 'ecpoint_part-corr':
        sample_ecpoint = benchmarks.ecpointboxensmodel(inputs['lo_res_inputs'],
                                                       ensemble_size=ensemble_members,
                                                       data_format="channels_last")
    elif ecpoint_model == 'ecpoint_full-corr': # this has ens=100 every time
        sample_ecpoint = benchmarks.ecpointPDFmodel(inputs['lo_res_inputs'],
                                                          data_format="channels_last")
    elif ecpoint_model == 'ecpoint_mean': # this has ens=100 every time
        sample_ecpoint = np.mean(benchmarks.ecpointPDFmodel(inputs['lo_res_inputs'],
                                                            data_format="channels_last"), axis=-1)
        
    # turn list into array
    samples_ecpoint = np.stack(samples_ecpoint, axis=-1) #shape of samples_ecpoint is [n, h, w, c] e.g. [1, 940, 940, 10]
    
    # calculate ranks
    # currently ranks only calculated without pooling
    # probably fine but may want to threshold in the future, e.g. <1mm, >5mm
    sample_truth_ranks = sample_truth.ravel() # unwrap into one long array, then unwrap samples_gen in same format
    samples_ecpoint_ranks = samples_ecpoint.reshape((-1, rank_samples)) # unknown batch size/img dims, known number of samples
    rank = np.count_nonzero(sample_truth_ranks[:, None] >= samples_ecpoint_ranks, axis=-1) # mask array where truth > samples gen, count
    ranks.append(rank)
    del samples_ecpoint_ranks, sample_truth_ranks
    gc.collect()
    
    # calculate CRPS score
    # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
    crps = crps.crps_ensemble(np.squeeze(sample_truth, axis=-1), samples_ecpoint).mean()
    
    if show_progress:
            losses = [("CRPS", np.mean(crps))]
            progbar.add(1, values=losses)
        

    ranks = np.concatenate(ranks)
    gc.collect()
    if normalize_ranks:
        ranks = ranks / rank_samples
        gc.collect()

# calculate mean and standard deviation
mean = ranks.mean()
std = ranks.std()

evaluation.log_line(out_fn, "{} {:.6f} {:.6f} {:.6f} ".format(ecpoint_model, crps, mean, std))

# save one directory up from model weights, in same dir as logfile
ranks_folder = os.path.dirname(out_fn)

if add_noise is False and load_full_image is False:
    fname = 'ranks-small_image-{}.npz'.format(ecpoint_model)
elif add_noise is True and load_full_image is False:
    fname = 'ranks-small_image-noise-{}.npz'.format(ecpoint_model)
elif add_noise is False and load_full_image is True:
    fname = 'ranks-full_image-{}.npz'.format(ecpoint_model)
elif add_noise is True and load_full_image is True:
    fname = 'ranks-full_image-noise-{}.npz'.format(ecpoint_model)
np.savez(os.path.join(ranks_folder, fname), ranks) 