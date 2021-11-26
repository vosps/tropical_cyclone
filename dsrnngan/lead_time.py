import argparse
import yaml
import os
from setupmodel import setup_model
from data_generator_ifsall import DataGenerator
from tensorflow.python.keras.utils import generic_utils
import numpy as np
import data
from noise import NoiseGenerator
import crps
import gc
import benchmarks
import ecpoint
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to configuration file")
group = parser.add_mutually_exclusive_group()
group.add_argument('--eval_full', dest='evalnum', action='store_const', const="full")
group.add_argument('--eval_short', dest='evalnum', action='store_const', const="short")
group.add_argument('--eval_tenth', dest='evalnum', action='store_const', const="tenth")
group.add_argument('--eval_blitz', dest='evalnum', action='store_const', const="blitz")
parser.set_defaults(evalnum=None)
parser.add_argument('--start_time', type=int, 
                    help="lead time to start at", default='1')
parser.add_argument('--stop_time', type=int, 
                    help="lead time to stop at", default='72')
parser.add_argument('--model_number', type=int, 
                    help="model number for GAN", default='0147200')
args = parser.parse_args()

if args.evalnum is None and (args.rank_small or args.rank_full or args.qual_small or args.qual_full or args.plot_roc_small or args.plot_roc_full):
    raise RuntimeError("You asked for evaluation to occur, but did not pass in '--eval_full', '--eval_short', '--eval_tenth', or '--eval_blitz' to specify length of evaluation")

# read in the configurations
if args.config is not None:
    config_path = args.config
else:
    raise Exception("Please specify configuration!")

with open(config_path, 'r') as f:
    try:
        setup_params = yaml.safe_load(f)
        print(setup_params)
    except yaml.YAMLError as exc:
        print(exc)
        
mode = setup_params["GENERAL"]["mode"]
arch = setup_params["MODEL"]["architecture"]
padding = setup_params["MODEL"]["padding"]
log_folder = setup_params["SETUP"]["log_folder"]
problem_type = setup_params["GENERAL"]["problem_type"]
filters_gen = setup_params["GENERATOR"]["filters_gen"]
noise_channels = setup_params["GENERATOR"]["noise_channels"]
latent_variables = setup_params["GENERATOR"]["latent_variables"]
filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]
batch_size = setup_params["TRAIN"]["batch_size"]
num_batches = setup_params["EVAL"]["num_batches"]
add_noise = setup_params["EVAL"]["add_postprocessing_noise"]
noise_factor = setup_params["EVAL"]["postprocessing_noise_factor"]

# otherwise these are of type string, e.g. '1e-5'
noise_factor = float(noise_factor)

if mode not in ['GAN', 'VAEGAN', 'det']:
    raise ValueError("Mode type is restricted to 'GAN' 'VAEGAN' 'det'")
if problem_type != 'normal':
    raise RuntimeError("Problem type is restricted to 'normal', lead time datagenerator not configured for easy problem")

downsample = False
input_channels = 9
eval_year = 2020 # only 2020 data is available
all_ifs_fields = ['tp','cp' ,'sp' ,'tisr','cape','tclw','tcwv','u700','v700']
num_batches = 256 # for now
rank_samples = 100 # like ecPoint
denormalise_data = True
if mode == "det":
    rank_samples = 1  # can't generate an ensemble deterministically

## where to find model weights
weights_fn = os.path.join(log_folder, 'models', 'gen_weights-{}.h5'.format(args.model_number))
## where to save results
lead_time_fname_model = os.path.join(log_folder, "model-lead-time.pickle")
lead_time_fname_ecpoint = os.path.join(log_folder, "ecpoint-lead-time.pickle")

## initialise model
model = setup_model(mode=mode,
                    arch=arch,
                    input_channels=input_channels,
                    filters_gen=filters_gen, 
                    filters_disc=filters_disc,
                    noise_channels=noise_channels, 
                    latent_variables=latent_variables,
                    padding=padding)

gen = model.gen
gen.load_weights(weights_fn)

crps_scores_model = {}
crps_scores_ecpoint = {}

for hour in list(range(args.start_hour, args.stop_hour+1)): 
    
    # load data generators for this hour
    data_gen = DataGenerator(year=eval_year,
                            lead_time=hour,
                            ifs_fields=all_ifs_fields,
                            batch_size=batch_size,
                            log_precip=True,
                            crop=True,
                            shuffle=True,
                            constants=True,
                            hour='random',
                            ifs_norm=True,
                            downsample=downsample)

    data_gen_iter = iter(data_gen)
    
    data_benchmarks = DataGenerator(year=eval_year,
                                    lead_time=hour,
                                    ifs_fields=ecpoint.ifs_fields,
                                    batch_size=batch_size,
                                    log_precip=False,
                                    crop=True,
                                    shuffle=True,
                                    constants=True,
                                    hour='random',
                                    ifs_norm=False)
    
    data_benchmarks_iter = iter(data_benchmarks)
    
    # Initialize progbar and batch counter
    progbar = generic_utils.Progbar(num_batches)
    
    for k in range(num_batches):
        # retrieve model data
        inputs, outputs = next(data_gen_iter)
        cond = inputs['lo_res_inputs']
        const = inputs['hi_res_inputs']
        sample_truth = outputs['output']
        if denormalise_data:
            sample_truth = data.denormalise(sample_truth)
            
        sample_truth = np.expand_dims(np.array(sample_truth), axis=-1) # must be 4D tensor for pooling NHWC
        # retrieve ecpoint data
        inp_benchmarks, outp_benchmarks = next(data_benchmarks_iter)
        ecpoint_sample = benchmarks.ecpointmodel(inp_benchmarks['lo_res_inputs'], ensemble_size=rank_samples)
        ecpoint_truth = outp_benchmarks['output']
        
        if add_noise:
            noise_dim_1, noise_dim_2 = sample_truth[0, ..., 0].shape
            noise = np.random.rand(batch_size, noise_dim_1, noise_dim_2, 1)*noise_factor
            sample_truth += noise
            noise = np.random.rand(batch_size, noise_dim_1, noise_dim_2, 1)*noise_factor
            ecpoint_truth += noise
            noise = np.random.rand(batch_size, noise_dim_1, noise_dim_2, rank_samples)*noise_factor
            ecpoint_sample += noise
            
        # generate predictions, depending on model type
        samples_gen = []
        if mode == "GAN":
            noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            for i in range(rank_samples):
                nn = noise_gen()
                sample_gen = gen.predict([cond, const, nn])
                samples_gen.append(sample_gen.astype("float32"))
        elif mode == "det":
            sample_gen = gen.predict([cond, const])
            samples_gen.append(sample_gen.astype("float32"))
        elif mode == 'VAEGAN':
            # call encoder once
            (mean, logvar) = gen.encoder([cond, const])
            noise_shape = np.array(cond)[0, ..., 0].shape + (latent_variables,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
            for i in range(rank_samples):
                nn = noise_gen()
                # generate ensemble of preds with decoder
                sample_gen = gen.decoder.predict([mean, logvar, nn, const])
                samples_gen.append(sample_gen.astype("float32"))
        for ii in range(len(samples_gen)):
            sample_gen = np.squeeze(samples_gen[ii], axis=-1) # squeeze out trival dim
            #sample_gen shape should be [n, h, w] e.g. [1, 940, 940]
            if denormalise_data:
                sample_gen = data.denormalise(sample_gen)
            if add_noise:
                (noise_dim_1, noise_dim_2) = sample_gen[0, ...].shape
                noise = np.random.rand(batch_size, noise_dim_1, noise_dim_2)*noise_factor
                sample_gen += noise
            samples_gen[ii] = sample_gen
        # turn list into array
        samples_gen = np.stack(samples_gen, axis=-1) #shape of samples_gen is [n, h, w, c] e.g. [1, 940, 940, 10]
        
        # calculate CRPS score. Pooling not implemented here.     
        # crps_ensemble expects truth dims [N, H, W], pred dims [N, H, W, C]
        crps_score_model = crps.crps_ensemble(np.squeeze(sample_truth, axis=-1), samples_gen).mean()
        crps_score_ecpoint = crps.crps_ensemble(ecpoint_truth, ecpoint_sample).mean()
        del sample_truth, samples_gen, ecpoint_truth, ecpoint_sample
        gc.collect()

        if hour not in crps_scores_model.keys():
            crps_scores_model[hour] = []
            crps_scores_ecpoint[hour] = []
        crps_scores_model[hour].append(crps_score_model)
        crps_scores_ecpoint[hour].append(crps_score_ecpoint)
            
        crps_mean = np.mean(crps_scores_model[hour])
        losses = [("CRPS", crps_mean)]
        progbar.add(1, values=losses)

with open(lead_time_fname_model, 'wb') as handle:
    pickle.dump(crps_scores_model, handle)
    
with open(lead_time_fname_ecpoint, 'wb') as handle:
    pickle.dump(crps_scores_ecpoint, handle)