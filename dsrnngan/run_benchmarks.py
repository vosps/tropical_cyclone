import argparse
import os
import gc
from data import get_dates
from data_generator_ifs import DataGenerator as DataGeneratorFull
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
import ecpoint
import benchmarks
import numpy as np
import crps
from evaluation import rapsd_batch, log_line

parser = argparse.ArgumentParser()
parser.add_argument('--log_folder', type=str, 
                    help="directory to store results")
parser.add_argument('--predict_year', type=int,
                    help="year to predict on", default=2019)
parser.add_argument('--num_batches', type=int,
                    help="number of images to predict on", default=256)
parser.set_defaults(max_pooling=False)
parser.set_defaults(avg_pooling=False)
parser.set_defaults(include_Lanczos=False)
parser.set_defaults(include_RainFARM=False)
parser.set_defaults(include_ecPoint=False)
parser.set_defaults(include_ecPoint_mean=False)
parser.set_defaults(include_constant=False)
parser.set_defaults(include_zeros=False)
parser.add_argument('--include_Lanczos', dest='include_Lanczos', action='store_true',
                    help="Include Lanczos benchmark")
parser.add_argument('--include_RainFARM', dest='include_RainFARM', action='store_true',
                    help="Include RainFARM benchmark")
parser.add_argument('--include_ecPoint', dest='include_ecPoint', action='store_true',
                    help="Include ecPoint benchmark")
parser.add_argument('--include_ecPoint_mean', dest='include_ecPoint_mean', action='store_true',
                    help="Include ecPoint mean benchmark")
parser.add_argument('--include_constant', dest='include_constant', action='store_true',
                    help="Include constant upscaling as benchmark")  
parser.add_argument('--include_zeros', dest='include_zeros', action='store_true',
                    help="Include zero prediction benchmark")  
parser.add_argument('--max_pooling', dest='max_pooling', action='store_true',
                    help="Include max pooling for CRPS")
parser.add_argument('--avg_pooling', dest='avg_pooling', action='store_true',
                    help="Include average pooling for CRPS")
args = parser.parse_args()

predict_year = args.predict_year
num_batches = args.num_batches
log_folder = args.log_folder
batch_size = 1 # memory issues
log_fname = os.path.join(log_folder, "benchmarks.txt")

# setup data
dates=get_dates(predict_year)
data_benchmarks = DataGeneratorFull(dates=dates,
                                    ifs_fields=ecpoint.ifs_fields,
                                    batch_size=batch_size,
                                    log_precip=False,
                                    crop=True,
                                    shuffle=True,
                                    constants=True,
                                    hour='random',
                                    ifs_norm=False)

benchmark_methods = []
if args.include_Lanczos:
    benchmark_methods.append('lanczos')
if args.include_RainFARM:
    benchmark_methods.append('rainfarm')
if args.include_ecPoint:
    benchmark_methods.append('ecpoint')
if args.include_ecPoint_mean:
    benchmark_methods.append('ecpoint_mean')
if args.include_constant:
    benchmark_methods.append('constant')
if args.include_zeros:
    benchmark_methods.append('zeros')


log_line(log_fname, "Number of samples {}".format(num_batches))
log_line(log_fname, "Evaluation year {}".format(predict_year))
log_line(log_fname, "Model CRPS CRPS_max_4 CRPS_max_16 CRPS_avg_4 CRPS_avg_16 RMSE MAE RAPSD")    

sample = {}
sample_crps = {}
crps_scores = {}
rmse_scores = {}
mae_scores = {}
rapsd_scores = {}
data_benchmarks_iter = iter(data_benchmarks)
for i in range(num_batches):
    print(f" calculating for sample number {i+1} of {num_batches}")
    (inp,outp) = next(data_benchmarks_iter)
    sample_truth = outp['output']
    
    if args.max_pooling:
        max_pool_2d_4 = MaxPooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')
        sample_crps['max_4'] = max_pool_2d_4(sample_truth.copy()).numpy()
        max_pool_2d_16 = MaxPooling2D(pool_size=(16, 16), strides=(1, 1), padding='valid')
        sample_crps['max_16'] = max_pool_2d_16(sample_truth.copy()).numpy()
    if args.avg_pooling:
        avg_pool_2d_4 = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')
        sample_crps['avg_4'] = avg_pool_2d_4(sample_truth.copy()).numpy()
        avg_pool_2d_16 = AveragePooling2D(pool_size=(16, 16), strides=(1, 1), padding='valid')
        sample_crps['avg_16'] = avg_pool_2d_16(sample_truth.copy()).numpy()
    sample_crps['no-pooling'] = sample_truth
    pooling_methods = ['no-pooling']
    if args.max_pooling:
        pooling_methods.append('max_4')
        pooling_methods.append('max_16')
    if args.avg_pooling:
        pooling_methods.append('avg_4')
        pooling_methods.append('avg_16')


    sample['lanczos'] = benchmarks.lanczosmodel(inp['lo_res_inputs'][...,1])
    sample['rainfarm'] = benchmarks.rainfarmensemble(inp['lo_res_inputs'][...,1])
    sample['ecpoint'] = benchmarks.ecpointPDFmodel(inp['lo_res_inputs'])
    sample['ecpoint_mean'] = np.mean(benchmarks.ecpointPDFmodel(inp['lo_res_inputs']),axis=-1)
    sample['constant'] = benchmarks.constantupscalemodel(inp['lo_res_inputs'][...,1])
    sample['zeros'] = benchmarks.zerosmodel(inp['lo_res_inputs'][...,1])


    for benchmark in benchmark_methods:
        sample[benchmark]['no_pooling'] = sample[benchmark]
        if args.max_pooling:
            sample[benchmark]['max_4'] = max_pool_2d_4(sample[benchmark].copy())
            sample[benchmark]['max_16'] = max_pool_2d_16(sample[benchmark].copy())
        if args.avg_pooling:
            sample[benchmark]['avg_4'] = avg_pool_2d_4(sample[benchmark].copy())
            sample[benchmark]['avg_16'] = avg_pool_2d_16(sample[benchmark].copy())
        for method in pooling_methods:
            if benchmark in ['rainfarm', 'ecpoint']:
                crps_score = (crps.crps_ensemble(sample_crps[method], sample[benchmark][method])).mean()
                for j in range(sample[benchmark].shape[-1]):
                    rmse_tmp = np.sqrt(((sample_truth - sample[benchmark][...,j])**2).mean(axis=(1,2)))
                    mae_tmp = (np.abs(sample_truth - sample[benchmark][...,j])).mean(axis=(1,2))
                    rapsd_tmp = rapsd_batch(sample_truth, sample[benchmark][...,j])
                rmse_score = rmse_tmp.mean()
                mae_score = mae_tmp.mean()
                rapsd_score = rapsd_tmp.mean()
                gc.collect()
            else:
                crps_score = benchmarks.mean_crps(sample_crps[method], sample[benchmark][method])
                rmse_score = (np.sqrt(((sample_truth - sample[benchmark])**2)).mean(axis=(1,2)))
                mae_score = (np.abs(sample_truth - sample[benchmark])).mean(axis=(1,2))
                if benchmark not in ['zeros']:
                    rapsd_score = rapsd_batch(sample_truth, sample[benchmark])
            crps_scores[benchmark][method].append(crps_score)
        rmse_scores[benchmark].append(rmse_score)
        mae_scores[benchmark].append(mae_score)
        rapsd_scores[benchmark].append(rapsd_score)
    
for benchmark in benchmark_methods:
    if not args.max_pooling:
        crps_scores[benchmark]['max_4'] = np.nan
        crps_scores[benchmark]['max_16'] = np.nan
    if not args.avg_pooling:
        crps_scores[benchmark]['avg_4'] = np.nan
        crps_scores[benchmark]['avg_16'] = np.nan
    
    log_line(log_fname, "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                                                                benchmark,
                                                                crps_scores[benchmark]['no-pooling'].mean(),
                                                                crps_scores[benchmark]['max_4'].mean(),
                                                                crps_scores[benchmark]['max_16'].mean(),
                                                                crps_scores[benchmark]['avg_4'].mean(),
                                                                crps_scores[benchmark]['avg_16'].mean(),
                                                                rmse_scores[benchmark].mean(),
                                                                mae_scores[benchmark].mean(),
                                                                rapsd_score[benchmark].mean()
                                                                ))        