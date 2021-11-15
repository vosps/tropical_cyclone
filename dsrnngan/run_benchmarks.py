import argparse
import os
import gc
import time
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
print(benchmark_methods)

pooling_methods = ['no_pooling']
if args.max_pooling:
    pooling_methods.append('max_4')
    pooling_methods.append('max_16')
if args.avg_pooling:
    pooling_methods.append('avg_4')
    pooling_methods.append('avg_16')
print(pooling_methods)

log_line(log_fname, "Number of samples {}".format(num_batches))
log_line(log_fname, "Evaluation year {}".format(predict_year))
log_line(log_fname, "Model CRPS CRPS_max_4 CRPS_max_16 CRPS_avg_4 CRPS_avg_16 RMSE MAE RAPSD")    


sample_crps = {}
crps_scores = {}
rmse_scores = {}
mae_scores = {}
rapsd_scores = {}

for benchmark in benchmark_methods:
    crps_scores[benchmark] = {}
    rmse_scores[benchmark] = []
    mae_scores[benchmark] = []
    rapsd_scores[benchmark] = []
    print(f"calculating for benchmark method {benchmark}")
    data_benchmarks_iter = iter(data_benchmarks)
    for i in range(num_batches):
        print(f"calculating for sample number {i+1} of {num_batches}")
        (inp,outp) = next(data_benchmarks_iter)
        sample_truth = outp['output']
        if benchmark == 'lanczos':
            sample_benchmark = benchmarks.lanczosmodel(inp['lo_res_inputs'][...,1])
        elif benchmark == 'rainfarm':
            sample_benchmark = benchmarks.rainfarmensemble(inp['lo_res_inputs'][...,1])
        elif benchmark == 'ecpoint':
            sample_benchmark = benchmarks.ecpointPDFmodel(inp['lo_res_inputs'])
        elif benchmark == 'ecpoint_mean':
            sample_benchmark = np.mean(benchmarks.ecpointPDFmodel(inp['lo_res_inputs']),axis=-1)
        elif benchmark == 'constant':
            sample_benchmark = benchmarks.constantupscalemodel(inp['lo_res_inputs'][...,1])
        elif benchmark == 'zeros':
            sample_benchmark = benchmarks.zerosmodel(inp['lo_res_inputs'][...,1])

        if benchmark in ['rainfarm', 'ecpoint']:
            for method in pooling_methods:
                if method == 'no_pooling':
                    sample_truth_pooled = sample_truth
                    sample_benchmark_pooled = sample_benchmark
                if method == 'max_4':
                    max_pool_2d_4 = MaxPooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')
                    sample_truth_pooled = max_pool_2d_4(np.expand_dims(sample_truth.astype("float32"), axis=-1)).numpy()
                    sample_benchmark_pooled = max_pool_2d_4(sample_benchmark.astype("float32")).numpy()
                if method == 'max_16':
                    max_pool_2d_16 = MaxPooling2D(pool_size=(16, 16), strides=(1, 1), padding='valid')
                    sample_truth_pooled = max_pool_2d_16(np.expand_dims(sample_truth.astype("float32"), axis=-1)).numpy()
                    sample_benchmark_pooled = max_pool_2d_16(sample_benchmark.astype("float32")).numpy()
                if method == 'avg_4':
                    avg_pool_2d_4 = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')
                    sample_truth_pooled = avg_pool_2d_4(np.expand_dims(sample_truth.astype("float32"), axis=-1)).numpy()
                    sample_benchmark_pooled = avg_pool_2d_4(sample_benchmark.astype("float32")).numpy()
                if method == 'avg_16':
                    avg_pool_2d_16 = AveragePooling2D(pool_size=(16, 16), strides=(1, 1), padding='valid')
                    sample_truth_pooled = avg_pool_2d_16(np.expand_dims(sample_truth.astype("float32"), axis=-1)).numpy()
                    sample_benchmark_pooled = avg_pool_2d_16(sample_benchmark.astype("float32")).numpy()
                if method != 'no_pooling':
                    sample_truth_pooled = np.squeeze(sample_truth_pooled)
                    sample_benchmark_pooled = np.squeeze(sample_benchmark_pooled)
                crps_score = (crps.crps_ensemble(sample_truth_pooled, sample_benchmark_pooled)).mean()
                del sample_truth_pooled, sample_benchmark_pooled
                if method not in crps_scores[benchmark].keys():
                    crps_scores[benchmark][method] = []
                crps_scores[benchmark][method].append(crps_score)     
            for j in range(sample_benchmark.shape[-1]):
                rmse_tmp = np.sqrt(((sample_truth - sample_benchmark[...,j])**2).mean(axis=(1,2)))
                mae_tmp = (np.abs(sample_truth - sample_benchmark[...,j])).mean(axis=(1,2))
                rapsd_tmp = rapsd_batch(sample_truth, sample_benchmark[...,j])
            rmse_score = rmse_tmp.mean()
            mae_score = mae_tmp.mean()
            rapsd_score = rapsd_tmp.mean()
            print(f"rapsd_score is {rapsd_score}")
            del rmse_tmp, mae_tmp, rapsd_tmp                      
            rmse_scores[benchmark].append(rmse_score)
            mae_scores[benchmark].append(mae_score)
            rapsd_scores[benchmark].append(rapsd_score)
            gc.collect()
        else: 
            for method in pooling_methods:
                if method == 'no_pooling':
                    sample_truth_pooled = sample_truth
                    sample_benchmark_pooled = sample_benchmark
                if method == 'max_4':
                    max_pool_2d_4 = MaxPooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')
                    sample_truth_pooled = max_pool_2d_4(np.expand_dims(sample_truth.astype("float32"), axis=-1)).numpy()
                    sample_benchmark_pooled = max_pool_2d_4(np.expand_dims(sample_benchmark.astype("float32"), axis=-1)).numpy()
                if method == 'max_16':
                    max_pool_2d_16 = MaxPooling2D(pool_size=(16, 16), strides=(1, 1), padding='valid')
                    sample_truth_pooled = max_pool_2d_16(np.expand_dims(sample_truth.astype("float32"), axis=-1)).numpy()
                    sample_benchmark_pooled = max_pool_2d_16(np.expand_dims(sample_benchmark.astype("float32"), axis=-1)).numpy()
                if method == 'avg_4':
                    avg_pool_2d_4 = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')
                    sample_truth_pooled = avg_pool_2d_4(np.expand_dims(sample_truth.astype("float32"), axis=-1)).numpy()
                    sample_benchmark_pooled = avg_pool_2d_4(np.expand_dims(sample_benchmark.astype("float32"), axis=-1)).numpy()
                if method == 'avg_16':
                    avg_pool_2d_16 = AveragePooling2D(pool_size=(16, 16), strides=(1, 1), padding='valid')
                    sample_truth_pooled = avg_pool_2d_16(np.expand_dims(sample_truth.astype("float32"), axis=-1)).numpy()
                    sample_benchmark_pooled = avg_pool_2d_16(np.expand_dims(sample_benchmark.astype("float32"), axis=-1)).numpy()
                if method != 'no_pooling':
                    sample_truth_pooled = np.squeeze(sample_truth_pooled)
                    sample_benchmark_pooled = np.squeeze(sample_benchmark_pooled)
                crps_score = (benchmarks.mean_crps(sample_truth_pooled, sample_benchmark_pooled))
                del sample_truth_pooled, sample_benchmark_pooled
                if method not in crps_scores[benchmark].keys():
                    crps_scores[benchmark][method] = []
                crps_scores[benchmark][method].append(crps_score) 
                gc.collect()
            rmse_score = (np.sqrt(((sample_truth - sample_benchmark)**2)).mean(axis=(1,2)))
            mae_score = (np.abs(sample_truth - sample_benchmark)).mean(axis=(1,2))
            if benchmark == 'zeros':
                rapsd_score = np.nan
            else:
                rapsd_score = rapsd_batch(sample_truth, sample_benchmark)    
            print(f"rapsd_score is {rapsd_score}")
            rmse_scores[benchmark].append(rmse_score)
            mae_scores[benchmark].append(mae_score)
            rapsd_scores[benchmark].append(rapsd_score)
            gc.collect()

for benchmark in benchmark_methods:
    if not args.max_pooling:
        crps_scores[benchmark]['max_4'] = np.nan
        crps_scores[benchmark]['max_16'] = np.nan
    if not args.avg_pooling:
        crps_scores[benchmark]['avg_4'] = np.nan
        crps_scores[benchmark]['avg_16'] = np.nan
    print(f"benchmark {benchmark}")
    print(f"RAPSD scores {rapsd_scores[benchmark]}")
    log_line(log_fname, "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                                                                benchmark,
                                                                np.array(crps_scores[benchmark]['no_pooling']).mean(),
                                                                np.array(crps_scores[benchmark]['max_4']).mean(),
                                                                np.array(crps_scores[benchmark]['max_16']).mean(),
                                                                np.array(crps_scores[benchmark]['avg_4']).mean(),
                                                                np.array(crps_scores[benchmark]['avg_16']).mean(),
                                                                np.array(rmse_scores[benchmark]).mean(),
                                                                np.array(mae_scores[benchmark]).mean(),
                                                                np.nanmean(np.array(rapsd_scores[benchmark]))
                                                                ))        
