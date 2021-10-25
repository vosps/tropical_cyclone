import argparse
import os
from data import get_dates
from data_generator_ifs import DataGenerator as DataGeneratorFull
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

if args.include_Lanczos:
    crps_lanczos = []
    rmse_lanczos = []
    mae_lanczos = []
    rapsd_lanczos = []
    
if args.include_RainFARM:
    crps_rainfarm = []
    rmse_rainfarm = []
    mae_rainfarm = []
    rapsd_rainfarm = []
    
if args.include_ecPoint:
    crps_ecpoint = []
    rmse_ecpoint = []
    mae_ecpoint = []
    rapsd_ecpoint = []
    
if args.include_ecPoint_mean:
    crps_ecpoint_mean = []
    rmse_ecpoint_mean = []
    mae_ecpoint_mean = []
    rapsd_ecpoint_mean = []
    
if args.include_constant:
    crps_constant = []
    rmse_constant = []
    mae_constant = []
    rapsd_constant = []
    
if args.include_zeros:
    crps_zeros = []
    rmse_zeros = []
    mae_zeros = []
    rapsd_zeros = []

log_line(log_fname, "Number of samples {}".format(num_batches))
log_line(log_fname, "Evaluation year {}".format(predict_year))
log_line(log_fname, "Model CRPS RMSE MAE RAPSD")    

data_benchmarks_iter = iter(data_benchmarks)
for i in range(num_batches):
    print(f" calculating for sample number {i+1} of {num_batches}")
    (inp,outp) = next(data_benchmarks_iter)
    sample_truth = outp['generator_output']

    if args.include_Lanczos:
        sample_lanczos = benchmarks.lanczosmodel(inp['generator_input'][...,1])
        crps_lanczos.append(benchmarks.mean_crps(sample_truth, sample_lanczos))
        rmse_lanczos.append(np.sqrt(((sample_truth - sample_lanczos)**2).mean(axis=(1,2))))
        mae_lanczos.append((np.abs(sample_truth - sample_lanczos)).mean(axis=(1,2)))
        rapsd_lanczos.append(rapsd_batch(sample_truth, sample_lanczos))

    if args.include_RainFARM:
        sample_rainfarm = benchmarks.rainfarmensemble(inp['generator_input'][...,1])
        crps_rainfarm.append(crps.crps_ensemble(sample_truth, sample_rainfarm))
        for j in range(sample_rainfarm.shape[-1]):
            rmse_rainfarm.append(np.sqrt(((sample_truth - sample_rainfarm[...,j])**2).mean(axis=(1,2))))
            mae_rainfarm.append((np.abs(sample_truth - sample_rainfarm[...,j])).mean(axis=(1,2)))
            rapsd_rainfarm.append(rapsd_batch(sample_truth, sample_rainfarm[...,j]))
        
    if args.include_ecPoint:
        sample_ecpoint = benchmarks.ecpointPDFmodel(inp['generator_input'])
        crps_ecpoint.append(crps.crps_ensemble(sample_truth, sample_ecpoint))
        for j in range(sample_ecpoint.shape[-1]):
            rmse_ecpoint.append(np.sqrt(((sample_truth - sample_ecpoint[...,j])**2).mean(axis=(1,2))))
            mae_ecpoint.append((np.abs(sample_truth - sample_ecpoint[...,j])).mean(axis=(1,2)))
            rapsd_ecpoint.append(rapsd_batch(sample_truth, sample_ecpoint[...,j]))
    
    if args.include_ecPoint_mean:
        sample_ecpoint_mean = np.mean(benchmarks.ecpointPDFmodel(inp['generator_input']),axis=-1)
        crps_ecpoint_mean.append(benchmarks.mean_crps(sample_truth, sample_ecpoint_mean))
        rmse_ecpoint_mean.append(np.sqrt(((sample_truth - sample_ecpoint_mean)**2).mean(axis=(1,2))))
        mae_ecpoint_mean.append((np.abs(sample_truth - sample_ecpoint_mean)).mean(axis=(1,2)))
        rapsd_ecpoint_mean.append(rapsd_batch(sample_truth, sample_ecpoint_mean))
    
    if args.include_constant:
        sample_constant = benchmarks.constantupscalemodel(inp['generator_input'][...,1])
        crps_constant.append(benchmarks.mean_crps(sample_truth, sample_constant))
        rmse_constant.append(np.sqrt(((sample_truth - sample_constant)**2).mean(axis=(1,2))))
        mae_constant.append((np.abs(sample_truth - sample_constant)).mean(axis=(1,2)))
        rapsd_constant.append(rapsd_batch(sample_truth, sample_constant))
        
    if args.include_zeros:
        sample_zeros = benchmarks.zerosmodel(inp['generator_input'][...,1])
        crps_zeros.append(benchmarks.mean_crps(sample_truth, sample_zeros))
        rmse_zeros.append(np.sqrt(((sample_truth - sample_zeros)**2).mean(axis=(1,2))))
        mae_zeros.append((np.abs(sample_truth - sample_zeros)).mean(axis=(1, 2)))  
        rapsd_zeros.append(rapsd_batch(sample_truth, sample_zeros))
   
if args.include_Lanczos:        
    log_line(log_fname, "{} {:.6f} {:.6f} {:.6f} {:.6f}".format('Lanczos',
                                                                np.array(crps_lanczos).mean(),
                                                                np.array(rmse_lanczos).mean(),
                                                                np.array(mae_lanczos).mean(),
                                                                np.isfinite(np.array(rapsd_lanczos)).mean()))
if args.include_RainFARM:        
    log_line(log_fname, "{} {:.6f} {:.6f} {:.6f} {:.6f}".format('RainFARM',
                                                                np.array(crps_rainfarm).mean(),
                                                                np.array(rmse_rainfarm).mean(),
                                                                np.array(mae_rainfarm).mean(),
                                                                np.isfinite(np.array(rapsd_rainfarm)).mean()))
if args.include_ecPoint:        
    log_line(log_fname, "{} {:.6f} {:.6f} {:.6f} {:.6f}".format('ecPoint',
                                                                np.array(crps_ecpoint).mean(),
                                                                np.array(rmse_ecpoint).mean(),
                                                                np.array(mae_ecpoint).mean(),
                                                                np.isfinite(np.array(rapsd_ecpoint)).mean()))
if args.include_ecPoint_mean:        
    log_line(log_fname, "{} {:.6f} {:.6f} {:.6f} {:.6f}".format('ecPoint',
                                                                np.array(crps_ecpoint_mean).mean(),
                                                                np.array(rmse_ecpoint_mean).mean(),
                                                                np.array(mae_ecpoint_mean).mean(),
                                                                np.isfinite(np.array(rapsd_ecpoint_mean)).mean()))
if args.include_constant:        
    log_line(log_fname, "{} {:.6f} {:.6f} {:.6f} {:.6f}".format('Constant',
                                                                np.array(crps_constant).mean(),
                                                                np.array(rmse_constant).mean(),
                                                                np.array(mae_constant).mean(),
                                                                np.isfinite(np.array(rapsd_constant)).mean()))
if args.include_zeros:        
    log_line(log_fname, "{} {:.6f} {:.6f} {:.6f} {:.6f}".format('Zeros',
                                                                np.array(crps_zeros).mean(),
                                                                np.array(rmse_zeros).mean(),
                                                                np.array(mae_zeros).mean(),
                                                                np.isfinite(np.array(rapsd_zeros)).mean()))
