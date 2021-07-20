import matplotlib
matplotlib.use("Agg")
import eval
import plots

mode = "ensemble"
train_years = [2016, 2017, 2018]
#val_years = [2016, 2017, 2018]
#train_years = 2018
val_years = 2019
application = "IFS"
batch_size = 16
num_batches = 64
filters_gen = 128
filters_disc = 512
lr_disc = 1e-5
lr_gen = 1e-5
noise_dim = (10, 10, 2)

## set up flags
downsample = True
#downsample = False
add_noise = True
#add_noise = False

if mode == "ensemble":
    log_path = "/ppdata/lucy-cGAN/logs/EASY/GAN/noise_2/g128_d512"
    rank_samples = 100
elif mode == "deterministic":
    log_path = "/ppdata/lucy-cGAN/logs/IFS/filters_128/softplus/det/lr_1e-4/"
    rank_samples = 1

## check this is the correct file name!!
out_fn = "{}/eval-{}_noise.txt".format(log_path, application)                                                                                                               
weights_dir = log_path


eval.rank_metrics_by_time(mode, 
                          train_years, 
                          val_years, 
                          application, 
                          out_fn, 
                          weights_dir, 
                          check_every=1, 
                          N_range=None, 
                          downsample=downsample,
                          add_noise=add_noise,
                          batch_size=batch_size, 
                          num_batches=num_batches, 
                          filters_gen=filters_gen, 
                          filters_disc=filters_disc, 
                          noise_dim=noise_dim, 
                          rank_samples=rank_samples, 
                          lr_disc=lr_disc, 
                          lr_gen=lr_gen)

## plot rank histograms
rank_metrics_files_1 = ["{}/ranks-124800.npz".format(log_path), "{}/ranks-198400.npz".format(log_path)]
rank_metrics_files_2 = ["{}/ranks-240000.npz".format(log_path), "{}/ranks-384000.npz".format(log_path)]
labels_1 = ['124800', '198400']
labels_2 = ['240000', '384000']
name_1 = 'noise-early'
name_2 = 'noise-late'

plots.plot_rank_histogram_all(rank_metrics_files_1, labels_1, log_path, name_1)
plots.plot_rank_histogram_all(rank_metrics_files_2, labels_2, log_path, name_2)

#log_path = "/ppdata/lucy-cGAN/jupyter"
#weights_fn="gen_weights-ERA-0012800.h5"

## note -- rank_metrics_by_noise will only run with GAN weights
#eval.rank_metrics_by_noise(weights_fn, mode, train_years, val_years, application, weights_dir, batch_size=batch_size, num_batches=num_batches, filters=filters, lr_disc=lr_disc, lr_gen=lr_gen)

#log_path = "/ppdata/lucy-cGAN/jupyter
#filename = "gen_weights-ERA-0012800.h5"
#weights_fn = "{}/{}".format(log_path, filename)
#eval.rank_metrics_table(weights_fn, mode, train_years, val_years, application, batch_size=batch_size, num_batches=num_batches, filters=filters, lr_disc=lr_disc, lr_gen=lr_gen, lr=lr)                        
