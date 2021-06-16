import eval

mode = "train"
train_years = 2018
val_years = 2019
application = "IFS"
batch_size = 16
num_batches = 64
filters_gen = 256
filters_disc = 256
lr_disc = 4e-5
lr_gen = 2e-5
noise_dim = (10,10,2)

if mode == "train":
    log_path = "/ppdata/lucy-cGAN/logs/IFS/filters_256/noise_2"
    rank_samples = 100
elif mode == "deterministic":
    log_path = "/ppdata/lucy-cGAN/logs/IFS/filters_128/softplus/det/lr_1e-4/"
    rank_samples = 1

out_fn = "{}/eval-{}.txt".format(log_path, application)                                                                                                               
weights_dir = log_path


eval.rank_metrics_by_time(mode, train_years, val_years, application, out_fn, weights_dir, check_every=1, 
                          N_range=None, batch_size=batch_size, num_batches=num_batches, 
                          filters_gen=filters_gen, filters_disc=filters_disc, 
                          noise_dim=noise_dim, rank_samples=rank_samples, 
                          lr_disc=lr_disc, lr_gen=lr_gen)

#log_path = "/ppdata/lucy-cGAN/jupyter"
#weights_fn="gen_weights-ERA-0012800.h5"

## note -- rank_metrics_by_noise will only run with GAN weights
#eval.rank_metrics_by_noise(weights_fn, mode, train_years, val_years, application, weights_dir, batch_size=batch_size, num_batches=num_batches, filters=filters, lr_disc=lr_disc, lr_gen=lr_gen)

#log_path = "/ppdata/lucy-cGAN/jupyter
#filename = "gen_weights-ERA-0012800.h5"
#weights_fn = "{}/{}".format(log_path, filename)
#eval.rank_metrics_table(weights_fn, mode, train_years, val_years, application, batch_size=batch_size, num_batches=num_batches, filters=filters, lr_disc=lr_disc, lr_gen=lr_gen, lr=lr)                        
