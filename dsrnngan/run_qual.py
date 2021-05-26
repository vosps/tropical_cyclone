import eval

mode = "train"
#mode = "deterministic"
train_years = 2019
val_years = 2018
application = "IFS"
batch_size = 16
num_batches = 64
filters = 128
lr_disc = 4e-6
lr_gen = 2e-6
noise_dim = (10,10,4)

if mode == "train":
    log_path = "/ppdata/lucy-cGAN/logs/IFS/filters_128/softplus/gan/noise_channels_4/g_2e-6_d_4e-6"
elif mode == "deterministic":
    log_path = "/ppdata/lucy-cGAN/logs/IFS/filters_128/softplus/det/lr_1e-4"
    
out_fn = "{}/qual-{}.txt".format(log_path, application)
weights_dir = log_path


eval.quality_metrics_by_time(mode, train_years, val_years, application, out_fn, weights_dir, check_every=1, batch_size=batch_size, num_batches=num_batches, 
                             filters=filters, noise_dim=noise_dim, lr_disc=lr_disc, lr_gen=lr_gen)

#log_path = "/ppdata/lucy-cGAN/jupyter"
#weights_fn="gen_weights-ERA-0012800.h5"

#eval.quality_metrics_table(mode, weights_fn, train_years, val_years, application, batch_size=batch_size, num_batches=num_batches, filters=filters, lr_disc=lr_disc, lr_gen=lr_gen, lr=lr)  
