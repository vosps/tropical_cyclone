import eval

#mode = "train"                                                                                                                                                      
mode = "deterministic"
#train_years = 2015                                                                                                                                                  
train_years = [2015, 2016, 2017, 2018]
val_years = 2019
application = "ERA"
batch_size = 16
num_batches = 64
#filters = 64
filters = 128

if mode == "train":
    log_path = "/ppdata/lucy-cGAN/logs/test-gan"
    rank_samples = 100
elif mode == "deterministic":
    log_path = "/ppdata/lucy-cGAN/logs/filters_128/det"
    rank_samples = 1

#out_fn = "{}/qual-{}.txt".format(log_path, application)
out_fn = "{}/eval-{}.txt".format(log_path, application)                                                                                                               
weights_dir = log_path


eval.rank_metrics_by_time(mode, train_years, val_years, application, out_fn, weights_dir, check_every=1, 
                          N_range=None, batch_size=batch_size, num_batches=num_batches, filters=filters, rank_samples=rank_samples)
