import eval

log_path = "/ppdata/lucy-cGAN/test-det"
train_years = 2015
# train_years = [2015, 2016, 2017, 2018]
val_years = 2019
application = "ERA"
out_fn = "{}/eval-{}.txt".format(log_path, application)
weights_dir = log_path
batch_size = 16
num_batches = 3
mode = "deterministic"

eval.rank_metrics_by_time(mode, train_years, val_years, application, out_fn, weights_dir, check_every=1, N_range=None, batch_size=batch_size, num_batches=num_batches)
