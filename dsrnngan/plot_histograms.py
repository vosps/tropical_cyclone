import matplotlib
matplotlib.use("Agg")
import plots

log_path ='/ppdata/lucy-cGAN/logs/IFS/GAN/weights_natural'
added_noise = True
full_image = True
ranks = [124800, 198400, 240000, 32000]

if added_noise:
    noise_label = "noise"
else: 
    noise_label = "no-noise"
    
if full_image:
    image_label = "full"
else:
    image_label = "small"


rank_metrics_files_1 = ["{}/ranks-{}-{}-{}.npz".format(log_path, image_label, noise_label, ranks[0]), 
                        "{}/ranks-{}-{}-{}.npz".format(log_path, image_label, noise_label, ranks[1])]
rank_metrics_files_2 = ["{}/ranks-{}-{}-{}.npz".format(log_path, image_label, noise_label, ranks[2]), 
                        "{}/ranks-{}-{}-{}.npz".format(log_path, image_label, noise_label, ranks[3])]
labels_1 = ['{}-{}'.format(noise_label, ranks[0]), '{}-{}'.format(noise_label, ranks[1])]
labels_2 = ['{}-{}'.format(noise_label, ranks[2]), '{}-{}'.format(noise_label, ranks[3])]
name_1 = '{}-early-{}'.format(noise_label, image_label)
name_2 = '{}-late-{}'.format(noise_label, image_label)

plots.plot_rank_histogram_all(rank_metrics_files_1, labels_1, log_path, name_1, N_ranks=11)
plots.plot_rank_histogram_all(rank_metrics_files_2, labels_2, log_path, name_2, N_ranks=11)
