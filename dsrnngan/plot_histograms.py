import matplotlib
matplotlib.use("Agg")
import plots

log_path ='/ppdata/lucy-cGAN/logs/IFS/gen_256_disc_512/noise_4/weights_4x'
added_noise = True
full_image = True

if added_noise and not full_image:
    rank_metrics_files_1 = ["{}/ranks-noise-124800.npz".format(log_path), "{}/ranks-noise-198400.npz".format(log_path)]
    rank_metrics_files_2 = ["{}/ranks-noise-240000.npz".format(log_path), "{}/ranks-noise-320000.npz".format(log_path)]
    labels_1 = ['noise-124800', 'noise-198400']
    labels_2 = ['noise-240000', 'noise-320000']
    name_1 = 'noise-early-small_image'
    name_2 = 'noise-late-small_image'
elif not added_noise and not full_image:
    rank_metrics_files_1 = ["{}/ranks-no-noise-124800.npz".format(log_path), "{}/ranks-no-noise-198400.npz".format(log_path)]
    rank_metrics_files_2 = ["{}/ranks-no-noise-240000.npz".format(log_path), "{}/ranks-no-noise-320000.npz".format(log_path)]
    labels_1 = ['no-noise-124800', 'no-noise-198400']
    labels_2 = ['no-noise-240000', 'no-noise-320000']
    name_1 = 'no-noise-early-small_image'
    name_2 = 'no-noise-late-small_image'
if added_noise and full_image:
    rank_metrics_files_1 = ["{}/ranks-full_image-noise-124800.npz".format(log_path), "{}/ranks-full_image-noise-198400.npz".format(log_path)]
    rank_metrics_files_2 = ["{}/ranks-full_image-noise-240000.npz".format(log_path), "{}/ranks-full_image-noise-320000.npz".format(log_path)]
    labels_1 = ['noise-124800', 'noise-198400']
    labels_2 = ['noise-240000', 'noise-320000']
    name_1 = 'noise-early-full_image'
    name_2 = 'noise-late-full_image'
elif not added_noise and full_image:
    rank_metrics_files_1 = ["{}/ranks-full_image-no-noise-124800.npz".format(log_path), "{}/ranks-full_image-no-noise-198400.npz".format(log_path)]
    rank_metrics_files_2 = ["{}/ranks-full_image-no-noise-240000.npz".format(log_path), "{}/ranks-full_image-no-noise-320000.npz".format(log_path)]  
    labels_1 = ['no-noise-124800', 'no-noise-198400']
    labels_2 = ['no-noise-240000', 'no-noise-320000']
    name_1 = 'no-noise-early-full_image'
    name_2 = 'no-noise-late-full_image'

plots.plot_rank_histogram_all(rank_metrics_files_1, labels_1, log_path, name_1)
plots.plot_rank_histogram_all(rank_metrics_files_2, labels_2, log_path, name_2)
