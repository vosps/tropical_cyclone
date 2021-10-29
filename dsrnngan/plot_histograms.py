import matplotlib
matplotlib.use("Agg")
import plots

log_folder ='/ppdata/lucy-cGAN/logs/IFS/GAN/weights_2x'
add_noise = True
full_image = False
ranks = [124800, 198400, 240000, 320000]
N_ranks = 11

plots.plot_histograms(log_folder, ranks, N_ranks, add_noise, full_image)