from tensorflow.python.keras.utils import generic_utils
import os

import numpy as np

import crps
import train
import data
import msssim
from noise import NoiseGenerator
import plots
import rainfarm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

path = os.path.dirname(os.path.abspath(__file__))


def setup_inputs(mode,
                 val_years,
                 downsample,
                 weights,
                 input_channels,
                 batch_size,
                 num_batches,
                 filters_gen,
                 filters_disc,
                 noise_channels,
                 latent_variables,
                 load_full_image):

    if load_full_image:
        # small batch size to prevent memory issues
        batch_size = 1
    else:
        batch_size = batch_size
        num_batches = num_batches

    # initialise model
    model = train.setup_model(mode,
                              train_years=None,
                              val_years=None,
                              val_size=None,
                              downsample=downsample,
                              weights=weights,
                              input_channels=input_channels,
                              batch_size=batch_size,
                              filters_gen=filters_gen,
                              filters_disc=filters_disc,
                              noise_channels=noise_channels,
                              latent_variables=latent_variables)

    gen = model.gen

    if load_full_image:
        print('Loading full sized image dataset')
        # load full size image
        batch_gen_valid = train.setup_full_image_dataset(val_years,
                                                         batch_size=batch_size,
                                                         downsample=downsample)
    else:
        print('Evaluating with smaller image dataset')
        _, _, batch_gen_valid, _, _ = train.setup_model(mode,
                                                        train_years=None,
                                                        val_years=val_years,
                                                        val_size=batch_size*num_batches,
                                                        downsample=downsample,
                                                        weights=weights,
                                                        input_channels=input_channels,
                                                        batch_size=batch_size)
    return (gen, batch_gen_valid)


def randomize_nans(x, rnd_mean, rnd_range):
    nan_mask = np.isnan(x)
    nan_shape = x[nan_mask].shape
    x[nan_mask] = rnd_mean + \
        (np.random.rand(*nan_shape)-0.5)*rnd_range


def ensemble_ranks(mode,
                   gen,
                   batch_gen,
                   noise_channels,
                   latent_variables,
                   batch_size,
                   num_batches,
                   noise_offset=0.0,
                   noise_mul=1.0,
                   denormalise_data=True,
                   add_noise=True,
                   rank_samples=100,
                   noise_factor=None,
                   normalize_ranks=True,
                   load_full_image=False,
                   show_progress=True):

    ranks = []
    crps_scores = []
    batch_gen_iter = iter(batch_gen)

    if mode == "det":
        rank_samples = 1

    if show_progress:
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(
            num_batches)

    for k in range(num_batches):
        if load_full_image:
            inputs, outputs = next(batch_gen_iter)
            cond = inputs['generator_input']
            const = inputs['constants']
            sample = outputs['generator_output']
            sample = np.expand_dims(np.array(sample), axis=-1)
        else:
            cond, const, sample = next(batch_gen_iter)
            sample = sample.numpy()

        if denormalise_data:
            sample = data.denormalise(sample)
        if add_noise:
            noise_dim_1, noise_dim_2 = sample[0, ..., 0].shape
            noise = np.random.rand(batch_size, noise_dim_1, noise_dim_2, 1)*noise_factor
            sample += noise
        sample_crps = sample
        sample = sample.ravel()
        samples_gen = []

        if mode == "GAN":
            for i in range(rank_samples):
                noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
                n = NoiseGenerator(noise_shape, batch_size=batch_size)
                for nn in n:
                    nn *= noise_mul
                    nn -= noise_offset
                sample_gen = gen.predict([cond, const, n])
                if denormalise_data:
                    sample_gen = data.denormalise(sample_gen)
                samples_gen.append(sample_gen)

        elif mode == "det":
            sample_gen = gen.predict([cond, const])
            if denormalise_data:
                sample_gen = data.denormalise(sample_gen)
            samples_gen.append(sample_gen)
        elif mode == 'VAEGAN':
            # call encoder once
            (mean, logvar) = gen.encoder([cond, const])
            for i in range(rank_samples):
                noise_shape = np.array(cond)[0, ..., 0].shape + (latent_variables,)
                n = NoiseGenerator(noise_shape, batch_size=batch_size)
                for nn in n:
                    nn *= noise_mul
                    nn -= noise_offset
                # generate ensemble of preds with decoder
                sample_gen = gen.decoder.predict([mean, logvar, n, const])
                if denormalise_data:
                    sample_gen = data.denormalise(sample_gen)
                if add_noise:
                    (noise_dim_1, noise_dim_2) = sample_gen[0, ..., 0].shape
                    noise = np.random.rand(batch_size, noise_dim_1, noise_dim_2, 1)*noise_factor
                    sample_gen += noise
                samples_gen.append(sample_gen)
        else:
            print("mode type not implemented in ensemble_ranks")

        samples_gen = np.stack(samples_gen, axis=-1)
        crps_score = crps.crps_ensemble(sample_crps, samples_gen)
        crps_scores.append(crps_score.ravel())
        samples_gen = samples_gen.reshape(
            (np.prod(samples_gen.shape[:-1]), samples_gen.shape[-1]))
        rank = np.count_nonzero(sample[:, None] >= samples_gen, axis=-1)
        ranks.append(rank)

        if show_progress:
            crps_mean = np.mean(crps_scores)
            losses = [("CRPS", crps_mean)]
            progbar.add(1, values=losses)

    ranks = np.concatenate(ranks)
    crps_scores = np.concatenate(crps_scores)
    if normalize_ranks:
        ranks = ranks / rank_samples

    return (ranks, crps_scores)


def rank_KS(norm_ranks, num_ranks=100):
    (h, b) = np.histogram(norm_ranks, num_ranks+1)
    h = h / h.sum()
    ch = np.cumsum(h)
    cb = b[1:]
    return abs(ch-cb).max()


def rank_CvM(norm_ranks, num_ranks=100):
    (h, b) = np.histogram(norm_ranks, num_ranks+1)
    h = h / h.sum()
    ch = np.cumsum(h)
    cb = b[1:]
    db = np.diff(b)

    return np.sqrt(((ch-cb)**2*db).sum())


def rank_DKL(norm_ranks, num_ranks=100):
    (h, b) = np.histogram(norm_ranks, num_ranks+1)
    q = h / h.sum()
    p = 1/len(h)
    return p*np.log(p/q).sum()


def rank_OP(norm_ranks, num_ranks=100):
    op = np.count_nonzero(
        (norm_ranks == 0) | (norm_ranks == 1)
    )
    op = float(op)/len(norm_ranks)
    return op


def rank_metrics_by_time(mode,
                         val_years,
                         out_fn,
                         weights_dir,
                         downsample=False,
                         weights=None,
                         add_noise=True,
                         noise_factor=None,
                         load_full_image=False,
                         model_number=None,
                         batch_size=16,
                         num_batches=64,
                         filters_gen=64,
                         filters_disc=64,
                         input_channels=9,
                         latent_variables=1,
                         noise_channels=4,
                         rank_samples=100):

    (gen, batch_gen_valid) = setup_inputs(mode=mode,
                                          val_years=val_years,
                                          downsample=downsample,
                                          weights=weights,
                                          input_channels=input_channels,
                                          batch_size=batch_size,
                                          num_batches=num_batches,
                                          filters_gen=filters_gen,
                                          filters_disc=filters_disc,
                                          noise_channels=noise_channels,
                                          latent_variables=latent_variables,
                                          load_full_image=load_full_image)

    if model_number is not None:
        out_fn += model_number

    def log_line(line):
        with open(out_fn, 'a') as f:
            print(line, file=f)
    log_line("N KS CvM DKL OP CRPS mean std")

    if model_number is None:
        files = os.listdir(weights_dir)

        def get_id(fn):
            return fn.split("-")[1]
        files = sorted(fn for fn in files if get_id(fn) == application)

        for fn in files[::check_every]:
            N_samples = int(fn.split("-")[-1].split(".")[0])
            if (N_range is not None) and not (N_range[0] <= N_samples < N_range[1]):
                continue
            print(weights_dir + "/" + fn)

            gen.load_weights(weights_dir + "/" + fn)
            ranks, crps_scores = ensemble_ranks(mode,
                                                gen,
                                                batch_gen_valid,
                                                noise_channels=noise_channels,
                                                latent_variables=latent_variables,
                                                batch_size=batch_size,
                                                num_batches=num_batches,
                                                add_noise=add_noise,
                                                rank_samples=rank_samples,
                                                noise_factor=noise_factor,
                                                load_full_image=load_full_image)
            KS = rank_KS(ranks)
            CvM = rank_CvM(ranks)
            DKL = rank_DKL(ranks)
            OP = rank_OP(ranks)
            CRPS = crps_scores.mean()
            mean = ranks.mean()
            std = ranks.std()

            log_line("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(N_samples, KS, CvM, DKL, OP, CRPS, mean, std))

            # quasi-random selection to avoid generating loads of data
            ranks_to_save = ['124800', '198400', '240000', '320000']
            if any(num in fn for num in ranks_to_save) and add_noise is False and load_full_image is False:
                np.savez('{}/ranks-{}.npz'.format(weights_dir, N_samples), ranks)
            elif any(num in fn for num in ranks_to_save) and add_noise is True and load_full_image is False:
                np.savez('{}/ranks-noise-{}.npz'.format(weights_dir, N_samples), ranks)
            elif any(num in fn for num in ranks_to_save) and add_noise is False and load_full_image is True:
                np.savez('{}/ranks-full_image-{}.npz'.format(weights_dir, N_samples), ranks)
            elif any(num in fn for num in ranks_to_save) and add_noise is True and load_full_image is True:
                np.savez('{}/ranks-full_image-noise-{}.npz'.format(weights_dir, N_samples), ranks)
    else:
        print(f"loading model number {model_number}")
        print(weights_dir + "/gen_weights-" + application + "-" + model_number + ".h5")
        gen.load_weights(weights_dir + "/gen_weights-" + application + "-" + model_number + ".h5")
        ranks, crps_scores = ensemble_ranks(mode,
                                            gen,
                                            batch_gen_valid,
                                            noise_channels=noise_channels,
                                            latent_variables=latent_variables,
                                            batch_size=batch_size,
                                            num_batches=num_batches,
                                            add_noise=add_noise,
                                            rank_samples=rank_samples,
                                            noise_factor=noise_factor,
                                            load_full_image=load_full_image)

        KS = rank_KS(ranks)
        CvM = rank_CvM(ranks)
        DKL = rank_DKL(ranks)
        OP = rank_OP(ranks)
        CRPS = crps_scores.mean()
        mean = ranks.mean()
        std = ranks.std()

        log_line("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(model_number, KS, CvM, DKL, OP, CRPS, mean, std))
        if add_noise is False and load_full_image is False:
            np.savez('{}/ranks-{}.npz'.format(weights_dir, model_number), ranks)
        elif add_noise is True and load_full_image is False:
            np.savez('{}/ranks-noise-{}.npz'.format(weights_dir, model_number), ranks)
        elif add_noise is False and load_full_image is True:
            np.savez('{}/ranks-full_image-{}.npz'.format(weights_dir, model_number), ranks)
        elif add_noise is True and load_full_image is True:
            np.savez('{}/ranks-full_image-noise-{}.npz'.format(weights_dir, model_number), ranks)


def rank_metrics_by_noise(filename,
                          mode,
                          train_years,
                          val_years,
                          application,
                          weights_dir,
                          downsample=False,
                          weights=None,
                          add_noise=True,
                          noise_factor=None,
                          batch_size=None,
                          num_batches=None,
                          filters_gen=None,
                          filters_disc=None,
                          input_channels=None,
                          noise_channels=None,
                          latent_variables=None,
                          load_full_image=None):

    (gen, batch_gen_valid) = setup_inputs(mode=mode,
                                          val_years=val_years,
                                          downsample=downsample,
                                          weights=weights,
                                          input_channels=input_channels,
                                          batch_size=batch_size,
                                          num_batches=num_batches,
                                          filters_gen=filters_gen,
                                          filters_disc=filters_disc,
                                          noise_channels=noise_channels,
                                          latent_variables=latent_variables,
                                          load_full_image=load_full_image)

    noise_mu_values = list([round(x * 0.01, 1) for x in range(50, 250, 10)])+[3.0, 3.5]

    for m in noise_mu_values:
        epoch = 1
        print("Run {}/{}".format(epoch, len(noise_mu_values)))
        N_samples = int(filename.split("-")[-1].split(".")[0])
        gen.load_weights(weights_dir + "/" + filename)
        (ranks, crps_scores) = ensemble_ranks(mode,
                                              gen,
                                              batch_gen_valid,
                                              noise_mul=m,
                                              noise_channels=noise_channels,
                                              latent_variables=latent_variables,
                                              batch_size=batch_size,
                                              num_batches=num_batches,
                                              add_noise=add_noise,
                                              noise_factor=noise_factor,
                                              load_full_image=load_full_image)

        KS = rank_KS(ranks)
        CvM = rank_CvM(ranks)
        DKL = rank_DKL(ranks)
        CRPS = crps_scores.mean()
        mean = ranks.mean()
        std = ranks.std()

        print(N_samples, KS, CvM, DKL, CRPS, mean, std)
        epoch += 1


def rank_metrics_table(weights_fn,
                       mode,
                       val_years,
                       downsample=False,
                       weights=None,
                       add_noise=True,
                       noise_factor=None,
                       batch_size=None,
                       num_batches=None,
                       filters_gen=None,
                       filters_disc=None,
                       input_channels=None,
                       noise_channels=None,
                       latent_variables=None,
                       load_full_image=None):

    train_years = None
    if mode in ["GAN", "det", "VAEGAN"]:
        (gen, batch_gen_valid) = setup_inputs(mode=mode,
                                              val_years=val_years,
                                              downsample=downsample,
                                              weights=weights,
                                              input_channels=input_channels,
                                              batch_size=batch_size,
                                              num_batches=num_batches,
                                              filters_gen=filters_gen,
                                              filters_disc=filters_disc,
                                              noise_channels=noise_channels,
                                              latent_variables=latent_variables,
                                              load_full_image=load_full_image)
        gen.load_weights(weights_fn)
        ranks, crps_scores = ensemble_ranks(mode,
                                            gen,
                                            batch_gen_valid,
                                            noise_channels=noise_channels,
                                            latent_variables=latent_variables,
                                            num_batches=num_batches,
                                            add_noise=add_noise,
                                            noise_factor=noise_factor)

    elif mode in ['rainfarm', 'lanczos', 'constant']:
        (_, batch_gen_valid) = train.setup_data(train_years,
                                                val_years,
                                                val_size=batch_size*num_batches,
                                                downsample=downsample,
                                                weights=weights,
                                                batch_size=batch_size,
                                                load_full_image=load_full_image)
        if mode == "rainfarm":
            gen = GeneratorRainFARM(10, data.denormalise)
            ranks, crps_scores = ensemble_ranks("GAN", gen,
                                                batch_gen_valid,
                                                num_batches=num_batches)
        if mode == "lanczos":
            gen = GeneratorLanczos((100, 100))
            ranks, crps_scores = ensemble_ranks("det", gen,
                                                batch_gen_valid,
                                                num_batches=num_batches)
        if mode == "constant":
            gen = GeneratorConstantUp(10)
            ranks, crps_scores = ensemble_ranks("det", gen,
                                                batch_gen_valid,
                                                noise_channels,
                                                num_batches=num_batches)
    else:
        print("rank_metrics_table not implemented for mode type")

    KS = rank_KS(ranks)
    CvM = rank_CvM(ranks)
    DKL = rank_DKL(ranks)
    OP = rank_OP(ranks)
    CRPS = crps_scores.mean()
    mean = ranks.mean()
    std = ranks.std()

    print("KS: {:.3f}".format(KS))
    print("CvM: {:.3f}".format(CvM))
    print("DKL: {:.3f}".format(DKL))
    print("OP: {:.3f}".format(OP))
    print("CRPS: {:.3f}".format(CRPS))
    print("mean: {:.3f}".format(mean))
    print("std: {:.3f}".format(std))


def log_spectral_distance(img1, img2):
    def power_spectrum_dB(img):
        fx = np.fft.fft2(img)
        fx = fx[:img.shape[0]//2, :img.shape[1]//2]
        px = abs(fx)**2
        return 10 * np.log10(px)

    d = (power_spectrum_dB(img1)-power_spectrum_dB(img2))**2

    d[~np.isfinite(d)] = np.nan
    return np.sqrt(np.nanmean(d))


def log_spectral_distance_batch(batch1, batch2):
    lsd_batch = []
    for i in range(batch1.shape[0]):
        lsd = log_spectral_distance(
                batch1[i, :, :], batch2[i, :, :]
            )
        lsd_batch.append(lsd)
    return np.array(lsd_batch)


def image_quality(mode,
                  gen,
                  batch_gen,
                  noise_channels,
                  latent_variables,
                  batch_size,
                  num_instances=1,
                  num_batches=100,
                  load_full_image=False,
                  denormalise_data=True,
                  show_progress=True):

    batch_gen_iter = iter(batch_gen)

    mae_all = []
    rmse_all = []
    ssim_all = []
    lsd_all = []

    if show_progress:
        # Initialize progbar and batch counter
        progbar = generic_utils.Progbar(num_batches)

    for k in range(num_batches):
        if load_full_image:
            (inputs, outputs) = next(batch_gen_iter)
            cond = inputs['generator_input']
            const = inputs['constants']
            sample = outputs['generator_output']
            sample = np.expand_dims(np.array(sample), axis=-1)
        else:
            (cond, const, sample) = next(batch_gen_iter)
            sample = sample.numpy()

        if denormalise_data:
            sample = data.denormalise(sample)

        if mode == 'VAEGAN':
            # call encoder once
            (mean, logvar) = gen.encoder([cond, const])

        for i in range(num_instances):
            if mode == "GAN":
                noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
                n = NoiseGenerator(noise_shape, batch_size=batch_size)
                img_gen = gen.predict([cond, const, n])
            elif mode == "det":
                img_gen = gen.predict([cond, const])
            elif mode == 'VAEGAN':
                noise_shape = np.array(cond)[0, ..., 0].shape + (noise_channels,)
                n = NoiseGenerator(noise_shape, batch_size=batch_size)
                img_gen = gen.decoder.predict([cond, const, n])
            else:
                try:
                    img_gen = gen.predict([cond, const])
                except:
                    assert False, 'image quality metrics not implemented for mode type'

            if denormalise_data:
                img_gen = data.denormalise(img_gen)

            mae = ((np.abs(sample - img_gen)).mean(axis=(1, 2)))
            rmse = np.sqrt(((sample - img_gen)**2).mean(axis=(1, 2)))
            ssim = msssim.MultiScaleSSIM(sample, img_gen, 1.0)
            lsd = log_spectral_distance_batch(sample, img_gen)
            mae_all.append(mae.flatten())
            rmse_all.append(rmse.flatten())
            ssim_all.append(ssim.flatten())
            lsd_all.append(lsd.flatten())

            if show_progress:
                rmse_mean = np.mean(rmse)
                losses = [("RMSE", rmse_mean)]
                progbar.add(1, values=losses)

    mae_all = np.concatenate(mae_all)
    rmse_all = np.concatenate(rmse_all)
    ssim_all = np.concatenate(ssim_all)
    lsd_all = np.concatenate(lsd_all)

    return (mae_all, rmse_all, ssim_all, lsd_all)


def quality_metrics_by_time(mode,
                            val_years,
                            out_fn,
                            weights_dir,
                            downsample=False,
                            weights=None,
                            load_full_image=False,
                            model_numbers=None,
                            batch_size=None,
                            num_batches=None,
                            filters_gen=None,
                            filters_disc=None,
                            input_channels=None,
                            latent_variables=None,
                            noise_channels=None):

    (gen, batch_gen_valid) = setup_inputs(mode=mode,
                                          val_yyears=val_years,
                                          downsample=downsample,
                                          weights=weights,
                                          input_channels=input_channels,
                                          batch_size=batch_size,
                                          num_batches=num_batches,
                                          filters_gen=filters_gen,
                                          filters_disc=filters_disc,
                                          noise_channels=noise_channels,
                                          latent_variables=latent_variables,
                                          load_full_image=load_full_image)

    files = os.listdir(weights_dir)

    def get_app(fn):
        return fn.split("-")[1]
    files = sorted(fn for fn in files if get_app(fn) == application)

    def log_line(line):
        with open(out_fn, 'a') as f:
            print(line, file=f)
    log_line("N RMSE MSSSIM LSD MAE")

    for fn in files[::check_every]:
        N_samples = int(fn.split("-")[-1].split(".")[0])
        print(N_samples)
        gen.load_weights(weights_dir+"/"+fn)

        mae, rmse, ssim, lsd = image_quality(mode,
                                             gen,
                                             batch_gen_valid,
                                             noise_channels=noise_channels,
                                             latent_variables=latent_variables,
                                             batch_size=batch_size,
                                             num_instances=1,
                                             num_batches=num_batches,
                                             load_full_image=load_full_image)

        log_line("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(N_samples, rmse.mean(), ssim.mean(), np.nanmean(lsd), mae.mean()))


def quality_metrics_table(weights_fn,
                          mode,
                          val_years,
                          downsample=False,
                          weights=None,
                          batch_size=None,
                          num_batches=None,
                          filters_gen=None,
                          filters_disc=None,
                          input_channels=None,
                          latent_variables=None,
                          noise_channels=None,
                          load_full_image=False):

    train_years = None
    if mode in ["GAN", "det", "VAEGAN"]:
        (gen, batch_gen_valid) = setup_inputs(mode=mode,
                                              val_years=val_years,
                                              downsample=downsample,
                                              weights=weights,
                                              input_channels=input_channels,
                                              batch_size=batch_size,
                                              num_batches=num_batches,
                                              filters_gen=filters_gen,
                                              filters_disc=filters_disc,
                                              noise_channels=noise_channels,
                                              latent_variables=latent_variables,
                                              load_full_image=load_full_image)
        gen.load_weights(weights_fn)

    elif mode in ['rainfarm', 'lanczos', 'constant']:
        (_, batch_gen_valid) = train.setup_data(train_years,
                                                val_years,
                                                val_size=batch_size*num_batches,
                                                downsample=downsample,
                                                weights=weights,
                                                batch_size=batch_size,
                                                load_full_image=load_full_image)
        if mode == "rainfarm":
            gen = GeneratorRainFARM(10, data.denormalise)

        if mode == "lanczos":
            gen = GeneratorLanczos((100, 100))

        if mode == "constant":
            gen = GeneratorConstantUp(10)
    else:
        print("quality_metrics_table not implemented for mode type")

    mae, rmse, ssim, lsd = image_quality(mode, gen,
                                         batch_gen_valid,
                                         noise_channels=noise_channels,
                                         latent_variables=latent_variables,
                                         batch_size=batch_size,
                                         num_instances=1,
                                         num_batches=num_batches,
                                         load_full_image=load_full_image)

    print("MAE: {:.3f}".format(mae.mean()))
    print("RMSE: {:.3f}".format(rmse.mean()))
    print("MSSSIM: {:.3f}".format(ssim.mean()))
    print("LSD: {:.3f}".format(np.nanmean(lsd)))


class GeneratorLanczos:
    # class that can be used in place of a generator for evaluation purposes,
    # using Lanczos filtering
    def __init__(self, out_size):
        self.out_size = out_size

    def predict(self, *args):
        y = np.array(args[0][0][..., :1])
        out_shape = y.shape[:1] + self.out_size + y.shape[3:]
        x = np.zeros(out_shape, dtype=y.dtype)
        for i in range(x.shape[0]):
            x[i, :, :, 0] = plots.resize_lanczos(y[i, :, :, 0],
                                                 self.out_size)
        return x


class GeneratorConstantUp:
    # class that can be used in place of a generator for evaluation purposes,
    # using constant upsampling
    def __init__(self, out_size):
        self.out_size = out_size

    def predict(self, *args):
        y = args[0][0][..., :1]
        return np.repeat(np.repeat(y, self.out_size, axis=1), self.out_size, axis=2)


class GeneratorDeterministicPlaceholder:
    def __init__(self, gen_det):
        self.gen_det = gen_det

    def predict(self, *args):
        y = args[0][:2]
        return self.gen_det.predict(y)


class GeneratorRainFARM:
    def __init__(self, ds_factor, decoder):
        self.ds_factor = ds_factor
        self.decoder = decoder
        self.batch = 0

    def predict(self, *args):
        self.batch += 1
        y = np.array(args[0][0][..., :1])
        P = self.decoder(y)
        # P = 10**y
        P[~np.isfinite(P)] = 0

        out_size = (y.shape[1]*self.ds_factor, y.shape[2]*self.ds_factor)
        out_shape = y.shape[:1] + out_size + y.shape[3:]
        x = np.zeros(out_shape, dtype=y.dtype)
        for i in range(y.shape[0]):
            r = rainfarm.rainfarm_downscale(P[i, ..., 0], threshold=0.,
                                            ds_factor=self.ds_factor)
            log_r = np.log10(1 + r)
            log_r[~np.isfinite(log_r)] = 0.0
            x[i, ..., 0] = log_r

        return x
