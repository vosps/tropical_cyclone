import gc
import os
import gan
import deterministic
import tfrecords_generator_ifs
from tfrecords_generator_ifs import DataGenerator
from vaegantrain import VAE
from tensorflow.keras.optimizers import Adam
import models
import noise
import plots


path = os.path.dirname(os.path.abspath(__file__))


def setup_batch_gen(train_years,
                    val_years,
                    batch_size=64,
                    val_size=None,
                    downsample=False,
                    weights=None,
                    val_fixed=True):

    tfrecords_generator_ifs.return_dic = False
    print(f"downsample flag is {downsample}")
    if train_years is not None:
        train = DataGenerator(train_years, batch_size=batch_size, downsample=downsample, weights=weights)
    else:
        train = None
    # note -- using create_fixed_dataset with a batch size not divisible by 16 will cause problems
    # create_fixed_dataset will not take a list
    if val_size is not None:
        if val_size <= batch_size:
            # val = DataGenerator(val_years, batch_size=val_size, repeat=False, downsample=downsample)
            val = tfrecords_generator_ifs.create_fixed_dataset(val_years, batch_size=val_size, downsample=downsample)
            val = val.take(1)
        else:
            # val = DataGenerator(val_years, batch_size=batch_size, repeat=False, downsample=downsample)
            val = tfrecords_generator_ifs.create_fixed_dataset(val_years, batch_size=batch_size, downsample=downsample)
            val = val.take(val_size//batch_size)
        if val_fixed:
            val = val.cache()
    else:
        # val = DataGenerator(val_years, batch_size=batch_size, downsample=downsample)
        val = tfrecords_generator_ifs.create_fixed_dataset(val_years, batch_size=batch_size, downsample=downsample)
    return train, val, None


def setup_full_image_dataset(years,
                             batch_size=1,
                             downsample=False):

    from data_generator_ifs import DataGenerator as DataGeneratorFull
    from data import get_dates

    all_ifs_fields = ['tp', 'cp', 'sp', 'tisr', 'cape', 'tclw', 'tcwv', 'u700', 'v700']
    dates = get_dates(years)
    data_full = DataGeneratorFull(dates=dates,
                                  ifs_fields=all_ifs_fields,
                                  batch_size=batch_size,
                                  log_precip=True,
                                  crop=True,
                                  shuffle=False,
                                  constants=True,
                                  hour=0,
                                  ifs_norm=True,
                                  downsample=downsample)
    return data_full


def setup_model(*,
                mode=None,
                arch=None,
                train_years=None,
                val_years=None,
                val_size=None,
                downsample=False,
                weights=None,
                input_channels=None,
                steps_per_epoch=None,
                batch_size=None,
                filters_gen=None,
                filters_disc=None,
                noise_channels=None,
                latent_variables=None,
                kl_weight=None,
                lr_disc=None,
                lr_gen=None):

    if mode in ("GAN", "VAEGAN"):
        gen_to_use = {"normal": models.generator}[arch]
        disc_to_use = {"normal": models.discriminator}[arch]
    elif mode == "det":
        gen_to_use = {"normal": models.generator}[arch]

    if mode == 'GAN':
        gen = gen_to_use(mode=mode,
                         input_channels=input_channels,
                         noise_channels=noise_channels,
                         filters_gen=filters_gen)
        disc = disc_to_use(input_channels=input_channels,
                           filters_disc=filters_disc)
        model = gan.WGANGP(gen, disc, mode, lr_disc=lr_disc, lr_gen=lr_gen)
    elif mode == 'VAEGAN':
        (encoder, decoder) = gen_to_use(mode=mode,
                                        input_channels=input_channels,
                                        latent_variables=latent_variables,
                                        filters_gen=filters_gen)
        disc = disc_to_use(input_channels=input_channels,
                           filters_disc=filters_disc)
        gen = VAE(encoder, decoder)
        model = gan.WGANGP(gen, disc, mode, lr_disc=lr_disc,
                           lr_gen=lr_gen, kl_weight=kl_weight)
    elif mode == 'det':
        gen = gen_to_use(mode=mode,
                         input_channels=input_channels,
                         filters_gen=filters_gen)
        model = deterministic.Deterministic(gen, lr_gen,
                                            loss='mse',
                                            optimizer=Adam)

    if train_years is None and val_years is None:
        print("loading model only")
        gc.collect()
        return model
    else:
        (batch_gen_train, batch_gen_valid, batch_gen_test) = setup_batch_gen(
            train_years=train_years,
            val_years=val_years,
            batch_size=batch_size,
            val_size=val_size,
            downsample=downsample,
            weights=weights)

    gc.collect()

    return (model, batch_gen_train, batch_gen_valid, batch_gen_test, steps_per_epoch)


def setup_data(train_years=None,
               val_years=None,
               val_size=None,
               downsample=False,
               weights=None,
               batch_size=None,
               load_full_image=None):

    if load_full_image is True:
        batch_gen_train = setup_full_image_dataset(train_years,
                                                   batch_size=batch_size,
                                                   downsample=downsample)
        batch_gen_valid = setup_full_image_dataset(val_years,
                                                   batch_size=batch_size,
                                                   downsample=downsample)

    else:
        (batch_gen_train, batch_gen_valid, _) = setup_batch_gen(
            train_years=train_years,
            val_years=val_years,
            batch_size=batch_size,
            val_size=val_size,
            downsample=downsample,
            weights=weights)

    gc.collect()

    return (batch_gen_train, batch_gen_valid)


def train_model(*,
                model=None,
                mode=None,
                batch_gen_train=None,
                batch_gen_valid=None,
                noise_channels=None,
                latent_variables=None,
                epoch=None,
                steps_per_epoch=None,
                plot_samples=8,
                plot_fn=None):

    for cond, _, _ in batch_gen_train.take(1).as_numpy_iterator():
        img_shape = cond.shape[1:-1]
        batch_size = cond.shape[0]
    del cond

    if mode == 'GAN':
        noise_shape = (img_shape[0], img_shape[1], noise_channels)
        noise_in = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
        loss_log = model.train(batch_gen_train, noise_in(),
                               steps_per_epoch, training_ratio=5)

    elif mode == 'VAEGAN':
        noise_shape = (img_shape[0], img_shape[1], latent_variables)
        noise_in = noise.NoiseGenerator(noise_shape, batch_size=batch_size)
        loss_log = model.train(batch_gen_train, noise_in(),
                               steps_per_epoch, training_ratio=5)

    elif mode == 'det':
        loss_log = model.train(batch_gen_train, steps_per_epoch)

    plots.plot_sequences(model.gen,
                         mode,
                         batch_gen_valid,
                         epoch,
                         noise_channels=noise_channels,
                         latent_variables=latent_variables,
                         num_samples=plot_samples,
                         out_fn=plot_fn)

    return loss_log
