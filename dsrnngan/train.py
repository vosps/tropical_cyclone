import gc
import os

import netCDF4
import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import gan
import deterministic
import tfrecords_generator_ifs
from tfrecords_generator_ifs import DataGenerator
import models
import noise
import plots


path = os.path.dirname(os.path.abspath(__file__))

def setup_batch_gen(train_years, 
                    val_years, 
                    batch_size=64, 
                    val_size=None, 
                    downsample=False, 
                    val_fixed=True):
   
    tfrecords_generator_ifs.return_dic = False
    print(f"downsample flag is {downsample}")
    if train_years is not None:
        train = DataGenerator(train_years, batch_size=batch_size, downsample=downsample)
    else:
        train = None
    if val_size is not None:
        if val_size <= batch_size:
            val = DataGenerator(val_years, batch_size=val_size, repeat=False, downsample=downsample)
            val = val.take(1)
        else:
            val = DataGenerator(val_years, batch_size=batch_size, repeat=False, downsample=downsample)
            val = val.take(val_size//batch_size)
        if val_fixed:
            val = val.cache()
    else:
        val = DataGenerator(val_years, batch_size=batch_size, downsample=downsample)
    return train,val,None


def setup_gan(train_years=None, 
              val_years=None, 
              val_size=None, 
              downsample=False, 
              steps_per_epoch=50, 
              batch_size=16, 
              filters_gen=64, 
              filters_disc=64, 
              noise_dim=(10,10,8), 
              lr_disc=0.0001, 
              lr_gen=0.0001):

    print(f"# gen filters is {filters_gen}")
    print(f"# disc filters is {filters_disc}")
    
    (gen, noise_shapes) = models.generator(noise_dim=noise_dim, filters_gen=filters_gen)
    disc = models.discriminator(filters_disc=filters_disc)
    wgan = gan.WGANGP(gen, disc, lr_disc=lr_disc, lr_gen=lr_gen)

    (batch_gen_train, batch_gen_valid, batch_gen_test) = setup_batch_gen(
        train_years=train_years, val_years=val_years, batch_size=batch_size,
        val_size=val_size, downsample=downsample)

    gc.collect()

    return (wgan, batch_gen_train, batch_gen_valid, batch_gen_test,
        noise_shapes, steps_per_epoch)


def setup_data(train_years=None, 
               val_years=None,
               val_size = None, 
               downsample=False,
               steps_per_epoch=50,
               batch_size=16, 
               noise_dim=(10,10,8)):
    
    (batch_gen_train, batch_gen_valid, batch_gen_test) = setup_batch_gen(
        train_years = train_years, val_years = val_years, 
        batch_size=batch_size, val_size=val_size, downsample=downsample)

    gc.collect()

    return (None, batch_gen_train, batch_gen_valid, batch_gen_test,
        noise_dim, steps_per_epoch)


def train_gan(wgan, 
              batch_gen_train, 
              batch_gen_valid, 
              noise_shapes, 
              epoch,
              steps_per_epoch, 
              num_epochs, 
              plot_samples=8, 
              plot_fn="../figures/progress.pdf"):
    
    for _, _, sample in batch_gen_train.take(1).as_numpy_iterator():
        img_shape = sample.shape[1:-1]
        batch_size = sample.shape[0]
    del sample
    noise_gen = noise.NoiseGenerator(noise_shapes(img_shape), batch_size=batch_size)
    loss_log = wgan.train(batch_gen_train, noise_gen,
                              steps_per_epoch, training_ratio=5)
    plots.plot_sequences(wgan.gen, batch_gen_valid, noise_gen, epoch,
            num_samples=plot_samples, out_fn=plot_fn)

    return loss_log


def setup_deterministic(train_years=None, 
                        val_years=None,
                        val_size=None, 
                        downsample=False,
                        steps_per_epoch=50,
                        batch_size=64,
                        filters_gen=64,
                        loss='mse', 
                        lr=1e-4, 
                        optimizer=Adam):

    print(f"downsample flag is {downsample}")
    print(f"learning rate is: {lr}")
    gen_det = models.generator_deterministic(filters_gen=filters_gen)
    det_model = deterministic.Deterministic(gen_det, lr, loss, optimizer)
    
    (batch_gen_train, batch_gen_valid, batch_gen_test) = setup_batch_gen(
        train_years=train_years, val_years=val_years, batch_size=batch_size, 
        val_size=val_size, downsample=downsample)

    gc.collect()

    return (det_model, batch_gen_train, batch_gen_valid, batch_gen_test,
        steps_per_epoch)


def train_deterministic(det_model, 
                        batch_gen_train, 
                        batch_gen_valid, 
                        epoch,
                        steps_per_epoch, 
                        num_epochs, 
                        plot_samples=8, 
                        plot_fn="../figures/progress.pdf"):
    
    loss_log = det_model.train_det(batch_gen_train, steps_per_epoch)
    plots.plot_sequences_deterministic(det_model.gen_det, batch_gen_valid, epoch, num_samples=plot_samples, out_fn=plot_fn)

    return loss_log
