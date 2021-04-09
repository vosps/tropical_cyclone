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
import tfrecords_generator
from tfrecords_generator import DataGenerator
import models
import noise
import plots


path = os.path.dirname(os.path.abspath(__file__))

def setup_batch_gen(train_years,val_years,batch_size=64,
                    val_size = None, val_fixed=True):# ,
                    # train_images=5,val_images=5):
    tfrecords_generator.return_dic = False
    train = DataGenerator(train_years,batch_size=batch_size)
    if val_size is not None:
        if val_size <= batch_size:
            val = DataGenerator(val_years,batch_size=val_size,repeat=False)
            val = val.take(1)
        else:
            val = DataGenerator(val_years,batch_size=batch_size,repeat=False)
            val = val.take(val_size//batch_size)
        if val_fixed:
            val = val.cache()
    else:
        val = DataGenerator(val_years,batch_size=batch_size)
    return train,val,None


def setup_gan(train_years=None, val_years=None,
              val_size = None,
              steps_per_epoch=50,
              batch_size=16, filters=64,
              lr_disc=0.0001, lr_gen=0.0001):

    (gen, noise_shapes) = models.generator(filters=filters)
    disc = models.discriminator(filters=filters)
    wgan = gan.WGANGP(gen, disc, lr_disc=lr_disc, lr_gen=lr_gen)

    (batch_gen_train, batch_gen_valid, batch_gen_test) = setup_batch_gen(
        train_years = train_years, val_years = val_years, 
        val_size = val_size,
        batch_size=batch_size
    )

    gc.collect()

    return (wgan, batch_gen_train, batch_gen_valid, batch_gen_test,
        noise_shapes, steps_per_epoch)


def train_gan(wgan, batch_gen_train, batch_gen_valid, noise_shapes,
    steps_per_epoch, num_epochs,
    plot_samples=8, plot_fn="../figures/progress.pdf"):
    
    for _, _, sample in batch_gen_train.take(1).as_numpy_iterator():
        img_shape = sample.shape[1:-1]
        batch_size = sample.shape[0]
    del sample
    noise_gen = noise.NoiseGenerator(noise_shapes(img_shape),
        batch_size=batch_size)
    
    epoch_print = 1
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch_print, num_epochs))
        epoch_print += 1
        loss_log = wgan.train(batch_gen_train, noise_gen,
                              steps_per_epoch, training_ratio=5)
        plots.plot_sequences(wgan.gen, batch_gen_valid, noise_gen, 
            num_samples=plot_samples, out_fn=plot_fn)

    return loss_log


def setup_deterministic(train_years=None, val_years=None,
                        val_size = None, 
                        steps_per_epoch=50,
                        batch_size=64,
                        filters=64,
                        loss='mse', 
                        lr=1e-4, optimizer=Adam):

    gen_det = models.generator_deterministic(filters=filters)
    #gen_det.compile(loss=loss, optimizer=Adam(lr=lr))
    det_model = deterministic.Deterministic(gen_det, lr, loss, optimizer)
    
    (batch_gen_train, batch_gen_valid, batch_gen_test) = setup_batch_gen(
        train_years = train_years, val_years = val_years, 
        val_size = val_size,
        batch_size=batch_size
    )

    gc.collect()

    return (det_model, batch_gen_train, batch_gen_valid, batch_gen_test,
        steps_per_epoch)


def train_deterministic(det_model, batch_gen_train, batch_gen_valid,
                        steps_per_epoch, num_epochs, plot_samples=8, plot_fn="../figures/progress.pdf"):
    
    #callback = EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)


    # gen_det.fit(batch_gen_train, epochs=num_epochs,
    #         steps_per_epoch=steps_per_epoch,
    #         validation_data=batch_gen_valid, validation_steps=32,
    #         callbacks=[callback])
    
    loss_log = det_model.train_det(batch_gen_train, steps_per_epoch)
    plots.plot_sequences_deterministic(det_model.gen_det, batch_gen_valid, num_samples=plot_samples, out_fn=plot_fn)

    return loss_log
