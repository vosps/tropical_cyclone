import gc
import os

import netCDF4
import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import gan
import tfrecords_generator
from tfrecords_generator import DataGenerator
import models
import noise
import plots


path = os.path.dirname(os.path.abspath(__file__))

def setup_batch_gen(train_years,val_years,batch_size=64,
                    val_size = None):# ,
                    # train_images=5,val_images=5):
    tfrecords_generator.return_dic = False
    train = DataGenerator(train_years,batch_size=batch_size)
    if val_size is not None:
        val = DataGenerator(val_years,batch_size=val_size)
        val = val.take(1)
    else:
        val = DataGenerator(val_years,batch_size=batch_size)
    # train_im = train.take(train_images)
    # val_im = val.take(val_images)
    return train,val,None


def setup_gan(train_years=None, val_years=None,
              val_size = None,
              steps_per_epoch=50,
              batch_size=16,
              lr_disc=0.0001, lr_gen=0.0001):

    (gen, noise_shapes) = models.generator()
    # (gen_init, noise_shapes) = models.generator_initialized(
    #     gen)
    disc = models.discriminator()
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

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1,num_epochs))
        loss_log = wgan.train(batch_gen_train, noise_gen,
            steps_per_epoch, training_ratio=5)
        plots.plot_sequences(wgan.gen, batch_gen_valid, noise_gen, 
            num_samples=plot_samples, out_fn=plot_fn)

    return loss_log


def setup_deterministic(train_years=None, val_years=None,
                        val_size = None,
                        steps_per_epoch=50,
                        batch_size=64,
                        loss='mse', 
                        lr=1e-4):

    (gen, _) = models.generator()
    init_model = models.initial_state_model()
    (gen_init, noise_shapes) = models.generator_initialized(
        gen, init_model)
    gen_det = models.generator_deterministic(gen_init)
    gen_det.compile(loss=loss, optimizer=Adam(lr=lr))

    (batch_gen_train, batch_gen_valid, batch_gen_test) = setup_batch_gen(
        train_years = train_years, val_years = val_years,
        val_size = val_size,
        batch_size=batch_size
    )

    gc.collect()

    return (gen_det, batch_gen_train, batch_gen_valid, batch_gen_test,
        steps_per_epoch)


def train_deterministic(gen, batch_gen_train, batch_gen_valid,
    steps_per_epoch, num_epochs):

    callback = EarlyStopping(monitor='val_loss', patience=5,
        restore_best_weights=True)

    gen.fit(batch_gen_train, epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=batch_gen_valid, validation_steps=32,
            callbacks=[callback])
