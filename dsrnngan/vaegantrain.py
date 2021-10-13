import tensorflow as tf
import numpy as np
from tensorflow import keras
from wloss import wasserstein_loss


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        raise RuntimeError("This should not be getting called; VAE is trained via VAE_trainer class")

    def predict(self, *args):
        raise RuntimeError("Do not call predict directly; call encoder and decoder separately")


class VAE_trainer(keras.Model):
    def __init__(self, VAE, disc, kl_weight, **kwargs):
        super(VAE_trainer, self).__init__(**kwargs)
        self.VAE = VAE
        self.disc = disc
        self.kl_weight = kl_weight
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.vaegen_loss_tracker = keras.metrics.Mean(name="vaegen_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.vaegen_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        [cond, const, noise], gen_target = data
        batch_size = cond.shape[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.VAE.encoder([cond, const])
            pred = self.VAE.decoder([z_mean, z_log_var, noise, const])
            # apply disc to decoder predictions
            y_pred = self.disc([cond, const, pred])
            # target vector of ones used for wasserstein loss
            vaegen_loss = wasserstein_loss(gen_target, y_pred)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # "flatten" kl_loss to batch_size x n_latent_vars
            data_shape = kl_loss.get_shape().as_list()
            temp_dim = tf.reduce_prod(data_shape[1:])
            kl_loss = tf.reshape(kl_loss, [-1, temp_dim])
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # calculate weighted compound loss
            # total_loss = float("NaN")
            total_loss = vaegen_loss + kl_loss*tf.constant(self.kl_weight)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.vaegen_loss_tracker.update_state(vaegen_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # tf.print(self.total_loss_tracker.result(), self.vaegen_loss_tracker.result(), self.kl_loss_tracker.result())
        return [self.total_loss_tracker.result(), self.vaegen_loss_tracker.result(), self.kl_loss_tracker.result()]
#         return {
#             "loss": self.total_loss_tracker.result(),
#             "vaegen loss": self.vaegen_loss_tracker.result(),
#             "kl_loss": self.kl_loss_tracker.result(),
#         }

    def predict(self, *args):
        raise RuntimeError("Should not be calling .predict on VAE_trainer")