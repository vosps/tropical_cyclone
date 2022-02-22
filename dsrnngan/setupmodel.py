import gc
import gan
import deterministic
from vaegantrain import VAE
from tensorflow.keras.optimizers import Adam
import models


def setup_model(*,
                mode=None,
                arch=None,
                input_channels=None,
                filters_gen=None,
                filters_disc=None,
                noise_channels=None,
                latent_variables=None,
                padding=None,
                kl_weight=None,
                ensemble_size=None, 
                content_loss_weight=None,
                lr_disc=None,
                lr_gen=None):
 
    if mode in ("GAN", "VAEGAN"):
        gen_to_use = {"normal": models.generator,
                      "forceconv": models.generator}[arch]
        disc_to_use = {"normal": models.discriminator,
                       "forceconv": models.discriminator}[arch]
    elif mode == "det":
        gen_to_use = {"normal": models.generator,
                      "forceconv": models.generator}[arch]

    if mode == 'GAN':
        print('padding is: ',padding)
        gen = gen_to_use(mode=mode,
                         arch=arch,
                         input_channels=input_channels,
                         noise_channels=noise_channels,
                         filters_gen=filters_gen,
                         padding=padding)
        disc = disc_to_use(arch=arch,
                           input_channels=input_channels,
                           filters_disc=filters_disc,
                           padding=padding)
        model = gan.WGANGP(gen, disc, mode, lr_disc=lr_disc, lr_gen=lr_gen,
                           ensemble_size=ensemble_size, 
                           content_loss_weight=content_loss_weight)
    elif mode == 'VAEGAN':
        (encoder, decoder) = gen_to_use(mode=mode,
                                        arch=arch,
                                        input_channels=input_channels,
                                        latent_variables=latent_variables,
                                        filters_gen=filters_gen,
                                        padding=padding)
        disc = disc_to_use(arch=arch,
                           input_channels=input_channels,
                           filters_disc=filters_disc,
                           padding=padding)
        gen = VAE(encoder, decoder)
        model = gan.WGANGP(gen, disc, mode, lr_disc=lr_disc,
                           lr_gen=lr_gen, kl_weight=kl_weight,
                           ensemble_size=ensemble_size, 
                           content_loss_weight=content_loss_weight)
    elif mode == 'det':
        gen = gen_to_use(mode=mode,
                         arch=arch,
                         input_channels=input_channels,
                         filters_gen=filters_gen,
                         padding=padding)
        model = deterministic.Deterministic(gen, 
                                            lr=lr_gen,
                                            loss='mse',
                                            optimizer=Adam)

    gc.collect()
    return model
