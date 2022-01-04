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
                         noise_channels=noise_channels, # should this be 1?
                         filters_gen=filters_gen)
        disc = disc_to_use(input_channels=input_channels,
                           filters_disc=filters_disc)
        print('printing variables')
        print('.................................')
        print(input_channels)
        print(gen)
        print(disc)
        print(mode)
        print(lr_disc)
        print(lr_gen)
        print(gen)
        
        model = gan.WGANGP(gen, disc, mode, lr_disc=lr_disc, lr_gen=lr_gen) #TODO: fix error with gen
        # essentially gen is some sort of function with the necessary shapes require to get this to work. 
        # I need to make sure the shapes of gen are correct, i.e. don't include const as the second item
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

    gc.collect()
    return model
