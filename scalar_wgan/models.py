import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, UpSampling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, LeakyReLU
from tensorflow.keras import backend as K
from keras.layers.core import Activation
from keras.utils.generic_utils import get_custom_objects

from blocks import residual_block, const_upscale_block_100, const_upscale_block_5


def generator(mode,
              arch,
              input_channels=8,
              latent_variables=1,
              noise_channels=6,
              filters_gen=64,
              img_shape=(100, 100),
              constant_fields=1, #2
              conv_size=(3, 3),
              padding=None,
              stride=1,
              relu_alpha=0.2,
              norm=None,
              dropout_rate=None):

    forceconv = True if arch == "forceconv" else False
    # Network inputs
    # low resolution condition
    generator_input = Input(shape=(None, None, input_channels), name="lo_res_inputs")
    # generator_input = Input(shape=(None, input_channels), name="lo_res_inputs")
    print(f"generator_input shape: {generator_input.shape}")
    # constant fields
    const_input = Input(shape=(None, None, constant_fields), name="hi_res_inputs")
    # const_input = Input(shape=(None, None, constant_fields), name="test")
    print(f"constants_input shape: {const_input.shape}")

    # Convolve constant fields down to match other input dimensions
    upscaled_const_input = const_upscale_block_100(const_input, filters=filters_gen)
    # upscaled_const_input = const_upscale_block_5(const_input, filters=filters_gen)
    print(f"upscaled constants shape: {upscaled_const_input.shape}")

    # concatenate with constant field?
    # but maybe with should happen after the residual blocks? Otherwise you're losing information?
    # generator_intermediate = concatenate([generator_input, upscaled_const_input])

    # (1,1) to (5,5), concatenate then upscale to (10,10) fingers crossed that works
    block_channels = [2*filters_gen, filters_gen]
    print('initial input shape',generator_input.shape)
    generator_intermediate = Dense(25, activation='relu')(generator_input)
    generator_intermediate = UpSampling2D(size=(5, 5), interpolation='bilinear')(generator_intermediate)
    print('shape after dense layer',generator_intermediate.shape)

    # generator_intermediate = UpSampling2D(size=(5, 5), interpolation='bilinear')(generator_input)
    print(f"Shape after upsampling step 1: {generator_intermediate.shape}")
    generator_intermediate = residual_block(generator_intermediate, filters=block_channels[0], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    generator_intermediate = UpSampling2D(size=(2, 2), interpolation='bilinear')(generator_intermediate)
    print(f"Shape after upsampling step 2: {generator_intermediate.shape}")
    generator_intermediate = residual_block(generator_intermediate, filters=block_channels[1], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)

    # feed in noise as 10 x 10 array
    
    noise_input = Input(shape=(None, None, noise_channels), name="noise_input") # when name='noise_input' there seems to be 2 noise input layers, even though noise_input_hr is a distinct layer, but works if this layer is called 'noise_inpu'
    print(f"noise_input shape 1: {noise_input.shape}")
    # Concatenate all inputs together
    generator_output = concatenate([generator_intermediate, upscaled_const_input, noise_input])
    # generator_output = concatenate([generator_input, noise_input])
    print(f"Shape after first concatenate: {generator_output.shape}")

    # Pass through 3 residual blocks
    n_blocks = 3 # this was 3 then 6 now 2 
    for i in range(n_blocks):
        generator_output = residual_block(generator_output, filters=filters_gen, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print('End of first residual block')
    print(f"Shape after first residual block: {generator_output.shape}")

    # Upsampling from (10,10) to (100,100) with alternating residual blocks
    # Now need to upsample from (1,1) to (100,100) I guess?
    # block_channels = [2*filters_gen, filters_gen]

    # continue with normal upsampling from og WGAN
    # generator_output = UpSampling2D(size=(5, 5), interpolation='bilinear')(generator_output)
    # print(f"Shape after upsampling step 3: {generator_output.shape}")
    # for i in range(1):
    #     generator_output = residual_block(generator_output, filters=block_channels[0], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    # print(f"Shape after residual block: {generator_output.shape}")

    # # # concatenate hr noise as a 50 x 50 array
    noise_input_hr = Input(shape=(None, None, noise_channels), name = "hr_noise_input_hr")
    # # print('hr noise input shape: ',noise_input_hr.shape)
    # # generator_output = concatenate([generator_output, noise_input_hr])
    # # Pass through 3 residual blocks

    # # for i in range(1):
    # #     generator_output = residual_block(generator_output, filters=filters_gen, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    # # print(f"Shape after third residual block: {generator_output.shape}")


    # generator_output = UpSampling2D(size=(2, 2), interpolation='bilinear')(generator_output)
    # print(f"Shape after upsampling step 4: {generator_output.shape}")
    # for i in range(1):
    #     generator_output = residual_block(generator_output, filters=block_channels[0], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    # print(f"Shape after residual block: {generator_output.shape}")

    # # # now upsampling to 200 x 200
    # # generator_output = UpSampling2D(size=(2, 2), interpolation='bilinear')(generator_output)
    # # print(f"Shape after upsampling step 4: {generator_output.shape}")
    # # for i in range(1):
    # #     generator_output = residual_block(generator_output, filters=block_channels[0], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    # # print(f"Shape after residual block: {generator_output.shape}")
    # # # and downsampling back to 100 x 100
    # # generator_output = Conv2D(filters=block_channels[1], kernel_size=(2, 2), strides=2, padding="valid", activation="relu")(generator_output)
    # # for i in range(1):
    # #     generator_output = residual_block(generator_output, filters=block_channels[0], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)


    # # TODO: add a downscaling and upscaling step here to improve spectral power?

    # # TODO: concantenate high res constant field with high res input features and maybe pass through some more residual blocks?
    # # and then edit the discriminator so that it matches.

    # # Concatenate with original size constants field and original size noise array?
    # # have to rename this layer to 'hr_noise_input_hr' becuase when it was 'noise_input_hr' that seemed to double count as both 'noise_input' and 'noise_input_hr'
    # noise_input_hr = Input(shape=(None, None, noise_channels), name = "hr_noise_input_hr")
    # # print('hr noise input shape: ',noise_input_hr.shape)
    # # generator_output = concatenate([generator_output, const_input, noise_input_hr])
    # generator_output = concatenate([generator_output, const_input])
    # print(f"Shape after second concatenate: {generator_output.shape}")

    # # Pass through 3 residual blocks
    # for i in range(3):
    #     generator_output = residual_block(generator_output, filters=filters_gen, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    # print(f"Shape after third residual block: {generator_output.shape}")

    # define new activation function
    def custom_activation(x):
        # return K.log(K.exp(x)+1)-K.log(K.exp((x-1)/1.1)+1) #too low
        # return K.log(K.exp(x)+1)-K.log(K.exp((x-1)/1.5)+1) #too extreme
        # return K.log(K.exp(x)+1)-K.log(K.exp((x-1)/1.15)+1) #too low again
        # return K.log(K.exp(x)+1)-K.log(K.exp((x-1)/1.2)+1)
        # return K.log(K.exp(x)+1)-K.log(K.exp((x-1)/1.25)+1)
        return K.log(K.exp(x)+1)-K.log(K.exp((x-1)/1.23)+1) #best for mean currently testing on more input variables
        # return K.log(K.exp(x)+1)-K.log(K.exp((x-1)/1.23)+1) # best for ver1
        # return K.log(K.exp(x)+1)-K.log(K.exp((x-1)/1.25)+1) # currently using for patchgan change v18
    get_custom_objects().update({'custom_activation': Activation(custom_activation)})
    
    # Output layer
    # generator_output = Conv2D(filters=1, kernel_size=(1, 1), activation='softplus', name="output")(generator_output)
    generator_output = Conv2D(filters=1, kernel_size=(1, 1), activation='custom_activation', name="output")(generator_output)
    print(f"Output shape: {generator_output.shape}")

    if mode == 'GAN':
        model = Model(inputs=[generator_input, const_input, noise_input, noise_input_hr], outputs=generator_output, name='gen')
        # model = Model(inputs=[generator_input, const_input, noise_input], outputs=generator_output, name='gen')
        # model = Model(inputs=[generator_input, noise_input], outputs=generator_output, name='gen')
        return model



def discriminator(arch,
                  input_channels=8,
                  constant_fields=1, #2
                  filters_disc=64,
                  conv_size=(3, 3),
                  padding=None,
                  stride=1,
                  relu_alpha=0.2,
                  norm=None,
                  dropout_rate=None):

    forceconv = True if arch == "forceconv" else False
    # Network inputs
    # low resolution condition
    generator_input = Input(shape=(None, None, input_channels), name="lo_res_inputs")
    print(f"generator_input shape: {generator_input.shape}")
    # constant fields
    const_input = Input(shape=(None, None, constant_fields), name="hi_res_inputs")
    print(f"constants_input shape: {const_input.shape}")
    # target image
    generator_output = Input(shape=(None, None, 1), name="output")
    print(f"generator_output shape: {generator_output.shape}")

    # convolve down constant fields to match ERA
    lo_res_const_input = const_upscale_block_100(const_input, filters=filters_disc)
    # lo_res_const_input = const_upscale_block_5(const_input, filters=filters_disc)
    print(f"upscaled constants shape: {lo_res_const_input.shape}")
    print(f"Shape of generator input before disc concatenation: {generator_input.shape}")
    print(tf.shape(generator_input))
    print(f"Shape of low res const input before disc concatenation: {lo_res_const_input.shape}")
    print(tf.shape(lo_res_const_input))

    # new step: upscale number values to (1,1) to (5,5) to (10,10) for concatenation!
    block_channels = [filters_disc, 2*filters_disc]
    # block_channels = [1, 2]
    lo_res_input = Dense(25, activation='relu')(generator_input)
    lo_res_input = UpSampling2D(size=(5, 5), interpolation='bilinear')(lo_res_input)
    print(f"Shape after upsampling lo_res_input input for disc step 1: {lo_res_input.shape}")

    # add new concat step in here
    # lo_res_input = concatenate([lo_res_input, lo_res_const_input])
    # print(f"Shape after lo-res concatenate: {lo_res_input.shape}")

    lo_res_input = residual_block(lo_res_input, filters=block_channels[0], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    lo_res_input = UpSampling2D(size=(2, 2), interpolation='bilinear')(lo_res_input)
    print(f"Shape after upsampling lo_res_input input for disc step 2: {lo_res_input.shape}")
    lo_res_input = residual_block(lo_res_input, filters=block_channels[1], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)


    # concatenate constants to lo-res input
    # lo_res_input = concatenate([generator_input, lo_res_const_input])

    # not concatenating here anymore, yes we are
    lo_res_input = concatenate([lo_res_input, lo_res_const_input])

    # lo_res_input = concatenate([generator_input])
    # lo_res_input = generator_input
    # print(f"Shape after lo-res concatenate: {lo_res_input.shape}")

    # concatenate constants to hi-res input
    # hi_res_input = concatenate([generator_output, const_input])
    hi_res_input = concatenate([generator_output, lo_res_const_input])

    # hi_res_input = generator_output
    print(f"Shape after hi-res concatenate: {hi_res_input.shape}")

    # # encode inputs using residual blocks
    # block_channels = [filters_disc, 2*filters_disc]

    # # run through one set of RBs
    for i in range(1):
        lo_res_input = residual_block(lo_res_input, filters=block_channels[0], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print(f"Shape of lo-res input after residual block: {lo_res_input.shape}")
    # hi_res_input = Conv2D(filters=block_channels[0], kernel_size=(5, 5), strides=5, padding="valid", activation="relu")(hi_res_input)
    print(f"Shape of hi_res_input after upsampling step 1: {hi_res_input.shape}")
    for i in range(1):
        hi_res_input = residual_block(hi_res_input, filters=block_channels[0], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    # print(f"Shape of hi-res input after residual block: {hi_res_input.shape}")

    # # run through second set of RBs
    for i in range(1):
        lo_res_input = residual_block(lo_res_input, filters=block_channels[1], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print(f"Shape of lo-res input after residual block: {lo_res_input.shape}")
    # hi_res_input = Conv2D(filters=block_channels[1], kernel_size=(2, 2), strides=2, padding="valid", activation="relu")(hi_res_input)
    print(f"Shape of hi_res_input after upsampling step 2: {hi_res_input.shape}")
    for i in range(1):
        hi_res_input = residual_block(hi_res_input, filters=block_channels[1], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print(f"Shape after residual block: {hi_res_input.shape}")
    print('End of first set of residual blocks')

    # # concatenate hi- and lo-res inputs channel-wise before passing through discriminator
    # print('lo-res-shape: ',lo_res_input.shape)
    # print('hi-res-shape: ',hi_res_input.shape)
    disc_input = concatenate([lo_res_input, hi_res_input])
    # disc_input = lo_res_input
    # print(f"Shape after concatenating lo-res input and hi-res input: {disc_input.shape}")

    # encode in residual blocks
    for i in range(2):
        disc_input = residual_block(disc_input, filters=filters_disc, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)

    print(f"Shape after residual block: {disc_input.shape}")
    print('End of second residual block')

    # discriminator output
    disc_output = GlobalAveragePooling2D()(disc_input)
    print(f"discriminator output shape after pooling: {disc_output.shape}")
    disc_output = Dense(64, activation='relu')(disc_output)
    print(f"discriminator output shape: {disc_output.shape}")
    disc_output = Dense(1, name="disc_output")(disc_output)
    print(f"discriminator output shape: {disc_output.shape}")

    disc = Model(inputs=[generator_input, const_input, generator_output], outputs=disc_output, name='disc')
    # disc = Model(inputs=[generator_input, generator_output], outputs=disc_output, name='disc')

    return disc
