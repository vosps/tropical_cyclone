import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, UpSampling2D, Layer
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, LeakyReLU
from keras.layers.core import Activation
from tensorflow.keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from blocks import residual_block, const_upscale_block_100

class GlobalWeightedAveragePooling2D(Layer):
    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                    shape=(input_shape[1], input_shape[2], 1),
                                    initializer='ones',
                                    trainable=True)
        Layer.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3],

    def call(self, x):
        
        x = x*self.kernel
        x = K.mean(x, axis=(1, 2))

        return x

def generator(mode,
              arch,
              input_channels=9,
              latent_variables=1,
              noise_channels=8,
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
    print(f"generator_input shape: {generator_input.shape}")
    # constant fields
    const_input = Input(shape=(None, None, constant_fields), name="hi_res_inputs")
    print(f"constants_input shape: {const_input.shape}")

    # Convolve constant fields down to match other input dimensions
    upscaled_const_input = const_upscale_block_100(const_input, filters=filters_gen)
    print(f"upscaled constants shape: {upscaled_const_input.shape}")

    if mode in ('det', 'VAEGAN'):
        # Concatenate all inputs together
        # generator_output = concatenate([generator_input, upscaled_const_input])
        generator_output = concatenate([generator_input])
    elif mode == 'GAN':
        # noise
        # noise_input = Input(shape=(None, None, noise_channels), name="noise_input") 
        noise_input = Input(shape=(None, None, noise_channels), name="noise_input") # change so shape isn't None
        print(f"noise_input shape: {noise_input.shape}")
        # Concatenate all inputs together
        generator_output = concatenate([generator_input, upscaled_const_input, noise_input])
        # generator_output = concatenate([generator_input, noise_input])
        print(f"Shape after first concatenate: {generator_output.shape}")

    # Pass through 3 residual blocks
    n_blocks = 6 # this was 3
    for i in range(n_blocks):
        generator_output = residual_block(generator_output, filters=filters_gen, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print('End of first residual block')
    print(f"Shape after first residual block: {generator_output.shape}")

    if mode == 'VAEGAN':
        # encoder model and outputs
        means = Conv2D(filters=latent_variables, kernel_size=1, activation=LeakyReLU(alpha=relu_alpha), padding="valid")(generator_output)
        logvars = Conv2D(filters=latent_variables, kernel_size=1, activation=LeakyReLU(alpha=relu_alpha), padding="valid")(generator_output)
        # encoder_model = Model(inputs=[generator_input, const_input], outputs=[means, logvars], name='encoder')
        encoder_model = Model(inputs=[generator_input], outputs=[means, logvars], name='encoder')
        # decoder model and inputs
        mean_input = tf.keras.layers.Input(shape=(None, None, latent_variables), name="mean_input")
        logvar_input = tf.keras.layers.Input(shape=(None, None, latent_variables), name="logvar_input")
        noise_input = Input(shape=(None, None, latent_variables), name="noise_input")
        # Generate random variables from mean & logvar
        generator_output = tf.multiply(noise_input, tf.exp(logvar_input * .5)) + mean_input
        print(f"Shape of random variables: {generator_output.shape}")
    else:
        pass

    if mode == "VAEGAN":
        # basically the vaegan had poor ensemble spread and this was because the noise was passed in too late in the process. Therefore 
        # we add three residual blocks so that the noise gets a change to diverge a bit
        # Pass through 3 more residual blocks
        n_blocks = 6 # before this was 3, but I'm increasing it to see if it helps with spread
        for i in range(n_blocks):
            generator_output = residual_block(generator_output, filters=filters_gen, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
        print('End of extra low-res residual blocks')
        print(f"Shape after extra low-res residual blocks: {generator_output.shape}")

    # Upsampling from (10,10) to (100,100) with alternating residual blocks
    block_channels = [2*filters_gen, filters_gen]
    generator_output = UpSampling2D(size=(5, 5), interpolation='bilinear')(generator_output)
    print(f"Shape after upsampling step 1: {generator_output.shape}")
    generator_output = residual_block(generator_output, filters=block_channels[0], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print(f"Shape after residual block: {generator_output.shape}")
    generator_output = UpSampling2D(size=(2, 2), interpolation='bilinear')(generator_output)
    print(f"Shape after upsampling step 2: {generator_output.shape}")
    generator_output = residual_block(generator_output, filters=block_channels[1], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print(f"Shape after residual block: {generator_output.shape}")

    # Concatenate with original size constants field
    generator_output = concatenate([generator_output, const_input])
    print(f"Shape after second concatenate: {generator_output.shape}")

    # Pass through 3 residual blocks
    for i in range(3):
        generator_output = residual_block(generator_output, filters=filters_gen, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print(f"Shape after third residual block: {generator_output.shape}")


    def custom_activation(x):
        return K.log(K.exp(x)+1)-K.log(K.exp((x-1)/1.1)+1)
        # return K.log(K.exp(x)+1)-K.log(K.exp((x-1)/1.01)+1)
    get_custom_objects().update({'custom_activation': Activation(custom_activation)})
    
    # Output layer
    # generator_output = Conv2D(filters=1, kernel_size=(1, 1), activation='softplus', name="output")(generator_output)
    generator_output = Conv2D(filters=1, kernel_size=(1, 1), activation='custom_activation', name="output")(generator_output)
    # generator_output = Conv2D(filters=1, kernel_size=(1, 1), activation='softplus', name="output")(generator_output)
    print(f"Output shape: {generator_output.shape}")

    if mode == 'VAEGAN':
        # decoder_model = Model(inputs=[mean_input, logvar_input, noise_input, const_input], outputs=generator_output, name='decoder')
        decoder_model = Model(inputs=[mean_input, logvar_input, noise_input], outputs=generator_output, name='decoder')
        return (encoder_model, decoder_model)
    elif mode == 'GAN':
        model = Model(inputs=[generator_input, const_input, noise_input], outputs=generator_output, name='gen')
        # model = Model(inputs=[generator_input, noise_input], outputs=generator_output, name='gen')
        return model
    elif mode == 'det':
        # model = Model(inputs=[generator_input, const_input], outputs=generator_output, name='gen')
        model = Model(inputs=[generator_input], outputs=generator_output, name='gen')
        return model


def discriminator(arch,
                  input_channels=9,
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
    print(f"upscaled constants shape: {lo_res_const_input.shape}")

    # concatenate constants to lo-res input
    lo_res_input = concatenate([generator_input, lo_res_const_input])
    # lo_res_input = generator_input
    print(f"Shape after lo-res concatenate: {lo_res_input.shape}")

    # concatenate constants to hi-res input
    hi_res_input = concatenate([generator_output, const_input])
    # hi_res_input = generator_output
    print(f"Shape after hi-res concatenate: {hi_res_input.shape}")

    # encode inputs using residual blocks
    block_channels = [filters_disc, 2*filters_disc]
    # run through one set of RBs
    lo_res_input = residual_block(lo_res_input, filters=block_channels[0], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print(f"Shape of lo-res input after residual block: {lo_res_input.shape}")
    hi_res_input = Conv2D(filters=block_channels[0], kernel_size=(5, 5), strides=5, padding="valid", activation="relu")(hi_res_input)
    print(f"Shape of hi_res_input after upsampling step 1: {hi_res_input.shape}")
    hi_res_input = residual_block(hi_res_input, filters=block_channels[0], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print(f"Shape of hi-res input after residual block: {hi_res_input.shape}")
    # run through second set of RBs
    lo_res_input = residual_block(lo_res_input, filters=block_channels[1], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print(f"Shape of lo-res input after residual block: {lo_res_input.shape}")
    hi_res_input = Conv2D(filters=block_channels[1], kernel_size=(2, 2), strides=2, padding="valid", activation="relu")(hi_res_input)
    print(f"Shape of hi_res_input after upsampling step 2: {hi_res_input.shape}")
    hi_res_input = residual_block(hi_res_input, filters=block_channels[1], conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print(f"Shape after residual block: {hi_res_input.shape}")
    print('End of first set of residual blocks')

    # concatenate hi- and lo-res inputs channel-wise before passing through discriminator
    disc_input = concatenate([lo_res_input, hi_res_input])
    print(f"Shape after concatenating lo-res input and hi-res input: {disc_input.shape}")

    # encode in residual blocks
    disc_input = residual_block(disc_input, filters=filters_disc, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    disc_input = residual_block(disc_input, filters=filters_disc, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate, padding=padding, force_1d_conv=forceconv)
    print(f"Shape after residual block: {disc_input.shape}")
    print('End of second residual block')

    # weighted average on lo res input values
    # GlobalWeightedAveragePooling2D
    # https://github.com/csvance/keras-global-weighted-pooling/blob/master/gwp.py
    # https://arxiv.org/pdf/1809.08264.pdf

    # 1. normalise lo_res_input
    # 2. w_avg = unit matrix + constant * lo_res_input
    # 3, weighted_disc_input = w_avg * disc_input

    # class GlobalWeightedAveragePooling2D(Layer):
    #     def __init__(self, **kwargs):
    #         Layer.__init__(self, **kwargs)
    #         self.kernel = None

    #     def build(self, input_shape):
    #         self.kernel = self.add_weight(name='kernel',
    #                                     shape=(input_shape[1], input_shape[2], 1),
    #                                     initializer='ones',
    #                                     trainable=True)
    #         Layer.build(self, input_shape)

    #     def compute_output_shape(self, input_shape):
    #         return input_shape[0], input_shape[3], 

    #     def call(self, x):
            
    #         x = x*self.kernel
    #         x = K.mean(x, axis=(1, 2))

    #         return x


    weights = False
    # discriminator output
    if weights == False:
        disc_output = GlobalAveragePooling2D()(disc_input)
    # weights = 
    # disc_weighted = weighted_layer_thing()(disc_input)
    else:
        disc_output = GlobalWeightedAveragePooling2D()(disc_input)
    print(f"discriminator output shape after pooling: {disc_output.shape}")
    disc_output = Dense(64, activation='relu')(disc_output)
    print(f"discriminator output shape: {disc_output.shape}")
    disc_output = Dense(1, name="disc_output")(disc_output)
    print(f"discriminator output shape: {disc_output.shape}")

    disc = Model(inputs=[generator_input, const_input, generator_output], outputs=disc_output, name='disc')
    # disc = Model(inputs=[generator_input, generator_output], outputs=disc_output, name='disc')

    return disc
