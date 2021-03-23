import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Flatten, Conv2D, UpSampling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate

from blocks import residual_block, const_upscale_block
from layers import ReflectionPadding2D


def initial_state_model(num_preproc=3):
    initial_frame_in = Input(shape=(None,None,1))
    noise_in_initial = Input(shape=(None,None,8),
        name="noise_in_initial")

    h = ReflectionPadding2D(padding=(1,1))(initial_frame_in)
    h = Conv2D(256-noise_in_initial.shape[-1], kernel_size=(3,3))(h)
    h = Concatenate()([h,noise_in_initial])
    for i in range(num_preproc):
        h = residual_block(h, filters=256)

    return Model(
        inputs=[initial_frame_in,noise_in_initial],
        outputs=h
    )


def generator(era_dim=(10,10,9), const_dim=(250,250,2), noise_dim=(10,10,2), filters=32, conv_size=(3,3), stride=1, relu_alpha=0.2, norm=None, dropout_rate=None):
    
    # Network inputs 
    ##rainfall image                                                                                                                                                                                           
    generator_input = Input(shape=era_dim, name="generator_input")
    print(f"generator_input shape: {generator_input.shape}")
    ##constant fields
    const_input = Input(shape=const_dim, name="constants")
    print(f"constants_input shape: {const_input.shape}")
    ##noise
    noise_input = Input(shape=noise_dim, name="noise_input")
    print(f"noise_input shape: {noise_input.shape}")

    ## Convolve constant fields down to match other input dimensions
    upscaled_const_input = const_upscale_block(const_input, filters=filters)
    print(f"upscaled constants shape: {upscaled_const_input.shape}")
    
    ## Concatenate all inputs together
    generator_output = concatenate([generator_input, upscaled_const_input, noise_input])
    print(f"Shape after first concatenate: {generator_output.shape}")

    ## Pass through 3 residual blocks
    for i in range(3):
        generator_output = residual_block(generator_output, filters=filters, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate)
    print('End of first residual block')
    print(f"Shape after first residual block: {generator_output.shape}")
    
    print(f"Shape before upsampling: {generator_output.shape}")
    ## Upsampling residual blocks 
    block_channels = [64, 32]
    ## There are 2 items in block_channels so we upsample 2 times
    ## Upsampling size is hardcoded at (5,5)
    for i, channels in enumerate(block_channels):
        channels = block_channels[i]
        generator_output = UpSampling2D(size=(5,5), interpolation='bilinear')(generator_output)
        print(f"Shape after upsampling: {generator_output.shape}")
        generator_output = residual_block(generator_output, filters=channels, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate)
        print(f"Shape after residual block: {generator_output.shape}")
    print('End of upsampling residual block')
    print(f"Shape after upsampling residual block: {generator_output.shape}")
    
    ## Concatenate with full size constants field
    generator_output = concatenate([generator_output, const_input])
    print(f"Shape after second concatenate: {generator_output.shape}")
    
     ## Pass through 3 residual blocks
    for i in range(3):
        generator_output = residual_block(generator_output, filters=filters, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate)
    print('End of third residual block')
    print(f"Shape after third residual block: {generator_output.shape}")
    
    ## Output layer
    generator_output = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid', name="generator_output")(generator_output)
    print(f"Output shape: {generator_output.shape}")
    
    model = Model(inputs=[generator_input, const_input, noise_input], outputs=generator_output, name='gen')
    
    def noise_shapes(img_shape=(250,250)):
        noise_shape = (img_shape[0]//25, img_shape[1]//25, 8)
        return noise_shape
    
    return (model, noise_shapes)

def generator_initialized(gen, num_channels=1):
    noise_in = Input(shape=(None,None,8),
        name="noise_input")
    lores_in = Input(shape=(None,None,num_channels),
        name="generator_input")
    const_in = Input(shape=(), name = "constants")
    inputs = [lores_in, const_in, noise_in]

    (img_out,h) = gen(inputs)

    model = Model(inputs=inputs, outputs=img_out)

    def noise_shapes(img_shape=(250,250)):
        noise_shape = (img_shape[0]//25, img_shape[1]//25, 8)
        return noise_shape

    return (model, noise_shapes)


def discriminator(era_dim=(10,10,9), const_dim=(250,250,2), nimrod_dim=(250,250,1), filters=32, conv_size=(3,3), stride=1, relu_alpha=0.2, norm=None, dropout_rate=None):
    
    # Network inputs 
    ##rainfall image                                                                                                                                                                                         
    generator_input = Input(shape=era_dim, name="generator_input")
    print(f"generator_input shape: {generator_input.shape}")
    ##constant fields
    const_input = Input(shape=const_dim,name="constants")
    print(f"constants shape: {const_input.shape}")
    ##generator output
    generator_output = Input(shape=nimrod_dim, name="generator_output")
    print(f"generator_output shape: {generator_output.shape}")
    
    ##convolve down constant fields to match ERA
    lo_res_const_input = const_upscale_block(const_input, filters=filters)
    print(f"upscaled constants shape: {lo_res_const_input.shape}")
    
    ##concatenate constants to lo-res input
    lo_res_input = concatenate([generator_input, lo_res_const_input])
    print(f"Shape after lo-res concatenate: {lo_res_input.shape}")
    
    ##concatenate constants to hi-res input
    hi_res_input = concatenate([generator_output, const_input])
    print(f"Shape after hi-res concatenate: {hi_res_input.shape}")
    
    ##encode inputs using residual blocks
    ##stride of 5 means hi-res inputs are downsampled
    ##there are 2 items in block_channels so we pass through 2 residual blocks
    block_channels = [32, 64]
    for (i,channels) in enumerate(block_channels):
        lo_res_input = residual_block(lo_res_input, filters=channels, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate)
        print(f"Shape of lo-res input after residual block: {lo_res_input.shape}")
        hi_res_input = Conv2D(filters=channels, kernel_size=(5,5), strides=5, padding="valid", activation="relu")(hi_res_input)
        print(f"Shape after upscaling: {hi_res_input.shape}")
        hi_res_input = residual_block(hi_res_input, filters=channels, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate)
        print(f"Shape of hi-res input after residual block: {hi_res_input.shape}")
    print('End of first set of residual blocks')
    
    ##concatenate hi- and lo-res inputs channel-wise before passing through discriminator 
    disc_input = concatenate([lo_res_input, hi_res_input])
    print(f"Shape after concatenate: {disc_input.shape}")
    
    ##encode in residual blocks
    disc_input = residual_block(disc_input, filters=64, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate)
    disc_input = residual_block(disc_input, filters=64, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate)
    print(f"Shape after residual block: {disc_input.shape}")
    print('End of second residual block')

    ##discriminator output (in logits)
    disc_output = Conv2D(1, (3,3), padding="valid")(disc_input)
    print(f"discriminator output shape: {disc_output.shape}")
    disc_output = GlobalAveragePooling2D()(disc_output)
    print(f"discriminator output shape after pooling: {disc_output.shape}")
    disc_output = Flatten(dtype=tf.dtypes.float32, name="disc_output")(disc_output)
    print(f"discriminator output shape: {disc_output.shape}")

    disc = Model(inputs=[generator_input, const_input, generator_output], outputs=disc_output, name='disc')

    return disc


# def generator(num_channels=1, num_timesteps=8, num_preproc=3):
#     initial_state = Input(shape=(None,None,256))
#     noise_in_update = Input(shape=(num_timesteps,None,None,8),
#         name="noise_in_update")
#     lores_in = Input(shape=(num_timesteps,None,None,num_channels),
#         name="cond_in")
#     inputs = [lores_in, initial_state, noise_in_update]

#     xt = TimeDistributed(ReflectionPadding2D(padding=(1,1)))(lores_in)
#     xt = TimeDistributed(Conv2D(256-noise_in_update.shape[-1], 
#         kernel_size=(3,3)))(xt)
#     xt = Concatenate()([xt,noise_in_update])
#     for i in range(num_preproc):
#         xt = res_block(256, time_dist=True, activation='relu')(xt)

#     def gen_gate(activation='sigmoid'):
#         def gate(x):
#             x = ReflectionPadding2D(padding=(1,1))(x)
#             x = Conv2D(256, kernel_size=(3,3))(x)
#             if activation is not None:
#                 x = Activation(activation)(x)
#             return x
#         return Lambda(gate)
    
#     x = CustomGateGRU(
#         update_gate=gen_gate(),
#         reset_gate=gen_gate(),
#         output_gate=gen_gate(activation=None),
#         return_sequences=True,
#         time_steps=num_timesteps
#     )([xt,initial_state])

#     h = x[:,-1,...]
    
#     block_channels = [256, 256, 128, 64, 32]
#     for (i,channels) in enumerate(block_channels):
#         if i > 0:
#             x = TimeDistributed(UpSampling2D(interpolation='bilinear'))(x)
#         x = res_block(channels, time_dist=True, activation='leakyrelu')(x)

#     x = TimeDistributed(ReflectionPadding2D(padding=(1,1)))(x)
#     img_out = TimeDistributed(Conv2D(num_channels, kernel_size=(3,3),
#         activation='sigmoid'))(x)

#     model = Model(inputs=inputs, outputs=[img_out,h])

#     def noise_shapes(img_shape=(128,128)):
#         noise_shape_update = (
#             num_timesteps, img_shape[0]//16, img_shape[1]//16, 8
#         )
#         return [noise_shape_update]

#     return (model, noise_shapes)



# def discriminator(num_channels=1, num_timesteps=8):
#     hires_in = Input(shape=(num_timesteps,None,None,num_channels), name="sample_in")
#     lores_in = Input(shape=(num_timesteps,None,None,num_channels), name="cond_in")

#     x_hr = hires_in
#     x_lr = lores_in

#     block_channels = [32, 64, 128, 256]
#     for (i,channels) in enumerate(block_channels):
#         x_hr = res_block(channels, time_dist=True,
#             norm="spectral", stride=2)(x_hr)
#         x_lr = res_block(channels, time_dist=True,
#             norm="spectral")(x_lr)

#     x_joint = Concatenate()([x_lr,x_hr])
#     x_joint = res_block(256, time_dist=True, norm="spectral")(x_joint)
#     x_joint = res_block(256, time_dist=True, norm="spectral")(x_joint)

#     x_hr = res_block(256, time_dist=True, norm="spectral")(x_hr)
#     x_hr = res_block(256, time_dist=True, norm="spectral")(x_hr)    

#     def disc_gate(activation='sigmoid'):
#         def gate(x):
#             x = ReflectionPadding2D(padding=(1,1))(x)
#             x = SNConv2D(256, kernel_size=(3,3),
#                 kernel_initializer='he_uniform')(x)
#             if activation is not None:
#                 x = Activation(activation)(x)
#             return x
#         return Lambda(gate)

#     h = Lambda(lambda x: tf.zeros_like(x[:,0,...]))
#     x_joint = CustomGateGRU(
#         update_gate=disc_gate(),
#         reset_gate=disc_gate(),
#         output_gate=disc_gate(activation=None),
#         return_sequences=True,
#         time_steps=num_timesteps
#     )([x_joint,h(x_joint)])
#     x_hr = CustomGateGRU(
#         update_gate=disc_gate(),
#         reset_gate=disc_gate(),
#         output_gate=disc_gate(activation=None),
#         return_sequences=True,
#         time_steps=num_timesteps
#     )([x_hr,h(x_hr)])

#     x_avg_joint = TimeDistributed(GlobalAveragePooling2D())(x_joint)
#     x_avg_hr = TimeDistributed(GlobalAveragePooling2D())(x_hr)

#     x = Concatenate()([x_avg_joint,x_avg_hr])
#     x = TimeDistributed(SNDense(256))(x)
#     x = LeakyReLU(0.2)(x)

#     disc_out = TimeDistributed(SNDense(1))(x)

#     disc = Model(inputs=[lores_in, hires_in], outputs=disc_out,
#         name='disc')

#     return disc
