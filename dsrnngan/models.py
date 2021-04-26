import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Flatten, Conv2D, UpSampling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

from blocks import residual_block, const_upscale_block

def generator(era_dim=(10,10,9), const_dim=(250,250,2), noise_dim=(10,10,8), filters=64, conv_size=(3,3), stride=1, relu_alpha=0.2, norm=None, dropout_rate=None):
    
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
    block_channels = [2*filters, filters]
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
    generator_output = Conv2D(filters=1, kernel_size=(1,1), activation='softplus', name="generator_output")(generator_output)
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

def generator_deterministic(era_dim=(10,10,9), const_dim=(250,250,2), filters=64, conv_size=(3,3), stride=1, relu_alpha=0.2, norm=None, dropout_rate=None):
    # Network inputs 
    ##rainfall image                                                                                                                                                 
    generator_input = Input(shape=era_dim, name="generator_input")
    ##constant fields
    const_input = Input(shape=const_dim, name="constants")

    ## Convolve constant fields down to match other input dimensions
    upscaled_const_input = const_upscale_block(const_input, filters=filters)
    ## Concatenate all inputs together
    generator_output = concatenate([generator_input, upscaled_const_input])

    ## Pass through 3 residual blocks
    for i in range(3):
        generator_output = residual_block(generator_output, filters=filters, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate)
    print('End of first residual block')
    print(f"Shape after first residual block: {generator_output.shape}")
    
    print(f"Shape before upsampling: {generator_output.shape}")
    ## Upsampling residual blocks 
    block_channels = [2*filters, filters]
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
    generator_output = Conv2D(filters=1, kernel_size=(1,1), activation='softplus', name="generator_output")(generator_output)
    print(f"Output shape: {generator_output.shape}")
    
    gen = Model(inputs=[generator_input, const_input], outputs=generator_output, name='gen')
    
    return gen

def discriminator(era_dim=(10,10,9), const_dim=(250,250,2), nimrod_dim=(250,250,1), filters=64, conv_size=(3,3), stride=1, relu_alpha=0.2, norm=None, dropout_rate=None):
    
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
    block_channels = [filters, 2*filters]
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
    disc_input = residual_block(disc_input, filters=filters, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate)
    disc_input = residual_block(disc_input, filters=filters, conv_size=conv_size, stride=stride, relu_alpha=relu_alpha, norm=norm, dropout_rate=dropout_rate)
    print(f"Shape after residual block: {disc_input.shape}")
    print('End of second residual block')

    ##discriminator output
    disc_output = GlobalAveragePooling2D()(disc_input)
    print(f"discriminator output shape after pooling: {disc_output.shape}")
    disc_output = Dense(64, activation='relu')(disc_output)
    print(f"discriminator output shape: {disc_output.shape}")
    disc_output = Dense(1, name="disc_output")(disc_output)
    print(f"discriminator output shape: {disc_output.shape}")

    disc = Model(inputs=[generator_input, const_input, generator_output], outputs=disc_output, name='disc')

    return disc

