from tensorflow.keras.layers import Add, Conv2D, Dropout, LeakyReLU, BatchNormalization, AveragePooling2D


def residual_block(x, filters, conv_size=(3, 3), stride=1, relu_alpha=0.2, norm=None, dropout_rate=None):
    in_channels = int(x.shape[-1])
    x_in = x

    x_in = AveragePooling2D(pool_size=(stride, stride))(x_in)
    if (filters != in_channels):
        x_in = Conv2D(filters=filters, kernel_size=(1, 1), padding="same")(x_in)

    # first block of activation and 3x3 convolution
    x = LeakyReLU(relu_alpha)(x)
    x = Conv2D(filters=filters, kernel_size=conv_size, padding="same")(x)
    if norm == "batch":
        x = BatchNormalization()(x)
    elif norm is None:
        pass
    else:
        print("norm type not implemented")
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
        print("Dropout rate is {dropout_rate}")

    # second block of activation and 3x3 convolution
    x = LeakyReLU(relu_alpha)(x)
    x = Conv2D(filters=filters, kernel_size=conv_size, padding="same")(x)
    if norm == "batch":
        x = BatchNormalization()(x)
    elif norm is None:
        pass
    else:
        print("norm type not implemented")
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
        print("Dropout rate is {dropout_rate}")

    # skip connection
    x = Add()([x, x_in])

    return x


def const_upscale_block(const_input, filters):

    # Map (n x 250 x 250 x 2) to (n x 10 x 10 x f)
    const_output = Conv2D(filters=filters, kernel_size=(6, 6), strides=4, padding="valid", activation="relu")(const_input)
    const_output = Conv2D(filters=filters, kernel_size=(2, 2), strides=3, padding="valid", activation="relu")(const_output)
    const_output = Conv2D(filters=filters, kernel_size=(3, 3), strides=2, padding="valid", activation="relu")(const_output)

    return const_output


def const_upscale_block_100(const_input, filters):

    # Map (n x 100 x 100 x 2) to (n x 10 x 10 x f)
    const_output = Conv2D(filters=filters, kernel_size=(5, 5), strides=5, padding="valid", activation="relu")(const_input)
    const_output = Conv2D(filters=filters, kernel_size=(2, 2), strides=2, padding="valid", activation="relu")(const_output)

    return const_output
