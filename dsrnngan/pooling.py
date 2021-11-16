import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, AvgPool2D


def pool(x, pool_type):
    """Apply pooling operation (via Tensorflow) to input Numpy array x.
    x should be 4-dimensional: N x W x H x C.
    Pooling is applied on W and H dimensions.
    
    """
    pool_op = {
        'max_4': MaxPool2D(pool_size=(4, 4), strides=(1, 1)),
        'max_16': MaxPool2D(pool_size=(16, 16), strides=(1, 1)),
        'max_10_no_overlap': MaxPool2D(pool_size=(10, 10), strides=(10, 10)),
        'avg_4': AvgPool2D(pool_size=(4, 4), strides=(1, 1)),
        'avg_16': AvgPool2D(pool_size=(16, 16), strides=(1, 1)),
        'avg_10_no_overlap': AvgPool2D(pool_size=(10, 10), strides=(10, 10)),
    }[pool_type]
    
    return pool_op(x.astype("float32")).numpy()