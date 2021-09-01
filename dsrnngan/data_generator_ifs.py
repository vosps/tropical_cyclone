""" Data generator class for batched training of precipitation downscaling network """
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from data import load_ifs_nimrod_batch,load_hires_constants

return_dic = True

class DataGenerator(Sequence):
    def __init__(self, dates, ifs_fields,batch_size, log_precip=True,
                 crop = False,
                 shuffle=True,constants=None,hour='random',ifs_norm=True,
                 downsample = False,
    ):
        self.dates=dates
        self.batch_size=batch_size
        self.ifs_fields=ifs_fields
        self.log_precip=log_precip
        self.shuffle=shuffle
        self.hour=hour
        self.ifs_norm=ifs_norm
        self.crop = crop
        self.downsample = downsample
        if constants is None:
            self.constants=constants
        elif constants == True:
            self.constants=load_hires_constants(self.batch_size,crop=self.crop)
        else:
            self.constants=np.repeat(constants,self.batch_size,axis=0)
         
    def __len__(self):
        # Number of batches in dataset
        return len(self.dates) // self.batch_size

    def _dataset_downsampler(self,nimrod):
        # nimrod = tf.convert_to_tensor(nimrod,dtype=tf.float32)
        # print(nimrod.shape)
        kernel_tf = tf.constant(0.01,shape=(10,10,1,1), dtype=tf.float32)
        image = tf.nn.conv2d(nimrod, filters=kernel_tf, strides=[1, 10, 10, 1], padding='VALID',
                             name='conv_debug',data_format='NHWC')
        return image
        
    def __getitem__(self, idx):
        #Get batch at index idx
        dates_batch=self.dates[idx*self.batch_size:(idx+1)*self.batch_size]
        
        #Load and return this batch of images
        data_x_batch, data_y_batch=load_ifs_nimrod_batch(dates_batch,
                                                         ifs_fields=self.ifs_fields,
                                                         log_precip=self.log_precip,
                                                         hour=self.hour,crop=self.crop,
                                                         norm=self.ifs_norm)
        if self.downsample:
            data_x_batch = self._dataset_downsampler(data_y_batch[...,np.newaxis])
        
        if self.constants is None:
            if return_dic:
                return {"generator_input":data_x_batch}, {"generator_output":data_y_batch}
            else:
                return data_x_batch, data_y_batch
        else:
            if return_dic:
                return {"generator_input":data_x_batch,
                        "constants":self.constants},{"generator_output":data_y_batch}
            else:
                return data_x_batch, self.constants, data_y_batch
    
    def shuffle_data(self):
        np.random.shuffle(self.dates)
        
    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_data()

if __name__ == "__main__":
    pass
