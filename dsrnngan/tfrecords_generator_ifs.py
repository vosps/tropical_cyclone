import numpy as np
import tensorflow as tf
import glob
from data import all_ifs_fields,ifs_hours

records_folder = '/ppdata/tfrecordsIFS20/'
return_dic = True

def DataGenerator(year, batch_size, repeat=True, downsample=False, weights=None):
    return create_mixed_dataset(year, batch_size, repeat=repeat, downsample=downsample, weights=weights)

# TODO: swap create_mixed_dataset for this commented out code
# def create_random_dataset(year,batch_size,era_shape=(10,10,9),con_shape=(100,100,2),
#                          out_shape=(100,100,1),repeat=True,
#                          folder=records_folder, shuffle_size = 1024):
#     dataset = create_dataset(year, '*', era_shape=era_shape,con_shape=con_shape,
#                                out_shape=out_shape,folder=folder,repeat=repeat,
#                                shuffle_size = shuffle_size)
#     return dataset.batch(batch_size).prefetch(2)
# 


def create_mixed_dataset(year,batch_size,era_shape=(10,10,1),
                         out_shape=(100,100,1),repeat=True,downsample = False,
                         folder=records_folder, shuffle_size = 1024,
                         weights = None):

    classes = 4
    if weights is None:
        weights = [1./classes]*classes
    # create a list of 4 datasets
    datasets = [create_dataset(year, i, era_shape=era_shape,
                            #    con_shape=con_shape,
                               out_shape=out_shape, folder=folder,
                               shuffle_size=shuffle_size, repeat=repeat)
                for i in range(classes)]
    # randomly sample this list with weights
    sampled_ds=tf.data.experimental.sample_from_datasets(datasets,
                                                         weights=weights).batch(batch_size)
    # TODO: so these steps aren't properly needed, so I need to replace these lines with perhaps just sampled_ds = create_dataset 
    
    if downsample and return_dic:
        sampled_ds=sampled_ds.map(_dataset_downsampler)
    elif downsample and not return_dic:
        sampled_ds=sampled_ds.map(_dataset_downsampler_list)
    sampled_ds=sampled_ds.prefetch(2)
    return sampled_ds

# Note, if we wanted fewer classes, we can use glob syntax to grab multiple classes as once
# e.g. create_dataset(2015,"[67]")
# will take classes 6 & 7 together

def _dataset_downsampler(inputs,outputs):
    image = outputs['output']
    kernel_tf = tf.constant(0.01,shape=(10,10,1,1), dtype=tf.float32)
    image = tf.nn.conv2d(image, filters=kernel_tf, strides=[1, 10, 10, 1], padding='VALID',
                         name='conv_debug',data_format='NHWC')
    inputs['lo_res_inputs'] = image
    return inputs,outputs

# def _dataset_downsampler_list(inputs, constants, outputs):
def _dataset_downsampler_list(inputs, outputs):
    image = outputs
    kernel_tf = tf.constant(0.01,shape=(10,10,1,1), dtype=tf.float32)
    # kernel_tf = tf.constant(0.01,shape=(10,10,1,1), dtype=tf.float64)
    image = tf.nn.conv2d(image, filters=kernel_tf, strides=[1, 10, 10, 1], padding='VALID', name='conv_debug',data_format='NHWC')
    inputs = image
    # return inputs, constants, outputs
    return inputs, outputs

def _parse_batch(record_batch,insize=(10,10,9),consize=(100,100,2),
                 outsize=(100,100,1)):
    # Create a description of the features
    feature_description = {
        'generator_input': tf.io.FixedLenFeature(insize, tf.float32),
        'constants': tf.io.FixedLenFeature(consize, tf.float32),
        'generator_output': tf.io.FixedLenFeature(outsize, tf.float32),
    }

    # Parse the input `tf.Example` proto using the dictionary above
    example = tf.io.parse_example(record_batch, feature_description)
    if return_dic:
        return ({'lo_res_inputs': example['generator_input'],
                 'hi_res_inputs': example['constants']},
                {'output': example['generator_output']})
    else:
        return example['generator_input'], example['constants'], example['generator_output']


# def create_dataset(year,clss,era_shape=(10,10,9),con_shape=(100,100,2),out_shape=(100,100,1),
#                    folder=records_folder, shuffle_size = 1024, repeat=True):
def create_dataset(year,clss,era_shape=(10,10,1),out_shape=(100,100,1),
                   folder=records_folder, shuffle_size = 1024, repeat=True):
    """
    this function creates the dataset in the format input, constants, output

    variables
                input : (10,10,9)
                low resolution input data with all the 9 data variables
                constant : (100,100,2)
                the two constant variables in high resolution; LSM and orography
                output : (100,100,1)
                real or fake image in high resolution

    I need to change it so that right now it only takes one variable, and we don't need the constant field yet

    """
    print('doing autotune...')
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    print('autotune done!')
    # TODO: not sure if this should be commented out.
    # if type(year)==str or type(year) == int:
    #     fl = glob.glob(f"{folder}/{year}_*.{clss}.tfrecords")
    # elif type(year)==list:
    #     fl = []
    #     for y in year:
    #         fl+=glob.glob(f"{folder}/{y}_*.{clss}.tfrecords")
    # else:
    #     assert False, f"TFRecords not configure for type {type(year)}"
    # files_ds = tf.data.Dataset.list_files(fl)
    # ds = tf.data.TFRecordDataset(files_ds,
    #                              num_parallel_reads=AUTOTUNE)
    # ds = ds.shuffle(shuffle_size)
    # ds = ds.map(lambda x: _parse_batch(x, insize=era_shape,consize=con_shape,
    #                                    outsize=out_shape))
    print('making ds the first time...')
    fl = ['/user/work/al18709/tc_data_flipped/train_X.npy']
    files_ds = tf.data.Dataset.list_files(fl)
    ds = tf.data.TFRecordDataset(files_ds,
                                 num_parallel_reads=AUTOTUNE)
    print('made ds!')
    # ds = ds.shuffle(shuffle_size)
    
    # insert my code here
    # TODO: ensure ds is in the correct shape
    # TODO: only open 8000 images to save time. done
    print('loading in actual data...')
    x = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/train_X.npy'),axis=3)) # inputs this will eventually be (nimags,10,10,nfeatures)
    y = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/train_y.npy'),axis=3)) # outputs
    # z = np.load('/user/work/al18709/tc_data/train_y.npy') # constants, this will eventually be (100,100,2)
    print('x shape: ',x.shape)
    print('y shape: ',y.shape)
    print('repeat is: ', repeat)
    print('number of nans in x: ',np.count_nonzero(np.isnan(x)))
    print('number of nans in y: ',np.count_nonzero(np.isnan(y)))
    print('making ds from data...')
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    print('ds made!')
    # shuffle, map, then repeat
    print('shufflinf dataset...')
    ds = ds.shuffle(shuffle_size)
    print('ds shuffled!')
    # ds = ds.map(lambda x: _parse_batch(x, insize=era_shape, outsize=out_shape))

    print('ds in new format',ds)

    if repeat:
        print('repeating... ')
        return ds.repeat()
    else:
        return ds

# def create_fixed_dataset(year=None,mode='validation',batch_size=16,
#                          downsample=False,
#                          era_shape=(20,20,9),con_shape=(200,200,2),out_shape=(200,200,1),
#                          name=None,folder=records_folder):
#     assert year is not None or name is not None, "Must specify year or file name"
#     if folder[-1] != '/':
#         folder = folder + '/'
#     if name is None:
#         name = f"{folder}{mode}{year}.tfrecords"
#     else:
#         if name[0] != '/':
#             name = folder + name
#     fl = glob.glob(name)
#     files_ds = tf.data.Dataset.list_files(fl)
#     ds = tf.data.TFRecordDataset(files_ds,
#                                  num_parallel_reads=1)
#     ds = ds.map(lambda x: _parse_batch(x, insize=era_shape,consize=con_shape,
#                                        outsize=out_shape))
#     ds = ds.batch(batch_size)
#     if downsample and return_dic:
#         ds=ds.map(_dataset_downsampler)
#     elif downsample and not return_dic:
#         ds=ds.map(_dataset_downsampler_list)
#     return ds

def create_fixed_dataset(year=None,mode='validation',batch_size=16,
                         downsample=False,
                         storm=False,
                         era_shape=(10,10,1),out_shape=(100,100,1),
                         name=None,folder=records_folder):
    print('opening fixed dataset...')
    # added this in

    if mode == 'storm':
        dataset = storm
        x = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_mswep_extend_flipped/X_%s.npy' % dataset),axis=3))
        y = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_mswep_extend_flipped/y_%s.npy' % dataset),axis=3))
    elif mode == 'storm_era5':
        dataset = storm
        x = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_era5_flipped_10/X_%s.npy' % dataset),axis=3))
        y = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_era5_flipped_10/y_%s.npy' % dataset),axis=3))
    elif mode == 'storm_era5_corrected':
        dataset = storm
        x = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_era5_flipped_10/X_%s_corrected.npy' % dataset),axis=3))
        y = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_era5_flipped_10/y_%s.npy' % dataset),axis=3))
    elif mode == 'era5':
        dataset = 'valid'
        x = np.float32(np.expand_dims(np.load('/user/home/al18709/work/tc_data_era5_flipped_10/%s_X.npy' % dataset),axis=3))
        y = np.float32(np.expand_dims(np.load('/user/home/al18709/work/tc_data_era5_flipped_10/%s_y.npy' % dataset),axis=3))
    elif mode == 'era5_corrected':
        dataset = 'valid'
        x = np.float32(np.load('/user/home/al18709/work/tc_data_era5_flipped_10/%s_corrected_X.npy' % dataset))
        y = np.float32(np.expand_dims(np.load('/user/home/al18709/work/tc_data_era5_flipped_10/%s_y.npy' % dataset),axis=3))
    else:
        if mode == 'validation':
            dataset = 'valid'
        else:
            dataset = mode
        x = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/%s_X.npy' % dataset),axis=3))
        y = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/%s_y.npy' % dataset),axis=3))

    # if mode == 'train':
    #     x = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/train_X.npy'),axis=3))
    #     y = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/train_y.npy'),axis=3))
    # elif mode == 'validation':
    #     x = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/valid_X.npy'),axis=3))
    #     y = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/valid_y.npy'),axis=3))
    # elif mode == 'extreme_valid':
    #     x = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/extreme_valid_X.npy'),axis=3))
    #     y = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/extreme_valid_y.npy'),axis=3))
    # elif mode == 'extreme_test':
    #     x = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/extreme_test_X.npy'),axis=3))
    #     y = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/extreme_test_y.npy'),axis=3))
    # elif mode == 'test':
    #     x = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/test_X.npy'),axis=3))
    #     y = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_flipped/test_y.npy'),axis=3))
    # elif mode == 'gcm':
    #     x = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_mswep/gcm_X.npy'),axis=3))
    #     y = np.float32(np.expand_dims(np.load('/user/work/al18709/tc_data_mswep/gcm_X.npy'),axis=3))
    # elif mode == 'cmip':
    #     x = np.float32(np.expand_dims(np.load('/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/storm_rain.npy')[-1000:,:,:],axis=3))
    #     y = np.float32(np.expand_dims(np.load('/user/home/al18709/work/CMIP6/HighResMIP/EC-Earth3p/historical/storm_rain.npy')[-1000:,:,:],axis=3))

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(batch_size)
    # return_dic=False #adding this in to get roc curve to work
    if downsample and return_dic:
        ds=ds.map(_dataset_downsampler)
    elif downsample and not return_dic:
        ds=ds.map(_dataset_downsampler_list)
    print('ds in new format',ds)

    return ds
        

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def write_data(year,
               ifs_fields=all_ifs_fields,
               hours=ifs_hours,
               era_chunk_width=20,
               num_class=4 ,
               log_precip=True,
               ifs_norm=True
):
    # from data import get_dates
    from data_generator_ifs import DataGenerator

    dates = get_dates(year)

    nim_size = 940
    era_size = 94

    upscaling_factor = 10

    nsamples = (era_size//era_chunk_width + 1)**2
    print("Samples per image:", nsamples)
    import random

    for hour in hours:
        dgc = DataGenerator(dates=dates,
                            ifs_fields=ifs_fields,
                            batch_size=1,
                            log_precip=log_precip,
                            crop=True,
                            shuffle=False,
                            constants=True,
                            hour=hour,
                            ifs_norm=ifs_norm)
        fle_hdles = []
        for fh in range(num_class):
            flename = f"{records_folder}{year}_{hour}.{fh}.tfrecords"
            fle_hdles.append( tf.io.TFRecordWriter(flename))
        for batch in range(len(dates)):
            print(hour, batch)
            sample = dgc.__getitem__(batch)
            for k in range(sample[1]['output'].shape[0]):
                for ii in range(nsamples):
                    # e.g. for ERA width 94 and era_chunk_width 20, can have 0:20 up to 74:94
                    idx = random.randint(0, era_size-era_chunk_width)
                    idy = random.randint(0, era_size-era_chunk_width)

                    nimrod = sample[1]['output'][k,
                                                 idx*upscaling_factor:(idx+era_chunk_width)*upscaling_factor,
                                                 idy*upscaling_factor:(idy+era_chunk_width)*upscaling_factor].flatten()
                    const = sample[0]['hi_res_inputs'][k,
                                                       idx*upscaling_factor:(idx+era_chunk_width)*upscaling_factor,
                                                       idy*upscaling_factor:(idy+era_chunk_width)*upscaling_factor,
                                                       :].flatten()
                    era = sample[0]['lo_res_inputs'][k,
                                                     idx:idx+era_chunk_width,
                                                     idy:idy+era_chunk_width,
                                                     :].flatten()
                    feature = {
                        'generator_input': _float_feature(era),
                        'constants': _float_feature(const),
                        'generator_output': _float_feature(nimrod)
                    }
                    features = tf.train.Features(feature=feature)
                    example = tf.train.Example(features=features)
                    example_to_string = example.SerializeToString()
                    clss = min(int(np.floor(((nimrod > 0.1).mean()*num_class))), num_class-1)  # all class binning is here!
                    fle_hdles[clss].write(example_to_string)
        for fh in fle_hdles:
            fh.close()



def save_dataset(tfrecords_dataset, flename, max_batches=None):

    assert return_dic, "Only works with return_dic=True"
    flename=f"{records_folder}/{flename}"
    fle_hdle =  tf.io.TFRecordWriter(flename)            
    for ii, sample in enumerate(tfrecords_dataset):
        print(ii)
        if max_batches is not None:
            if ii == max_batches:
                break
        for k in range(sample[1]['output'].shape[0]):
            feature = {
                'generator_input': _float_feature(sample[0]['lo_res_inputs'][k,...].numpy().flatten()),
                'constants': _float_feature(sample[0]['hi_res_inputs'][k,...].numpy().flatten()),
                'generator_output': _float_feature(sample[1]['output'][k,...].numpy().flatten())
            }
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            example_to_string = example.SerializeToString()
            fle_hdle.write(example_to_string)
    fle_hdle.close()
    return
