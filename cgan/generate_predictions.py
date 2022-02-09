# import os
import numpy as np
# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt
from tfrecords_generator_ifs import create_fixed_dataset
# from data_generator_ifs import DataGenerator as DataGeneratorFull
import setupmodel
# import data
# import ecpoint
# import benchmarks
from noise import NoiseGenerator
# from data import get_dates
# from tensorflow.keras.layers import Input

def generate_predictions(*,
                    mode,
                    checkpoint,
                    arch,
                    log_folder, 
                    # weights_dir,
                    # model_numbers=None,
                    # problem_type='normal',
                    filters_gen=None,
                    filters_disc=None,
                    noise_channels=None,
                    latent_variables=None,
                    predict_year=2019,
                    predict_full_image=True,
                    # ensemble_members=100,
                    gcm = False,
                    # plot_ecpoint=True,
                    ):
        
    # define initial variables
    # downsample = True
    input_channels = 1
    noise_channels = 4
    # batch_size = 32
    batch_size = 1
    num_images = 150
    num_images,_,_ = np.load('/user/work/al18709/tc_data_mswep/valid_X.npy').shape
    print(num_images)

    if gcm == True:
        batch_size = 1
        num_images = 5
    

    # initialise model
    print(mode)
    model = setupmodel.setup_model(mode=mode,
                                   arch=arch,
                                   input_channels=input_channels,
                                   filters_gen=filters_gen,
                                   filters_disc=filters_disc,
                                   noise_channels=noise_channels,
                                   latent_variables=latent_variables)

    # load appropriate dataset 
    # plot_label = 'small'
    mode = 'validation'
    
    # mode = 'train'
    # mode = 'extreme_valid'
    if gcm == True:
        mode = 'gcm'
    
    data_predict = create_fixed_dataset(predict_year,
                                        batch_size=batch_size,
                                        downsample=False,
                                        mode = mode)

    
    # load model weights from main file
    print('checkpoint = ',checkpoint)
    if checkpoint == 'opt':
        gen_weights_file = "/user/home/al18709/work/cgan/logs/models-gen_weights.h5"
    else:
        gen_weights_file = "/user/home/al18709/work/cgan/logs/models/gen_weights-%s.h5" % checkpoint
    model.gen.load_weights(gen_weights_file)

    # define initial variables
    pred = []
    seq_real = []
    low_res_inputs = []
    data_pred_iter = iter(data_predict)
    
    # loop through images and get model to predict them
    for i in range(num_images):
        print('image ',i,end='\r')
        inputs, outputs = next(data_pred_iter)
        img_real = outputs
        img_pred = []       
        noise_shape = inputs[0,...,0].shape + (noise_channels,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        
        # do for 1 ensemble member to start?
        # img_pred = np.array(data.denormalise(model.gen.predict([inputs,noise_gen()]))[...,0])
        # make prediction
        img_pred = np.array(model.gen.predict([inputs,noise_gen()]))
        
        # pred = data.denormalise(model.gen.predict([inputs,noise_gen()]))[0][0]
        
        if i == 0:
            seq_real.append(img_real)
            pred.append(img_pred)
            low_res_inputs.append(inputs)
            seq_real = np.array(seq_real)
            pred = np.array(pred)
            low_res_inputs = np.array(low_res_inputs)
            
        else:
            seq_real = np.concatenate((seq_real, np.expand_dims(img_real, axis=0)), axis=1)
            pred = np.concatenate((pred, np.expand_dims(img_pred, axis=0)), axis=1)
            low_res_inputs = np.concatenate((low_res_inputs,np.expand_dims(inputs,axis=0)),axis=1)
            seq_real = np.array(seq_real)
            pred = np.array(pred)
            low_res_inputs = np.array(low_res_inputs)
            
        
    print(mode)
    print(seq_real.shape)
    print(pred.shape)
    print(low_res_inputs.shape)
    print(seq_real)
    np.save('/user/home/al18709/work/cgan_predictions/%s_real-%s.npy' % (mode,checkpoint),seq_real)
    np.save('/user/home/al18709/work/cgan_predictions/%s_pred-%s.npy' % (mode,checkpoint),pred)
    np.save('/user/home/al18709/work/cgan_predictions/%s_input-%s.npy' % (mode,checkpoint),low_res_inputs)









