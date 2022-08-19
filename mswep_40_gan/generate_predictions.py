
import numpy as np
from tfrecords_generator_ifs import create_fixed_dataset
import setupmodel
from noise import NoiseGenerator

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
                    padding=None,
                    noise_channels=None,
                    latent_variables=None,
                    predict_year=2019,
                    predict_full_image=True,
                    # ensemble_members=100,
                    gcm = False,
                    # plot_ecpoint=True,
                    ):
        
    # define initial variables
    print('generating predictions...')
    # downsample = True
    input_channels = 1
    noise_channels = 4
    batch_size = 2560
    # batch_size = 1
    # TODO: highest batch size possible
    num_images = 150
    num_images,_,_ = np.load('/user/work/al18709/tc_data_mswep/valid_X.npy').shape
    # num_images = 1000
    # num_images,_,_ = np.load('/user/work/al18709/tc_data_mswep/train_X.npy').shape
    print('number of images: ',num_images)

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
                                   padding=padding,
                                   noise_channels=noise_channels,
                                   latent_variables=latent_variables)

    # load appropriate dataset 
    # plot_label = 'small'
    # set initial variables
    mode = 'extreme_valid'
    mode = 'validation'
    # mode = 'cmip'
    # mode = 'train'
    if gcm == True:
        mode = 'gcm'
    
    # load relevant data
    data_predict = create_fixed_dataset(predict_year,
                                        batch_size=batch_size,
                                        downsample=False,
                                        mode = mode)

    
    # load model weights from main file
    print('checkpoint = ',checkpoint)
    if checkpoint == 'opt':
        gen_weights_file = "/user/home/al18709/work/dsrnngan/logs/models-gen_weights.h5"
    else:
        gen_weights_file = "/user/home/al18709/work/dsrnngan/logs/models/gen_weights-%s.h5" % checkpoint

    vaegan = False
    if vaegan:
        gen_weights_file = "/user/home/al18709/work/vaegan/logs/models-gen_weights.h5"
    
    gen_weights_file = "/user/home/al18709/work/dsrnngan/logs/models-gen_weights.h5"
    model.gen.built = True
    model.gen.load_weights(gen_weights_file)


    # define initial variables
    pred = []
    seq_real = []
    low_res_inputs = []
    data_pred_iter = iter(data_predict)
    
    # loop through images and get model to predict them
    for i in range(num_images):
        print('image ',i)
        inputs, outputs = next(data_pred_iter)
        img_real = outputs
        img_pred = []       
        noise_shape = inputs[0,...,0].shape + (noise_channels,)
        noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size) # does noise gen need to be outside of the for loop?
        
        # generate ensemble noise channels
        # nn = noise_gen()
        # nn *= 1.0
        # nn -= 0.0


        # img_pred = np.array(data.denormalise(model.gen.predict([inputs,noise_gen()]))[...,0])
        # pred = data.denormalise(model.gen.predict([inputs,noise_gen()]))[0][0]
        # make prediction
        # img_pred = np.array(model.gen.predict([inputs,noise_gen()]))
        # print('nn: ',nn)
        # print('nn shape: ',nn.shape)
        # number of ensembles
        # img_pred = np.zeros((1,100,100,20))
        img_pred = np.zeros((batch_size,100,100,20))

        for j in range(20): #do 50 ensemble members
            # if gan
            # 
            if vaegan:
                noise_shape = np.array(inputs)[0, ..., 0].shape + (latent_variables,)
                noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
                mean, logvar = model.gen.encoder([inputs])
                pred_single = np.array(model.gen.decoder.predict([mean, logvar, noise_gen()]))[:,:,:,0]
            else:
                nn = noise_gen()
                pred_single = np.array(model.gen.predict([inputs,nn]))[:,:,:,0]
            print(pred_single.shape)
            img_pred[:,:,:,j] = pred_single
            print(img_pred.shape)
        

        print('img pred shape: ',img_pred.shape)
        # append to relevant array
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
    # print(seq_real)
    np.save('/user/home/al18709/work/gan_predictions_50/%s_real-%s.npy' % (mode,checkpoint),seq_real)
    np.save('/user/home/al18709/work/gan_predictions_50/%s_pred-%s.npy' % (mode,checkpoint),pred)
    np.save('/user/home/al18709/work/gan_predictions_50/%s_input-%s.npy' % (mode,checkpoint),low_res_inputs)








