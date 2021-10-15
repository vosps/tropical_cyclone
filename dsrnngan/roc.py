import os
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tfrecords_generator_ifs import create_fixed_dataset
from data_generator_ifs import DataGenerator as DataGeneratorFull
import setupmodel
import data
from noise import NoiseGenerator
from data import get_dates

def plot_roc_curves(*,
                    mode,
                    arch,
                    log_folder, 
                    weights_dir,
                    model_numbers=None,
                    problem_type='normal',
                    filters_gen=None,
                    filters_disc=None,
                    noise_channels=None,
                    latent_variables=None,
                    predict_year=2019,
                    predict_full_image=True,
                    ensemble_members=100,
                    plot_ecpoint=True):
       
    if problem_type == "normal":
        downsample = False
        plot_input_title = 'IFS'
        input_channels = 9
        noise_channels = 4
    elif problem_type == "superresolution":
        downsample = True
        plot_input_title = 'Superresolution'
        input_channels = 1
        noise_channels = 2
    else:
        raise Exception("no such problem type, try again!")
    
    if predict_full_image:
        batch_size = 2 # this will stop your computer having a strop
        num_images = 10
    else:
        batch_size = 16
        num_images = 50
    
    precip_values = np.array([0.01, 0.1, 1, 2, 5])
   
    ## initialise model
    model = setupmodel.setup_model(mode=mode,
                                   arch=arch,
                                   input_channels=input_channels,
                                   filters_gen=filters_gen,
                                   filters_disc=filters_disc,
                                   noise_channels=noise_channels,
                                   latent_variables=latent_variables)
    
    # load appropriate dataset
    if predict_full_image:
        plot_label = 'large'
        all_ifs_fields = ['tp','cp' ,'sp' ,'tisr','cape','tclw','tcwv','u700','v700']
        dates=get_dates(predict_year)
        data_predict = DataGeneratorFull(dates=dates,
                                         ifs_fields=all_ifs_fields,
                                         batch_size=batch_size,
                                         log_precip=True,
                                         crop=True,
                                         shuffle=False,
                                         constants=True,
                                         hour=0,
                                         ifs_norm=True,
                                         downsample=downsample)
    
        
    if not predict_full_image:
        plot_label = 'small'
        data_predict = create_fixed_dataset(predict_year,
                                            batch_size=batch_size,
                                            downsample=downsample)

    auc_scores = []
    for model_number in model_numbers:
        print(f"calculating for model number {model_number}")
        gen_weights_file = os.path.join(weights_dir, "gen_weights-{:07d}.h5".format(model_number))
        if not os.path.isfile(gen_weights_file):
            print(gen_weights_file, "not found, skipping")
        else:
            print(gen_weights_file)
            model.gen.load_weights(gen_weights_file)
        model_label = str(model_number)
    
        pred = []
        seq_real = []
    
        data_pred_iter = iter(data_predict)
        for i in range(num_images):
            inputs, outputs = next(data_pred_iter)
            if predict_full_image:
                im_real = data.denormalise(np.array(outputs['generator_output']))
            else:
                im_real = data.denormalise(outputs['generator_output'])[...,0]        
            if mode == 'det':
                pred_ensemble = []
                ensemble_members = 1 #can't generate an ensemble with deterministic method
                pred_ensemble.append(data.denormalise(model.gen.predict([inputs['generator_input'], 
                                                                         inputs['constants']]))[...,0])
            else:
                pred_ensemble = []
                noise_shape = inputs['generator_input'][0,...,0].shape + (noise_channels,)
                if mode == 'VAEGAN':
                    # call encoder once
                    mean, logvar = model.gen.encoder([inputs['generator_input'], inputs['constants']])       
                for j in range(ensemble_members):
                    noise = NoiseGenerator(noise_shape, batch_size=batch_size)
                    inputs['noise_input'] = noise()
                    if mode == 'GAN':
                        pred_ensemble.append(data.denormalise(model.gen.predict([inputs['generator_input'], 
                                                                                 inputs['constants'], 
                                                                                 inputs['noise_input']]))[...,0])
                    elif mode == 'VAEGAN':
                        dec_inputs = [mean, logvar, inputs['noise_input'], inputs['constants']]
                        pred_ensemble.append(data.denormalise(model.gen.decoder.predict(dec_inputs))[...,0])
                pred_ensemble = np.array(pred_ensemble)

            if i == 0:
                seq_real.append(im_real)
                pred.append(pred_ensemble)
                seq_real = np.array(seq_real)
                pred = np.squeeze(np.array(pred))
            else:
                seq_real = np.concatenate((seq_real, np.expand_dims(im_real, axis=0)), axis=1)
                pred = np.concatenate((pred, pred_ensemble), axis=1)
    
        seq_real = np.array(seq_real)
        pred = np.array(pred)
        
        fpr = []
        tpr = []
        roc_auc = []
        for value in precip_values:
            # produce y_true
            ## binary instance of truth > threshold
            y_true = np.squeeze(1*(seq_real > value), axis = 0)
            # produce y_score
            ## check if pred > threshold 
            y_score = np.mean(1*(pred > value), axis=0)
            # flatten matrices
            y_true = np.ravel(y_true)
            y_score = np.ravel(y_score)
            # Compute ROC curve and ROC area for each precip value
            fpr_pv, tpr_pv, _ = roc_curve(y_true, y_score)
            roc_auc_pv = auc(fpr_pv, tpr_pv)
            fpr.append(fpr_pv)
            tpr.append(tpr_pv)
            roc_auc.append(roc_auc_pv)
        auc_scores.append(np.array(roc_auc))
        
        # Plot all ROC curves
        plt.figure(figsize=(7,5))
        lw = 2
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy']
        for i, color in zip(range(len(precip_values)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label=f"ROC curve for precip value {precip_values[i]} (area = %0.2f)" %roc_auc[i])
    
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve for {plot_input_title} problem, {ensemble_members} ensemble members, {batch_size*num_images} images')
        plt.legend(loc="lower right")
        plt.savefig("{}/ROC-{}-{}-{}.pdf".format(log_folder, problem_type, plot_label, model_label), bbox_inches='tight')
        plt.show()
    
    auc_scores=np.transpose(np.array(auc_scores))
    plt.figure(figsize=(8,5))
    
    colors = ['darkturquoise', 'teal', 'dodgerblue', 'navy', 'purple']
    for i, color in zip(range(len(precip_values)), colors):
        plt.plot(model_numbers, auc_scores[i], color=color, lw=lw,
                 label=f"AUC values for precip_values {precip_values[i]}")
    
    plt.ylim([0, 1.0])
    plt.xlabel('Epoch number')
    plt.ylabel('Area under ROC curve')
    plt.title('AUC values for varying precip thresholds')
    plt.legend(loc="best")
    plt.savefig("{}/AUC-{}-{}.pdf".format(log_folder, problem_type, plot_label), bbox_inches='tight')
    plt.show()

    ##ecPoint
    ## requires a different data generator
    if plot_ecpoint:
        dates=get_dates(predict_year)
        data_benchmarks = DataGeneratorFull(dates=dates,
                                            ifs_fields=ecpoint.ifs_fields,
                                            batch_size=batch_size,
                                            log_precip=False,
                                            crop=True,
                                            shuffle=False,
                                            constants=True,
                                            hour=0,
                                            ifs_norm=False,
                                            downsample=downsample)
    
        # generate predictions
        ## store preds
        seq_ecpoint = []
        ## store ground truth images for comparison
        seq_real_ecpoint = []
    
    
        data_benchmarks_iter = iter(data_benchmarks)
        (inp,outp) = next(data_benchmarks_iter)
        ## store GT data
        seq_real_ecpoint.append(data.denormalise(outp['generator_output']))
        
        ## store mean ecPoint prediction
        #seq_ecpoint.append(np.mean(benchmarks.ecpointPDFmodel(inputs['generator_input']), axis=-1))
        seq_ecpoint.append(benchmarks.ecpointPDFmodel(inp['generator_input']))
    
        seq_ecpoint = np.array(seq_ecpoint)
        seq_real_ecpoint = np.array(seq_real_ecpoint)
    
        fpr_ecpoint = []
        tpr_ecpoint = []
        roc_auc_ecpoint = []
        for value in precip_values:
            # produce y_true
            ## binary instance of truth > threshold
            y_true_ecpoint = np.squeeze(1*(seq_real_ecpoint > value), axis=0)
            # produce y_score
            ## check if pred > threshold 
            y_score_ecpoint = np.squeeze(np.mean(1*(seq_ecpoint > value), axis=-1), axis = 0)
            # flatten matrices
            y_true_ecpoint = np.ravel(y_true_ecpoint)
            y_score_ecpoint = np.ravel(y_score_ecpoint)
            # Compute ROC curve and ROC area for each precip value
            fpr_pv_ecpoint, tpr_pv_ecpoint, _ = roc_curve(y_true_ecpoint, y_score_ecpoint)
            roc_auc_pv_ecpoint = auc(fpr_pv_ecpoint, tpr_pv_ecpoint)
            fpr_ecpoint.append(fpr_pv_ecpoint)
            tpr_ecpoint.append(tpr_pv_ecpoint)
            roc_auc_ecpoint.append(roc_auc_pv_ecpoint)
    
        # Plot all ROC curves
        plt.figure(figsize=(7,5))
        lw = 2
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy']
        for i, color in zip(range(len(precip_values)), colors):
            plt.plot(fpr_ecpoint[i], tpr_ecpoint[i], color=color, lw=lw,
                     label=f"ROC curve for precip value {precip_values[i]} (area = %0.2f)" %roc_auc_ecpoint[i])
    
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve for ecPoint approach, batch size {batch_size}')
        plt.legend(loc="lower right")
        plt.savefig("{}/ROC-ecPoint-{}-{}.pdf".format(log_folder,problem_type,
                                                      plot_label), bbox_inches='tight')
