import sys
sys.path.append('/ppdata/lucy-cGAN/dsrnngan/')
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tfrecords_generator_ifs import create_fixed_dataset
from data_generator_ifs import DataGenerator as DataGeneratorFull
import setupmodel
import data
import ecpoint
import benchmarks
from noise import NoiseGenerator
from data import get_dates

# input parameters
log_path = '/ppdata/lucy-cGAN/logs/IFS/gen_256_disc_512/noise_4/weights_4x/'
model_number = '0198400'
weights_fn = log_path + "/gen_weights-" +  model_number + ".h5"
plot_ecpoint = False
mode = 'GAN'
weights = None
predict_year = 2019
filters_disc = 512
filters_gen = 256
latent_variables = 1
problem_type = 'normal'
predict_full_image = True
ensemble_members = 100
precip_values = np.array([0.01, 0.1, 1, 2, 5])

if predict_full_image:
    batch_size = 2
    num_images = 10
else:
    batch_size = 16
    num_images = 50

if problem_type == "normal":
    downsample = False
    plot_input_title = 'IFS'
    input_channels = 9
    noise_channels = 4
elif problem_type == "superresolution":
    downsample = True
    plot_input_title = "Superresolution"
    input_channels = 1
    noise_channels = 2
else:
    raise Exception("no such problem type, try again!")


## initialise model
model = setupmodel.setup_model(mode,
                               downsample=downsample, 
                               weights=weights,
                               input_channels=input_channels,
                               batch_size=batch_size,
                               filters_gen=filters_gen, 
                               filters_disc=filters_disc,
                               noise_channels=noise_channels, 
                               latent_variables=latent_variables)
# load weights
model.gen.load_weights(weights_fn)

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

# generate predictions
## store preds
pred = []
## store ground truth images for comparison
seq_real = []

data_pred_iter = iter(data_predict)
for i in range(num_images):
    print(f"image number {i+1} of {num_images}")
    (inputs,outputs) = next(data_pred_iter)
    if predict_full_image:
        im_real = data.denormalise(np.array(outputs['generator_output']))
    elif predict_full_image:
        im_real = data.denormalise(outputs['generator_output'])[...,0]
        
    if mode == 'det':
        num_samples = 1 #can't generate an ensemble with deterministic method
        pred.append(data.denormalise(model.gen.predict(inputs))[...,0])
    else:
        pred_ensemble = []
        noise_shape = inputs['generator_input'][0,...,0].shape + (noise_channels,)
        if mode == 'VAEGAN':
            # call encoder once
            mean, logvar = model.gen.encoder([inputs['generator_input'], inputs['constants']])       
        for j in range(ensemble_members):
            inputs['noise_input'] = NoiseGenerator(noise_shape, batch_size=batch_size)
            if mode == 'GAN':
                pred_ensemble.append(data.denormalise(model.gen.predict(inputs))[...,0])
            elif mode == 'VAEGAN':
                dec_inputs = [mean, logvar, inputs['noise_input'], inputs['constants']]
                pred_ensemble.append(data.denormalise(model.gen.decoder.predict(dec_inputs))[...,0])
            pred_ensemble = np.array(pred_ensemble)
        pred.append(pred_ensemble)    

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
plt.savefig("{}/ROC-{}-{}-{}.pdf".format(log_path,problem_type,plot_label,model_number), bbox_inches='tight')
plt.close()

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
    plt.savefig("{}/ROC-ecPoint-{}-{}.pdf".format(log_path,problem_type,
                                                  plot_label), bbox_inches='tight')
