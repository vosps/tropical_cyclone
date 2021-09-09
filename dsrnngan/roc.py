import sys
sys.path.append('/ppdata/lucy-cGAN/dsrnngan/')
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tfrecords_generator_ifs import create_fixed_dataset
from data_generator_ifs import DataGenerator as DataGeneratorFull
import train
import data
import ecpoint
import benchmarks
from noise import noise_generator
from data import get_dates

# input parameters
log_path = '/ppdata/lucy-cGAN/logs/IFS/gen_128_disc_512/noise_4/weights_4x/'
application = 'IFS'
model_number = '0048000'
weights_fn = log_path + "/gen_weights-" + application + "-" +  model_number + ".h5"
plot_ecpoint = False

train_years = None
val_years = None
val_size = None
weights = None
predict_year = 2019
constant_fields = 2
filters_disc = 512
filters_gen = 128
lr_disc = 1e-5
lr_gen = 1e-5
problem_type = 'normal'
predict_full_image = True
ensemble_members = 100
precip_values = np.array([0.01, 0.1, 1, 2, 5])

if problem_type == "normal":
    downsample = False
    plot_input_title = 'IFS'
    input_channels = 9
    batch_size = 2
    noise_channels = 4
elif problem_type == "easy":
    downsample = True
    plot_input_title = 'Downscaled'
    input_channels = 1
    batch_size = 64
    noise_channels = 2
else:
    raise Exception("no such problem type, try again!")

# initialize GAN model
(wgan) = train.setup_gan(train_years, 
                         val_years, 
                         val_size=val_size, 
                         downsample=downsample,
                         weights=weights,
                         input_channels=input_channels,
                         constant_fields=constant_fields,
                         steps_per_epoch=50, 
                         batch_size=batch_size,
                         filters_gen=filters_gen,
                         filters_disc=filters_disc,
                         noise_channels=noise_channels,  
                         lr_disc=lr_disc, 
                         lr_gen=lr_gen)
# load weights
gen = wgan.gen
gen.load_weights(weights_fn)

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
(inputs,outputs) = next(data_pred_iter)
    
## make sure ground truth image has correct dimensions
if predict_full_image == True:
    seq_real.append(data.denormalise(np.array(outputs['generator_output'])))
elif predict_full_image == False:
     seq_real.append(data.denormalise(outputs['generator_output'])[...,0])
## generate ensemble members
for j in range(ensemble_members):
    ## retrieve noise dimensions from input condition
    noise_dim = inputs['generator_input'][0,...,0].shape + (noise_channels,)
    inputs['noise_input'] = noise_generator(noise_dim, batch_size=batch_size)
    ## store denormalised predictions
    pred.append(data.denormalise(gen.predict(inputs))[...,0])
    print(j)

pred = np.array(pred)
seq_real = np.array(seq_real)

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
plt.title(f'ROC curve for {plot_input_title} problem, {ensemble_members} ensemble members, batch size {batch_size}')
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
