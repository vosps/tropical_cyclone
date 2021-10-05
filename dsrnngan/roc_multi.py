import sys
sys.path.append('/ppdata/lucy-cGAN/dsrnngan/')
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from tensorflow.python.keras.utils import generic_utils
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
log_path = '/ppdata/lucy-cGAN/logs/IFS/gen_256_disc_512/noise_4/weights_4x/'
precip_values = np.array([0.01, 0.1, 1, 2, 5])
model_numbers = ['0006400', '0048000', '0160000', '0169600', '0198400', '0240000', '0262400', '0275200', '0313600', '0320000']
model_number = np.array([6400, 48000, 160000, 169600, 198400, 240000, 262400, 275200, 313600, 320000])

train_years = None
val_years = None
val_size = None
application = 'IFS'
weights = None
predict_year = 2019
constant_fields = 2
filters_disc = 512
filters_gen = 256
lr_disc = 1e-5
lr_gen = 1e-5
problem_type = 'normal'
predict_full_image = False
ensemble_members = 100

if problem_type == "normal":
    downsample = False
    plot_input_title = 'IFS'
    input_channels = 9
    noise_channels = 4
elif problem_type == "easy":
    downsample = True
    plot_input_title = 'Downscaled'
    input_channels = 1
    noise_channels = 2
else:
    raise Exception("no such problem type, try again!")

if predict_full_image:
    batch_size = 2
    num_images = 50
else:
    batch_size = 16
    num_images = 50

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
for model in model_numbers:
    print(f"calculating for model number {model}")
    # load weights
    weights_fn = log_path + "/gen_weights-" + application + "-" +  model + ".h5"
    wgan.gen.load_weights(weights_fn)
    
    # generate predictions
    ## store preds
    pred = []
    ## store ground truth images for comparison
    seq_real = []

    data_pred_iter = iter(data_predict)
    for i in range(num_images):
        (inputs,outputs) = next(data_pred_iter)

        ## make sure ground truth image has correct dimensions
        if predict_full_image == True:
            im_real = data.denormalise(np.array(outputs['generator_output']))
        elif predict_full_image == False:
            im_real = data.denormalise(outputs['generator_output'])[...,0]
        
        ## generate ensemble members
        pred_ensemble = []
        for j in range(ensemble_members):
            ## retrieve noise dimensions from input condition
            noise_dim = inputs['generator_input'][0,...,0].shape + (noise_channels,)
            inputs['noise_input'] = noise_generator(noise_dim, batch_size=batch_size)
            ## store denormalised predictions
            pred_ensemble.append(data.denormalise(wgan.gen.predict(inputs))[...,0])
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
    plt.savefig("{}/ROC-{}-{}-{}.pdf".format(log_path,problem_type,plot_label,model), bbox_inches='tight')
    plt.show()

auc_scores=np.transpose(np.array(auc_scores))
plt.figure(figsize=(8,5))

colors = ['darkturquoise', 'teal', 'dodgerblue', 'navy', 'purple']
for i, color in zip(range(len(precip_values)), colors):
    plt.plot(model_numbers, auc_scores[i], color=color, lw=lw,
             label=f"AUC values for precip_values {precip_values[i]}")

# plt.xlim([0.01, 5.0])
plt.ylim([0, 1.0])
plt.xlabel('Epoch number')
plt.ylabel('Area under ROC curve')
plt.title(f'AUC values for varying precip thresholds')
plt.legend(loc="best")
plt.savefig("{}/AUC-{}-{}.pdf".format(log_path,problem_type,plot_label), bbox_inches='tight')
plt.show()
