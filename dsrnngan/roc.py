import os
import gc
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
from data import get_dates, all_ifs_fields


def plot_roc_curves(*,
                    mode,
                    arch,
                    log_folder,
                    weights_dir,
                    model_numbers,
                    problem_type,
                    filters_gen,
                    filters_disc,
                    noise_channels,
                    latent_variables,
                    padding,
                    predict_year,
                    predict_full_image,
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
        batch_size = 2  # this will stop your computer having a strop
        num_batches = 8
        ensemble_members = 40
    else:
        batch_size = 16
        num_batches = 50

    if mode == 'det':
        ensemble_members = 1  # in this case, only used for printing

    precip_values = np.array([0.01, 0.1, 1, 2, 5])

    # initialise model
    model = setupmodel.setup_model(mode=mode,
                                   arch=arch,
                                   input_channels=input_channels,
                                   filters_gen=filters_gen,
                                   filters_disc=filters_disc,
                                   noise_channels=noise_channels,
                                   latent_variables=latent_variables,
                                   padding=padding)

    # load appropriate dataset
    if predict_full_image:
        plot_label = 'large'
        dates = get_dates(predict_year)
        data_predict = DataGeneratorFull(dates=dates,
                                         ifs_fields=all_ifs_fields,
                                         batch_size=batch_size,
                                         log_precip=True,
                                         crop=True,
                                         shuffle=True,
                                         constants=True,
                                         hour='random',
                                         ifs_norm=True,
                                         downsample=downsample)

    if not predict_full_image:
        plot_label = 'small'
        data_predict = create_fixed_dataset(predict_year,
                                            batch_size=batch_size,
                                            downsample=downsample)

    auc_scores = []
    for model_number in model_numbers:
        gc.collect()
        print(f"calculating for model number {model_number}")
        gen_weights_file = os.path.join(weights_dir, "gen_weights-{:07d}.h5".format(model_number))

        if not os.path.isfile(gen_weights_file):
            print(gen_weights_file, "not found, skipping")
            continue

        print(gen_weights_file)
        model.gen.load_weights(gen_weights_file)
        model_label = str(model_number)

        y_true = {}
        y_score = {}
        gc.collect()  # for subsequent model_numbers after the first
        for value in precip_values:
            y_true[value] = []
            y_score[value] = []

        data_pred_iter = iter(data_predict)  # "restarts" data iterator
        for ii in range(num_batches):
            inputs, outputs = next(data_pred_iter)
            if predict_full_image:
                im_real = data.denormalise(np.array(outputs['output']))  # shape: batch_size x H x W
            else:
                im_real = data.denormalise(outputs['output'])[..., 0]

            pred_ensemble = []
            if mode == 'det':
                pred_ensemble.append(data.denormalise(model.gen.predict([inputs['lo_res_inputs'],
                                                                         inputs['hi_res_inputs']]))[..., 0])
            else:
                if mode == 'GAN':
                    noise_shape = inputs['lo_res_inputs'][0, ..., 0].shape + (noise_channels,)
                elif mode == 'VAEGAN':
                    noise_shape = inputs['lo_res_inputs'][0, ..., 0].shape + (latent_variables,)
                noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
                if mode == 'VAEGAN':
                    # call encoder once
                    mean, logvar = model.gen.encoder([inputs['lo_res_inputs'], inputs['hi_res_inputs']])
                for j in range(ensemble_members):
                    inputs['noise_input'] = noise_gen()
                    if mode == 'GAN':
                        pred_ensemble.append(data.denormalise(model.gen.predict([inputs['lo_res_inputs'],
                                                                                 inputs['hi_res_inputs'],
                                                                                 inputs['noise_input']]))[..., 0])
                    elif mode == 'VAEGAN':
                        dec_inputs = [mean, logvar, inputs['noise_input'], inputs['hi_res_inputs']]
                        pred_ensemble.append(data.denormalise(model.gen.decoder.predict(dec_inputs))[..., 0])

            # turn accumulated list into numpy array
            pred_ensemble = np.array(pred_ensemble)  # shape: ensemble_mem x batch_size x H x W

            # list is large, so force garbage collect
            gc.collect()

            # need to calculate averages each batch; can't store n_images x n_ensemble x H x W!
            for value in precip_values:
                # binary instance of truth > threshold
                # append an array of shape batch_size x H x W
                y_true[value].append((im_real > value))
                # check what proportion of pred > threshold
                # collapse over ensemble dim, so append an array also of shape batch_size x H x W
                y_score[value].append(np.mean(pred_ensemble > value, axis=0, dtype=np.single))

        for value in precip_values:
            y_true[value] = np.concatenate(y_true[value], axis=0)  # n_images x W x H
            gc.collect()

        for value in precip_values:
            y_score[value] = np.concatenate(y_score[value], axis=0)  # n_images x W x H
            gc.collect()

#         for value in precip_values:
#             # debug code for testing memory usage without generating samples
#             print("Generating random array", value)
#             y_true[value] = np.random.randint(0, 2, (256, 940, 940), dtype=np.bool)
#             y_score[value] = (np.random.randint(0, 101, (256, 940, 940))/100.0).astype(np.single)
#             gc.collect()

        fpr = []
        tpr = []
        roc_auc = []
        for value in precip_values:
            print("Computing ROC for", value)
            # Compute ROC curve and ROC area for each precip value
            fpr_pv, tpr_pv, _ = roc_curve(np.ravel(y_true[value]), np.ravel(y_score[value]), drop_intermediate=False)
            roc_auc_pv = auc(fpr_pv, tpr_pv)
            fpr.append(fpr_pv)
            tpr.append(tpr_pv)
            roc_auc.append(roc_auc_pv)
            gc.collect()

        auc_scores.append(np.array(roc_auc))

        # Plot all ROC curves
        plt.figure(figsize=(7, 5))
        lw = 2
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy']
        for i, color in zip(range(len(precip_values)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label=f"ROC curve for precip value {precip_values[i]} (area = %0.2f)" % roc_auc[i])

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve for {plot_input_title} problem, {ensemble_members} ensemble members, {batch_size*num_batches} images')
        plt.legend(loc="lower right")
        plt.savefig("{}/ROC-{}-{}-{}.pdf".format(log_folder, problem_type, plot_label, model_label), bbox_inches='tight')
        plt.show()

    auc_scores = np.transpose(np.array(auc_scores))
    plt.figure(figsize=(8, 5))

    colors = ['darkturquoise', 'teal', 'dodgerblue', 'navy', 'purple']
    for i, color in zip(range(len(precip_values)), colors):
        plt.plot(model_numbers, auc_scores[i], color=color, lw=lw,
                 label=f"AUC values for precip_values {precip_values[i]}")

    plt.ylim([0, 1.0])
    plt.xlabel('Checkpoint number')
    plt.ylabel('Area under ROC curve')
    plt.title('AUC values for varying precip thresholds')
    plt.legend(loc="best")
    plt.savefig("{}/AUC-{}-{}.pdf".format(log_folder, problem_type, plot_label), bbox_inches='tight')
    plt.show()

    # ecPoint
    # requires a different data generator
    if plot_ecpoint:
        dates = get_dates(predict_year)
        if predict_full_image:
            batch_size = 4
        else:
            raise RuntimeError('Data generator for benchmarks not implemented for small images')
        data_benchmarks = DataGeneratorFull(dates=dates,
                                            ifs_fields=ecpoint.ifs_fields,
                                            batch_size=batch_size,
                                            log_precip=False,
                                            crop=True,
                                            shuffle=True,
                                            constants=True,
                                            hour="random",
                                            ifs_norm=False,
                                            downsample=downsample)

        # generate predictions
        # store preds
        seq_ecpoint = []
        # store ground truth images for comparison
        seq_real_ecpoint = []

        data_benchmarks_iter = iter(data_benchmarks)
        inp, outp = next(data_benchmarks_iter)
        # store GT data
        seq_real_ecpoint.append(data.denormalise(outp['output']))

        # store mean ecPoint prediction
        # seq_ecpoint.append(np.mean(benchmarks.ecpointPDFmodel(inputs['generator_input']), axis=-1))
        seq_ecpoint.append(benchmarks.ecpointPDFmodel(inp['lo_res_inputs']))

        seq_ecpoint = np.array(seq_ecpoint)
        seq_real_ecpoint = np.array(seq_real_ecpoint)

        fpr_ecpoint = []
        tpr_ecpoint = []
        roc_auc_ecpoint = []
        for value in precip_values:
            # produce y_true
            # binary instance of truth > threshold
            y_true_ecpoint = np.squeeze(1*(seq_real_ecpoint > value), axis=0)
            # produce y_score
            # check if pred > threshold
            y_score_ecpoint = np.squeeze(np.mean(1*(seq_ecpoint > value), axis=-1), axis=0)
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
        plt.figure(figsize=(7, 5))
        lw = 2
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy']
        for i, color in zip(range(len(precip_values)), colors):
            plt.plot(fpr_ecpoint[i], tpr_ecpoint[i], color=color, lw=lw,
                     label=f"ROC curve for precip value {precip_values[i]} (area = %0.2f)" % roc_auc_ecpoint[i])

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve for ecPoint approach, batch size {batch_size}')
        plt.legend(loc="lower right")
        plt.savefig("{}/ROC-ecPoint-{}-{}.pdf".format(log_folder,
                                                      problem_type,
                                                      plot_label), bbox_inches='tight')
