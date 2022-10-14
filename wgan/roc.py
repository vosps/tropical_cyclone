import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gc
import pickle
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
# import matplotlib.pyplot as plt
from tfrecords_generator_ifs import create_fixed_dataset
from data_generator_ifs import DataGenerator as DataGeneratorFull
import setupmodel
import data
# import ecpoint
import benchmarks
from noise import NoiseGenerator
# from data import get_dates, all_ifs_fields
from data import all_ifs_fields
from pooling import pool


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
        input_channels = 9
    elif problem_type == "superresolution":
        downsample = True
        input_channels = 1
    else:
        raise Exception("no such problem type, try again!")

    if predict_full_image:
        batch_size = 2  # this will stop your computer having a strop
        num_batches = 64
        ensemble_members = 100
    else:
        batch_size = 16
        num_batches = 50

    if mode == 'det':
        ensemble_members = 1  # in this case, only used for printing

    precip_values = np.array([0.1, 0.5, 2.0, 5.0])

    pooling_methods = ['no_pooling', 'max_4', 'max_16', 'max_10_no_overlap', 'avg_4', 'avg_16', 'avg_10_no_overlap']
    # full list: ['no_pooling', 'max_4', 'max_16', 'max_10_no_overlap', 'avg_4', 'avg_16', 'avg_10_no_overlap']

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
        data_predict = create_fixed_dataset(predict_year,
                                            batch_size=batch_size,
                                            downsample=downsample)

    if plot_ecpoint:
        if not predict_full_image:
            raise RuntimeError('Data generator for benchmarks not implemented for small images')
        # requires a different data generator with different fields and no ifs_norm
        data_benchmarks = DataGeneratorFull(dates=dates,
                                            ifs_fields=ecpoint.ifs_fields,
                                            batch_size=batch_size,
                                            log_precip=False,
                                            crop=True,
                                            shuffle=True,
                                            constants=True,
                                            hour="random",
                                            ifs_norm=False)

    auc_scores_roc = {}  # will only contain GAN AUCs; used for "progress vs time" plot
    auc_scores_pr = {}
    for method in pooling_methods:
        auc_scores_roc[method] = []
        auc_scores_pr[method] = []

    # tidier to iterate over GAN checkpoints and ecPoint using joint code
    model_numbers_ec = model_numbers.copy()
    if plot_ecpoint:
        model_numbers_ec.extend(["ecPoint-nocorr", "ecPoint-partcorr", "ecPoint-fullcorr", "ecPoint-mean", "constupsc"])

    for model_number in model_numbers_ec:
        print(f"calculating for model number {model_number}")
        if model_number in model_numbers:
            # only load weights for GAN, not ecPoint
            gen_weights_file = os.path.join(weights_dir, "gen_weights-{:07d}.h5".format(model_number))
            if not os.path.isfile(gen_weights_file):
                print(gen_weights_file, "not found, skipping")
                continue
            print(gen_weights_file)
            model.gen.load_weights(gen_weights_file)

        y_true = {}
        y_score = {}
        for method in pooling_methods:
            y_true[method] = {}
            y_score[method] = {}
            for value in precip_values:
                y_true[method][value] = []
                y_score[method][value] = []

        if model_number in model_numbers:
            data_pred_iter = iter(data_predict)  # "restarts" GAN data iterator
        else:
            data_benchmarks_iter = iter(data_benchmarks)  # ecPoint data iterator

        for ii in range(num_batches):
            print(ii, num_batches)

            if model_number in model_numbers:
                # GAN, not ecPoint
                inputs, outputs = next(data_pred_iter)
                # need to denormalise
                if predict_full_image:
                    im_real = data.denormalise(outputs['output']).astype(np.single)  # shape: batch_size x H x W
                else:
                    im_real = (data.denormalise(outputs['output'])[..., 0]).astype(np.single)
            else:
                # ecPoint, no need to denormalise
                inputs, outputs = next(data_benchmarks_iter)
                im_real = outputs['output'].astype(np.single)  # shape: batch_size x H x W

            if model_number in model_numbers:
                # get GAN predictions
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
                        # mean, logvar = model.gen.encoder([inputs['lo_res_inputs'], inputs['hi_res_inputs']])
                        mean, logvar = model.gen.encoder([inputs['lo_res_inputs']])
                    for j in range(ensemble_members):
                        inputs['noise_input'] = noise_gen()
                        if mode == 'GAN':
                            pred_ensemble.append(data.denormalise(model.gen.predict([inputs['lo_res_inputs'],
                                                                                     inputs['hi_res_inputs'],
                                                                                     inputs['noise_input']]))[..., 0])
                        elif mode == 'VAEGAN':
                            # dec_inputs = [mean, logvar, inputs['noise_input'], inputs['hi_res_inputs']]
                            dec_inputs = [mean, logvar, inputs['noise_input']]
                            pred_ensemble.append(data.denormalise(model.gen.decoder.predict(dec_inputs))[..., 0])

                # turn accumulated list into numpy array
                pred_ensemble = np.stack(pred_ensemble, axis=1)  # shape: batch_size x ens x H x W

                # list is large, so force garbage collect
                gc.collect()
            else:
                # pred_ensemble will be batch_size x ens x H x W
                if model_number == "ecPoint-nocorr":
                    pred_ensemble = benchmarks.ecpointmodel(inputs['lo_res_inputs'],
                                                            ensemble_size=ensemble_members,
                                                            data_format="channels_first")
                elif model_number == "ecPoint-partcorr":
                    pred_ensemble = benchmarks.ecpointboxensmodel(inputs['lo_res_inputs'],
                                                                  ensemble_size=ensemble_members,
                                                                  data_format="channels_first")
                elif model_number == "ecPoint-fullcorr":
                    # this has ens=100 every time
                    pred_ensemble = benchmarks.ecpointPDFmodel(inputs['lo_res_inputs'],
                                                               data_format="channels_first")
                elif model_number == "ecPoint-mean":
                    # this has ens=100 every time
                    pred_ensemble = benchmarks.ecpointPDFmodel(inputs['lo_res_inputs'],
                                                               data_format="channels_first")
                    pred_ensemble = np.mean(pred_ensemble, axis=1)
                    pred_ensemble = np.expand_dims(pred_ensemble, 1)  # batch_size x 1 x H x W
                elif model_number == "constupsc":
                    pred_ensemble = np.expand_dims(inputs['lo_res_inputs'][:, :, :, 1], 1)
                    pred_ensemble = np.repeat(np.repeat(pred_ensemble, 10, axis=-1), 10, axis=-2)
                else:
                    raise RuntimeError('Unknown model_number {}' % model_number)

            # need to calculate averages each batch; can't store n_images x n_ensemble x H x W!
            for method in pooling_methods:
                if method == 'no_pooling':
                    im_real_pooled = im_real.copy()
                    pred_ensemble_pooled = pred_ensemble.copy()
                else:
                    # im_real only has 3 dims but the pooling needs 4,
                    # so add a fake extra dim and squeeze back down
                    im_real_pooled = np.expand_dims(im_real, axis=1)
                    im_real_pooled = pool(im_real_pooled, method, data_format='channels_first')
                    im_real_pooled = np.squeeze(im_real_pooled, axis=1)
                    pred_ensemble_pooled = pool(pred_ensemble, method, data_format='channels_first')

                for value in precip_values:
                    # binary instance of truth > threshold
                    # append an array of shape batch_size x H x W
                    y_true[method][value].append((im_real_pooled > value))
                    # check what proportion of pred > threshold
                    # collapse over ensemble dim, so append an array also of shape batch_size x H x W
                    y_score[method][value].append(np.mean(pred_ensemble_pooled > value, axis=1, dtype=np.single))
                del im_real_pooled
                del pred_ensemble_pooled
                gc.collect()

            # pred_ensemble is pretty large
            del im_real
            del pred_ensemble
            gc.collect()

        for method in pooling_methods:
            for value in precip_values:
                # turn list of batch_size x H x W into a single array
                y_true[method][value] = np.concatenate(y_true[method][value], axis=0)  # n_images x W x H
                gc.collect()  # clean up the list representation of y_true[value]

            for value in precip_values:
                # ditto
                y_score[method][value] = np.concatenate(y_score[method][value], axis=0)  # n_images x W x H
                gc.collect()

#         for method in pooling_methods:
#             for value in precip_values:
#                 # debug code for testing memory usage without generating samples
#                 print("Generating random array", value)
#                 y_true[method][value] = np.random.randint(0, 2, (256, 940, 940), dtype=np.bool)
#                 y_score[method][value] = (np.random.randint(0, 101, (256, 940, 940))/100.0).astype(np.single)
#                 gc.collect()

        fpr = {}; tpr = {}; rec = {}; pre = {}; baserates = {}; roc_auc = {}; pr_auc = {}
        for method in pooling_methods:
            fpr[method] = []  # list of ROC fprs
            tpr[method] = []  # list of ROC tprs
            rec[method] = []  # list of precision-recall recalls
            pre[method] = []  # list of precision-recall precisions
            baserates[method] = []  # precision-recall 'no-skill' levels
            roc_auc[method] = []  # list of ROC AUCs
            pr_auc[method] = []  # list of p-r AUCs

            print("Computing ROC and prec-recall for", method)
            for value in precip_values:
                # Compute ROC curve and ROC area for each precip value
                fpr_pv, tpr_pv, _ = roc_curve(np.ravel(y_true[method][value]), np.ravel(y_score[method][value]), drop_intermediate=False)
                gc.collect()
                pre_pv, rec_pv, _ = precision_recall_curve(np.ravel(y_true[method][value]), np.ravel(y_score[method][value]))
                gc.collect()
                # note: fpr_pv, tpr_pv, etc., are at most the size of the number of unique values of y_score.
                # for us, this is just "fraction of ensemble members > threshold" which is relatively small,
                # but if y_score took arbirary values, this could be really large (particularly with drop_intermediate=False)
                roc_auc_pv = auc(fpr_pv, tpr_pv)
                pr_auc_pv = auc(rec_pv, pre_pv)
                fpr[method].append(fpr_pv)
                tpr[method].append(tpr_pv)
                pre[method].append(pre_pv)
                rec[method].append(rec_pv)
                baserates[method].append(y_true[method][value].mean())
                roc_auc[method].append(roc_auc_pv)
                pr_auc[method].append(pr_auc_pv)
                del y_true[method][value]
                del y_score[method][value]
                gc.collect()

            if model_number in model_numbers:
                # i.e., don't do this for ecPoint
                auc_scores_roc[method].append(np.array(roc_auc[method]))
                auc_scores_pr[method].append(np.array(pr_auc[method]))

        if model_number in model_numbers:
            fname1 = "ROC-GAN-" + str(model_number) + "-fpr.pickle"
            fname2 = "ROC-GAN-" + str(model_number) + "-tpr.pickle"
            fname3 = "ROC-GAN-" + str(model_number) + "-auc.pickle"
            fname4 = "PRC-GAN-" + str(model_number) + "-rec.pickle"
            fname5 = "PRC-GAN-" + str(model_number) + "-pre.pickle"
            fname6 = "PRC-GAN-" + str(model_number) + "-auc.pickle"
            fname7 = "PRC-GAN-" + str(model_number) + "-base.pickle"
        else:
            fname1 = "ROC-" + model_number + "-fpr.pickle"
            fname2 = "ROC-" + model_number + "-tpr.pickle"
            fname3 = "ROC-" + model_number + "-auc.pickle"
            fname4 = "PRC-" + model_number + "-rec.pickle"
            fname5 = "PRC-" + model_number + "-pre.pickle"
            fname6 = "PRC-" + model_number + "-auc.pickle"
            fname7 = "PRC-" + model_number + "-base.pickle"
        fnamefull1 = os.path.join(log_folder, fname1)
        fnamefull2 = os.path.join(log_folder, fname2)
        fnamefull3 = os.path.join(log_folder, fname3)
        fnamefull4 = os.path.join(log_folder, fname4)
        fnamefull5 = os.path.join(log_folder, fname5)
        fnamefull6 = os.path.join(log_folder, fname6)
        fnamefull7 = os.path.join(log_folder, fname7)

        with open(fnamefull1, 'wb') as f:
            pickle.dump(fpr, f)
        with open(fnamefull2, 'wb') as f:
            pickle.dump(tpr, f)
        with open(fnamefull3, 'wb') as f:
            pickle.dump(roc_auc, f)
        with open(fnamefull4, 'wb') as f:
            pickle.dump(rec, f)
        with open(fnamefull5, 'wb') as f:
            pickle.dump(pre, f)
        with open(fnamefull6, 'wb') as f:
            pickle.dump(pr_auc, f)
        with open(fnamefull7, 'wb') as f:
            pickle.dump(baserates, f)

#         for method in pooling_methods:
#             # Plot all ROC curves
#             plt.figure(figsize=(7, 5))
#             lw = 2
#             colors = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy']
#             for i, color in zip(range(len(precip_values)), colors):
#                 plt.plot(fpr[method][i], tpr[method][i], color=color, lw=lw,
#                          label=f"ROC curve for precip value {precip_values[i]} (area = %0.2f)" % roc_auc[method][i])

#             plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # no-skill
#             plt.xlim([0.0, 1.0])
#             plt.ylim([0.0, 1.05])
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title(f'Model {model_label} ROC curve for {plot_input_title} problem, {ensemble_members} ensemble members, {batch_size*num_batches} images, pooling type {method}')
#             plt.legend(loc="lower right")
#             plt.savefig("{}/ROC-{}-{}-{}-{}.pdf".format(log_folder, problem_type, plot_label, model_label, method), bbox_inches='tight')
#             plt.close()

#             # Plot all precision-recall curves
#             plt.figure(figsize=(7, 5))
#             lw = 2
#             colors = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy']
#             for i, color in zip(range(len(precip_values)), colors):
#                 plt.plot(rec[method][i], pre[method][i], color=color, lw=lw,
#                          label=f"Precision-recall curve for precip value {precip_values[i]} (area = %0.2f)" % pr_auc[method][i])
#                 plt.plot([0, 1], [baserates[method][i], baserates[method][i]], '--', lw=0.5, color=color)  # no skill

#             plt.xlim([0.0, 1.0])
#             # plt.ylim([0.0, 1.05])
#             plt.xlabel('Recall')
#             plt.ylabel('Precision')
#             plt.title(f'Model {model_label} precision-recall curve for {plot_input_title} problem, {ensemble_members} ensemble members, {batch_size*num_batches} images, pooling type {method}')
#             plt.legend(loc="upper right")
#             plt.savefig("{}/PR-{}-{}-{}-{}.pdf".format(log_folder, problem_type, plot_label, model_label, method), bbox_inches='tight')
#             plt.close()

#     if len(model_numbers) > 0:  # just in case we're in ecPoint-only mode!
#         for method in pooling_methods:
#             auc_scores_roc[method] = np.transpose(np.array(auc_scores_roc[method]))
#             plt.figure(figsize=(8, 5))
#             colors = ['darkturquoise', 'teal', 'dodgerblue', 'navy', 'purple']
#             for i, color in zip(range(len(precip_values)), colors):
#                 plt.plot(model_numbers, auc_scores_roc[method][i], color=color, lw=lw,
#                          label=f"AUC values for precip_values {precip_values[i]}")

#             plt.ylim([0, 1.0])
#             plt.xlabel('Checkpoint number')
#             plt.ylabel('Area under ROC curve')
#             plt.title(f'ROC AUC values for varying precip thresholds, pooling type {method}')
#             plt.legend(loc="best")
#             plt.savefig("{}/AUC-ROC-{}-{}-{}.pdf".format(log_folder, problem_type, plot_label, method), bbox_inches='tight')
#             plt.close()

#             auc_scores_pr[method] = np.transpose(np.array(auc_scores_pr[method]))
#             plt.figure(figsize=(8, 5))
#             colors = ['darkturquoise', 'teal', 'dodgerblue', 'navy', 'purple']
#             for i, color in zip(range(len(precip_values)), colors):
#                 plt.plot(model_numbers, auc_scores_pr[method][i], color=color, lw=lw,
#                          label=f"AUC values for precip_values {precip_values[i]}")

#             plt.ylim([0, 1.0])
#             plt.xlabel('Checkpoint number')
#             plt.ylabel('Area under ROC curve')
#             plt.title(f'PR AUC values for varying precip thresholds, pooling type {method}')
#             plt.legend(loc="best")
#             plt.savefig("{}/AUC-PR-{}-{}-{}.pdf".format(log_folder, problem_type, plot_label, method), bbox_inches='tight')
#             plt.close()
