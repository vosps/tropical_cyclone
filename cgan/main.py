import argparse
import json
import os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")  # noqa: E702
import numpy as np
import pandas as pd
import yaml
import train
import setupmodel
import setupdata
import evaluation
import plots
import roc
from generate_predictions import generate_predictions

import tensorflow as tf
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Is GPU available? ",tf.test.is_gpu_available())
if tf.test.is_built_with_cuda():
    print("The installed version of TensorFlow includes GPU support.")
else:
    print("The installed version of TensorFlow does not include GPU support.")
# exit()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration file")
    parser.set_defaults(do_training=True)
    parser.add_argument('--no_train', dest='do_training', action='store_false',
                        help="Do NOT carry out training, only perform eval")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--eval_full', dest='evalnum', action='store_const', const="full")
    group.add_argument('--eval_short', dest='evalnum', action='store_const', const="short")
    group.add_argument('--eval_blitz', dest='evalnum', action='store_const', const="blitz")
    parser.set_defaults(evalnum=None)
    parser.set_defaults(rank_small=False)
    parser.set_defaults(rank_full=False)
    parser.set_defaults(qual_small=False)
    parser.set_defaults(qual_full=False)
    parser.set_defaults(plot_ranks_small=False)
    parser.set_defaults(plot_ranks_full=False)
    parser.set_defaults(plot_roc_small=False)
    parser.set_defaults(plot_roc_full=False)
    parser.add_argument('--rank_small', dest='rank_small', action='store_true',
                        help="Include CRPS/rank evaluation on small images")
    parser.add_argument('--rank_full', dest='rank_full', action='store_true',
                        help="Include CRPS/rank evaluation on full images")
    parser.add_argument('--qual_small', dest='qual_small', action='store_true',
                        help="Include image quality metrics on small images")
    parser.add_argument('--qual_full', dest='qual_full', action='store_true',
                        help="Include image quality metrics on full images")
    parser.add_argument('--plot_ranks_small', dest='plot_ranks_small', action='store_true',
                        help="Plot rank histograms for small images")
    parser.add_argument('--plot_ranks_full', dest='plot_ranks_full', action='store_true',
                        help="Plot rank histograms for full images")
    parser.add_argument('--plot_roc_small', dest='plot_roc_small', action='store_true',
                        help="Plot ROC and AUC curves for small images")
    parser.add_argument('--plot_roc_full', dest='plot_roc_full', action='store_true',
                        help="Plot ROC and AUC curves for full images")                   
    args = parser.parse_args()

    if args.evalnum is None and (args.rank_small or args.rank_full or args.qual_small or args.qual_full or args.plot_roc_small or args.plot_roc_full):
        raise RuntimeError("You asked for evaluation to occur, but did not pass in '--eval_full', '--eval_short', or '--eval_blitz' to specify length of evaluation")

        # Read in the configurations
    if args.config is not None:
        config_path = args.config
    else:
        raise Exception("Please specify configuration!")

    with open(config_path, 'r') as f:
        try:
            setup_params = yaml.safe_load(f)
            print(setup_params)
        except yaml.YAMLError as exc:
            print(exc)

    mode = setup_params["GENERAL"]["mode"]
    arch = setup_params["MODEL"]["architecture"]
    log_folder = setup_params["SETUP"]["log_folder"]
    problem_type = setup_params["GENERAL"]["problem_type"]
    filters_gen = setup_params["GENERATOR"]["filters_gen"]
    lr_gen = setup_params["GENERATOR"]["learning_rate_gen"]
    noise_channels = setup_params["GENERATOR"]["noise_channels"]
    latent_variables = setup_params["GENERATOR"]["latent_variables"]
    filters_disc = setup_params["DISCRIMINATOR"]["filters_disc"]
    lr_disc = setup_params["DISCRIMINATOR"]["learning_rate_disc"]
    train_years = setup_params["TRAIN"]["train_years"]
    training_weights = setup_params["TRAIN"]["training_weights"]
    num_samples = setup_params["TRAIN"]["num_samples"]
    steps_per_epoch = setup_params["TRAIN"]["steps_per_epoch"]
    batch_size = setup_params["TRAIN"]["batch_size"]
    kl_weight = setup_params["TRAIN"]["kl_weight"]
    val_years = setup_params["VAL"]["val_years"]
    val_size = setup_params["VAL"]["val_size"]
    num_batches = setup_params["EVAL"]["num_batches"]
    add_noise = setup_params["EVAL"]["add_postprocessing_noise"]
    noise_factor = setup_params["EVAL"]["postprocessing_noise_factor"]

    # otherwise these are of type string, e.g. '1e-5'
    lr_gen = float(lr_gen)
    lr_disc = float(lr_disc)
    kl_weight = float(kl_weight)
    noise_factor = float(noise_factor)

    if mode not in ['GAN', 'VAEGAN', 'det']:
        raise ValueError("Mode type is restricted to 'GAN' 'VAEGAN' 'det'")
    if problem_type not in ['normal', 'superresolution']:
        raise ValueError("Problem type is restricted to 'normal' 'superresolution'")

    num_epochs = int(num_samples/(steps_per_epoch * batch_size))
    epoch = 1

    # create log folder and model save/load subfolder if they don't exist
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    model_weights_root = os.path.join(log_folder, "models")
    Path(model_weights_root).mkdir(parents=True, exist_ok=True)
    
    # save setup parameters
    save_config = os.path.join(log_folder, 'setup_params.yaml')
    with open(save_config, 'w') as outfile:
        yaml.dump(setup_params, outfile, default_flow_style=False)

    if problem_type == "normal":
        downsample = False
        input_channels = 9
    elif problem_type == "superresolution":
        downsample = True
        input_channels = 1
    else:
        raise ValueError("no such problem type, try again!")

    if args.do_training:
        
        # initialize GAN
        model = setupmodel.setup_model(
            mode=mode,
            arch=arch,
            input_channels=input_channels,
            latent_variables=latent_variables,
            filters_gen=filters_gen,
            filters_disc=filters_disc,
            noise_channels=noise_channels,
            lr_disc=lr_disc,
            lr_gen=lr_gen,
            kl_weight=kl_weight)

        batch_gen_train, batch_gen_valid = setupdata.setup_data(
            train_years=train_years,
            val_years=val_years,
            val_size=val_size,
            downsample=downsample,
            weights=training_weights,
            batch_size=batch_size,
            load_full_image=False) #TODO: fix issue here
        
        print('batch_gen_train',batch_gen_train.take(1))
        print('batch_gen_valid',batch_gen_valid.take(1))

    # Leaving around for now in case this is useful for e.g. multiple training
#         if load_weights_root:  # load weights and run status
#             model.load(model.filenames_from_root(load_weights_root))
#             with open(load_weights_root + "-run_status.json", 'r') as f:
#                 run_status = json.load(f)
#             training_samples = run_status["training_samples"]

#             if log_path:
#                 log_file = "{}/log.txt".format(log_path)
#                 log = pd.read_csv(log_file)

        if False:
            pass
        else:  # initialize run status
            training_samples = 0

            log_file = os.path.join(log_folder, "log.txt")
            # TODO: does VAEGAN need more?
            if mode == "GAN":
                log = pd.DataFrame(
                    columns=["training_samples",
                             "disc_loss", "disc_loss_real",
                             "disc_loss_fake", "disc_loss_gp",
                             "gen_loss"])
            elif mode == "det":
                log = pd.DataFrame(columns=["training_samples",
                                            "loss"])
            elif mode == "VAEGAN":
                log = pd.DataFrame(
                    columns=["training_samples",
                             "disc_loss", "disc_loss_real",
                             "disc_loss_fake", "disc_loss_gp",
                             "gen_loss_total",
                             "gen_loss_disc", "gen_loss_kl"])

        plot_fname = os.path.join(log_folder, "progress.pdf")

        while (training_samples < num_samples):  # main training loop

            print("Epoch {}/{}".format(epoch, num_epochs))

            # train for some number of batches
            print('training model')
            print('batch_gen_train',batch_gen_train)
            print('epoch',epoch)
            print('steps per epoch',steps_per_epoch)

            loss_log = train.train_model(model=model,
                                         mode=mode,
                                         batch_gen_train=batch_gen_train,
                                         batch_gen_valid=batch_gen_valid,
                                         noise_channels=noise_channels,
                                         latent_variables=latent_variables,
                                         epoch=epoch,
                                         steps_per_epoch=steps_per_epoch,
                                         plot_samples=val_size,
                                         plot_fn=plot_fname)

            loss_log = np.mean(loss_log, axis=0)
            training_samples += steps_per_epoch * batch_size
            epoch += 1

            # save results
            model.save(model_weights_root)
            run_status = {
                "training_samples": training_samples,
            }
            with open(os.path.join(log_folder, "run_status.json"), 'w') as f:
                json.dump(run_status, f)

            if mode == "GAN":
                log = log.append(pd.DataFrame(data={
                    "training_samples": [training_samples],
                    "disc_loss": [loss_log[0]],
                    "disc_loss_real": [loss_log[1]],
                    "disc_loss_fake": [loss_log[2]],
                    "disc_loss_gp": [loss_log[3]],
                    "gen_loss": [loss_log[4]]
                }))
            elif mode == "det":
                log = log.append(pd.DataFrame(data={
                    "training_samples": [training_samples],
                    "loss": [loss_log],
                }))
            elif mode == "VAEGAN":
                log = log.append(pd.DataFrame(data={
                    "training_samples": [training_samples],
                    "disc_loss": [loss_log[0]],
                    "disc_loss_real": [loss_log[1]],
                    "disc_loss_fake": [loss_log[2]],
                    "disc_loss_gp": [loss_log[3]],
                    "gen_loss_total": [loss_log[4]],
                    "gen_loss_disc": [loss_log[5]],
                    "gen_loss_kl": [loss_log[6]]
                }))
            else:
                assert False
            log.to_csv(log_file, index=False, float_format="%.6f")

            # Save model weights each epoch
            gen_weights_file = os.path.join(model_weights_root, "gen_weights-{:07d}.h5".format(training_samples))
            model.gen.save_weights(gen_weights_file)
        print('model training complete!')

    else:
        print("Training skipped...")

    rank_small_fname = os.path.join(log_folder, "rank-small.txt")
    rank_full_fname = os.path.join(log_folder, "rank-full.txt")
    qual_small_fname = os.path.join(log_folder, "qual-small.txt")
    qual_full_fname = os.path.join(log_folder, "qual-full.txt")

    # This works nicely for the 100 * 3200 training samples that we have been
    # working with. If these numbers change, may want to update evaluation.py
    # accordingly.
    if args.evalnum == "blitz":
        model_numbers = [124800, 198400, 240000, 320000]
    elif args.evalnum == "short":
        # this assumes 100 'epochs', may want to generalise?!
        interval = steps_per_epoch * batch_size
        model_numbers = [37*interval, 38*interval, 39*interval, 40*interval,
                         59*interval, 60*interval, 61*interval, 62*interval,
                         75*interval, 76*interval, 77*interval, 78*interval,
                         97*interval, 98*interval, 99*interval, 100*interval]
    elif args.evalnum == "full":
        interval = steps_per_epoch * batch_size
        model_numbers = np.arange(0, num_samples + 1, interval)[1:].tolist()

    # evaluate model performance
    if args.qual_small:
        evaluation.quality_metrics_by_time(mode=mode,
                                           arch=arch,
                                           val_years=val_years,
                                           log_fname=qual_small_fname,
                                           weights_dir=model_weights_root,
                                           downsample=downsample,
                                           weights=training_weights,
                                           load_full_image=False,
                                           model_numbers=model_numbers,
                                           batch_size=batch_size,
                                           num_batches=num_batches,
                                           filters_gen=filters_gen,
                                           filters_disc=filters_disc,
                                           input_channels=input_channels,
                                           latent_variables=latent_variables,
                                           noise_channels=noise_channels)

    if args.qual_full:
        evaluation.quality_metrics_by_time(mode=mode,
                                           arch=arch,
                                           val_years=val_years,
                                           log_fname=qual_full_fname,
                                           weights_dir=model_weights_root,
                                           downsample=downsample,
                                           weights=training_weights,
                                           load_full_image=True,
                                           model_numbers=model_numbers,
                                           batch_size=1,  # memory issues
                                           num_batches=num_batches,
                                           filters_gen=filters_gen,
                                           filters_disc=filters_disc,
                                           input_channels=input_channels,
                                           latent_variables=latent_variables,
                                           noise_channels=noise_channels)

    if args.rank_small:
        evaluation.rank_metrics_by_time(mode=mode,
                                        arch=arch,
                                        val_years=val_years,
                                        log_fname=rank_small_fname,
                                        weights_dir=model_weights_root,
                                        downsample=downsample,
                                        weights=training_weights,
                                        add_noise=add_noise,
                                        noise_factor=noise_factor,
                                        load_full_image=False,
                                        model_numbers=model_numbers,
                                        batch_size=batch_size,
                                        num_batches=num_batches,
                                        filters_gen=filters_gen,
                                        filters_disc=filters_disc,
                                        input_channels=input_channels,
                                        latent_variables=latent_variables,
                                        noise_channels=noise_channels,
                                        rank_samples=100)

    if args.rank_full:
        evaluation.rank_metrics_by_time(mode=mode,
                                        arch=arch,
                                        val_years=val_years,
                                        log_fname=rank_full_fname,
                                        weights_dir=model_weights_root,
                                        downsample=downsample,
                                        weights=training_weights,
                                        add_noise=add_noise,
                                        noise_factor=noise_factor,
                                        load_full_image=True,
                                        model_numbers=model_numbers,
                                        batch_size=1,  # memory issues
                                        num_batches=num_batches,
                                        filters_gen=filters_gen,
                                        filters_disc=filters_disc,
                                        input_channels=input_channels,
                                        latent_variables=latent_variables,
                                        noise_channels=noise_channels,
                                        rank_samples=100)

    if args.plot_ranks_small:
        if add_noise:
            noise_label = "noise"
            rank_metrics_files_1 = ["{}/ranks-noise-124800.npz".format(log_folder), "{}/ranks-noise-198400.npz".format(log_folder)]
            rank_metrics_files_2 = ["{}/ranks-noise-240000.npz".format(log_folder), "{}/ranks-noise-320000.npz".format(log_folder)]
            labels_1 = ['noise-124800', 'noise-198400']
            labels_2 = ['noise-240000', 'noise-320000']
            name_1 = 'noise-early-small_image'
            name_2 = 'noise-late-small_image'
        else:
            noise_label = "no_noise"
            rank_metrics_files_1 = ["{}/ranks-no-noise-124800.npz".format(log_folder), "{}/ranks-no-noise-198400.npz".format(log_folder)]
            rank_metrics_files_2 = ["{}/ranks-no-noise-240000.npz".format(log_folder), "{}/ranks-no-noise-320000.npz".format(log_folder)]
            labels_1 = ['no-noise-124800', 'no-noise-198400']
            labels_2 = ['no-noise-240000', 'no-noise-320000']
            name_1 = 'no-noise-early-small_image'
            name_2 = 'no-noise-late-small_image'
        plots.plot_rank_histogram_all(rank_files=rank_metrics_files_1, 
                                      labels=labels_1, 
                                      log_path=log_folder, 
                                      name=name_1)
        plots.plot_rank_histogram_all(rank_files=rank_metrics_files_2, 
                                      labels=labels_2, 
                                      log_path=log_folder, 
                                      name=name_2)
    if args.plot_ranks_full:
        if add_noise:
            rank_metrics_files_1 = ["{}/ranks-full_image-noise-124800.npz".format(log_folder), "{}/ranks-full_image-noise-198400.npz".format(log_folder)]
            rank_metrics_files_2 = ["{}/ranks-full_image-noise-240000.npz".format(log_folder), "{}/ranks-full_image-noise-320000.npz".format(log_folder)]
            labels_1 = ['noise-124800', 'noise-198400']
            labels_2 = ['noise-240000', 'noise-320000']
            name_1 = 'noise-early-full_image'
            name_2 = 'noise-late-full_image'
        else:
            rank_metrics_files_1 = ["{}/ranks-full_image-no-noise-124800.npz".format(log_folder), "{}/ranks-full_image-no-noise-198400.npz".format(log_folder)]
            rank_metrics_files_2 = ["{}/ranks-full_image-no-noise-240000.npz".format(log_folder), "{}/ranks-full_image-no-noise-320000.npz".format(log_folder)]  
            labels_1 = ['no-noise-124800', 'no-noise-198400']
            labels_2 = ['no-noise-240000', 'no-noise-320000']
            name_1 = 'no-noise-early-full_image'
            name_2 = 'no-noise-late-full_image'
        plots.plot_rank_histogram_all(rank_files=rank_metrics_files_1, 
                                      labels=labels_1, 
                                      log_path=log_folder, 
                                      name=name_1)
        plots.plot_rank_histogram_all(rank_files=rank_metrics_files_2, 
                                      labels=labels_2, 
                                      log_path=log_folder, 
                                      name=name_2)

    if args.plot_roc_small:
        predict_full_image = False
        roc.plot_roc_curves(mode=mode,
                            arch=arch,
                            log_folder=log_folder,
                            weights_dir=model_weights_root,
                            model_numbers=model_numbers,
                            problem_type=problem_type,
                            filters_gen=filters_gen,
                            filters_disc=filters_disc,
                            noise_channels=noise_channels,
                            latent_variables=latent_variables,
                            predict_year=val_years,
                            predict_full_image=predict_full_image,
                            )

    if args.plot_roc_full:
        predict_full_image = True
        roc.plot_roc_curves(mode=mode,
                            arch=arch,
                            log_folder=log_folder,
                            weights_dir=model_weights_root,
                            model_numbers=model_numbers,
                            problem_type=problem_type,
                            filters_gen=filters_gen,
                            filters_disc=filters_disc,
                            noise_channels=noise_channels,
                            latent_variables=latent_variables,
                            predict_year=val_years,
                            predict_full_image=predict_full_image,
                            )


    generate_predictions(mode=mode,
                            arch=arch,
                            log_folder=log_folder,
                            weights_dir=model_weights_root,
                            model_numbers=model_numbers,
                            problem_type=problem_type,
                            filters_gen=filters_gen,
                            filters_disc=filters_disc,
                            noise_channels=noise_channels,
                            latent_variables=latent_variables,
                            predict_year=val_years,
                            predict_full_image=False,
                            )

    # generate_predictions(mode=mode,
    #                         arch=arch,
    #                         log_folder=log_folder,
    #                         weights_dir=model_weights_root,
    #                         model_numbers=model_numbers,
    #                         problem_type=problem_type,
    #                         filters_gen=filters_gen,
    #                         filters_disc=filters_disc,
    #                         noise_channels=noise_channels,
    #                         latent_variables=latent_variables,
    #                         predict_year=val_years,
    #                         predict_full_image=False,
    #                         gcm = True
    #                         )
