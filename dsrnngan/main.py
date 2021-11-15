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


# model iterations to save full rank data to disk for during evaluations;
# necessary for plot rank histograms. these are large files, so quasi-random
# selection used to avoid storing gigabytes of data
# TODO: this could live in .yaml file, but leave here for now
ranks_to_save = [128000, 192000, 256000, 320000]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration file")
    parser.set_defaults(do_training=True)
    parser.add_argument('--no_train', dest='do_training', action='store_false',
                        help="Do NOT carry out training, only perform eval")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--eval_full', dest='evalnum', action='store_const', const="full")
    group.add_argument('--eval_short', dest='evalnum', action='store_const', const="short")
    group.add_argument('--eval_tenth', dest='evalnum', action='store_const', const="tenth")
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
        raise RuntimeError("You asked for evaluation to occur, but did not pass in '--eval_full', '--eval_short', '--eval_tenth', or '--eval_blitz' to specify length of evaluation")

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
    padding = setup_params["MODEL"]["padding"]
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
    ensemble_size = setup_params["TRAIN"]["ensemble_size"]
    content_loss_weight = setup_params["TRAIN"]["content_loss_weight"]
    val_years = setup_params["VAL"]["val_years"]
    val_size = setup_params["VAL"]["val_size"]
    num_batches = setup_params["EVAL"]["num_batches"]
    add_noise = setup_params["EVAL"]["add_postprocessing_noise"]
    noise_factor = setup_params["EVAL"]["postprocessing_noise_factor"]
    max_pooling = setup_params["EVAL"]["max_pooling"]
    avg_pooling = setup_params["EVAL"]["avg_pooling"]

    # otherwise these are of type string, e.g. '1e-5'
    lr_gen = float(lr_gen)
    lr_disc = float(lr_disc)
    kl_weight = float(kl_weight)
    noise_factor = float(noise_factor)
    content_loss_weight = float(content_loss_weight)

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
            padding=padding,
            lr_disc=lr_disc,
            lr_gen=lr_gen,
            kl_weight=kl_weight,
            ensemble_size=ensemble_size,
            content_loss_weight=content_loss_weight)

        batch_gen_train, batch_gen_valid = setupdata.setup_data(
            train_years=train_years,
            val_years=val_years,
            val_size=val_size,
            downsample=downsample,
            weights=training_weights,
            batch_size=batch_size,
            load_full_image=False)

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

        plot_fname = os.path.join(log_folder, "progress.pdf")

        while (training_samples < num_samples):  # main training loop

            print("Epoch {}/{}".format(epoch, num_epochs))

            # train for some number of batches
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

            training_samples += steps_per_epoch * batch_size

            if epoch == 1:
                # set up log DataFrame based on loss_log entries
                col_names = ["training_samples"] + [foo for foo in loss_log]
                log = pd.DataFrame(columns=col_names)

            epoch += 1

            # save results
            model.save(model_weights_root)
            run_status = {
                "training_samples": training_samples,
            }
            with open(os.path.join(log_folder, "run_status.json"), 'w') as f:
                json.dump(run_status, f)

            data = {"training_samples": [training_samples]}
            for foo in loss_log:
                data[foo] = loss_log[foo]

            log = log.append(pd.DataFrame(data=data))
            log.to_csv(log_file, index=False, float_format="%.6f")

            # Save model weights each epoch
            gen_weights_file = os.path.join(model_weights_root, "gen_weights-{:07d}.h5".format(training_samples))
            model.gen.save_weights(gen_weights_file)

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
        model_numbers = ranks_to_save.copy()  # should not be modifying list in-place, but just in case!
    elif args.evalnum == "short":
        # hand-picked set of 12; 2 lots of 6 consecutive epochs including
        # the default ranks_to_save
        # this assumes 25 'epochs', may want to generalise?!
        interval = steps_per_epoch * batch_size
        model_numbers = [10*interval, 11*interval, 12*interval, 13*interval, 14*interval, 15*interval,
                         20*interval, 21*interval, 22*interval, 23*interval, 24*interval, 25*interval]
    elif args.evalnum == "tenth":  # every 10th; does NOT include fav numbers
        interval = steps_per_epoch * batch_size
        model_numbers = np.arange(0, num_samples + 1, 10*interval)[1:].tolist()
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
                                           noise_channels=noise_channels,
                                           padding=padding)

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
                                           noise_channels=noise_channels,
                                           padding=padding)

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
                                        padding=padding,
                                        rank_samples=10,
                                        max_pooling=max_pooling,
                                        avg_pooling=avg_pooling)

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
                                        padding=padding,
                                        rank_samples=10,
                                        max_pooling=max_pooling,
                                        avg_pooling=avg_pooling)

    if args.plot_ranks_small:
        plots.plot_histograms(log_folder, ranks=ranks_to_save, N_ranks=11, 
                              add_noise=add_noise, full_image=False)
    if args.plot_ranks_full:
        plots.plot_histograms(log_folder, ranks=ranks_to_save, N_ranks=11, 
                              add_noise=add_noise, full_image=True)

    if args.plot_roc_small:
        predict_full_image = False
        roc.plot_roc_curves(mode=mode,
                            arch=arch,
                            log_folder=log_folder,
                            weights_dir=model_weights_root,
                            model_numbers=ranks_to_save,
                            problem_type=problem_type,
                            filters_gen=filters_gen,
                            filters_disc=filters_disc,
                            noise_channels=noise_channels,
                            latent_variables=latent_variables,
                            padding=padding,
                            predict_year=val_years,
                            predict_full_image=predict_full_image)
    if args.plot_roc_full:
        predict_full_image = True
        roc.plot_roc_curves(mode=mode,
                            arch=arch,
                            log_folder=log_folder,
                            weights_dir=model_weights_root,
                            model_numbers=ranks_to_save,
                            problem_type=problem_type,
                            filters_gen=filters_gen,
                            filters_disc=filters_disc,
                            noise_channels=noise_channels,
                            latent_variables=latent_variables,
                            padding=padding,
                            predict_year=val_years,
                            predict_full_image=predict_full_image)
