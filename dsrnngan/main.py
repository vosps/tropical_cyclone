import argparse
import json
import os
from pathlib import Path

import matplotlib; matplotlib.use("Agg")  # noqa: E702
import numpy as np
import pandas as pd

import train
import evaluation

# TODO: Plots

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=("GAN", "det", "VAEGAN"),
                        help="type of model (pure GAN / deterministic / VAEGAN)")
    parser.add_argument('--log_folder', type=str, required=True,
       help="Folder for saving/loading model weights, log files, etc.  Will be created if it doesn't already exist.")  # noqa: E128
    parser.add_argument('--problem_type', type=str, default="normal",
                        choices=("normal", "superresolution"),
        help="normal: IFS to NIMROD. superresolution: coarsened NIMROD to NIMROD")  # noqa: E128
    parser.add_argument('--architecture', type=str, default="normal",
                        choices=("normal",),
                        help="name of model architecture to use")
    parser.add_argument('--train_years', type=int, nargs='+',
                        default=[2016, 2017, 2018],
                        help="Training years")
    parser.add_argument('--val_years', type=int, nargs='+', default=2019,
      help="Validation years -- cannot pass a list if using create_fixed_dataset")  # noqa: E128
    parser.add_argument('--val_size', type=int, default=8,
                        help='Number of validation examples')
    parser.add_argument('--num_samples', type=int, default=320000,
                        help="Training samples")
    parser.add_argument('--steps_per_epoch', type=int, default=200,
                        help="Batches per epoch")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for training and small-image eval")
    parser.add_argument('--num_batches', type=int, default=64,
                        help="Number of batches for eval metrics")
    parser.add_argument('--filters_gen', type=int, default=128,
                        help="Number of filters used in generator")
    parser.add_argument('--filters_disc', type=int, default=512,
                        help="Number of filters used in discriminator")
    parser.add_argument('--noise_channels', type=int, default=4,
                        help="Dimensions of noise passed to generator")
    parser.add_argument('--latent_variables', type=int, default=1,
                        help="Latent variables per 'pixel' in VAEGAN")
    parser.add_argument('--learning_rate_disc', type=float, default=1e-5,
                        help="Learning rate used for discriminator optimizer")
    parser.add_argument('--learning_rate_gen', type=float, default=1e-5,
                        help="Learning rate used for generator optimizer")
    parser.add_argument('--kl_weight', type=float, default=1e-8,
                        help="Weight of KL term in VAEGAN")

    parser.set_defaults(do_training=True)
    parser.add_argument('--no_train', dest='do_training', action='store_false',
                        help="Do NOT carry out training, only perform eval")
    parser.set_defaults(rank_small=False)
    parser.set_defaults(rank_full=False)
    parser.set_defaults(qual_small=False)
    parser.set_defaults(qual_full=False)
    parser.add_argument('--rank_small', dest='rank_small', action='store_true',
                        help="Include CRPS/rank evaluation on small images")
    parser.add_argument('--rank_full', dest='rank_full', action='store_true',
                        help="Include CRPS/rank evaluation on full images")
    parser.add_argument('--qual_small', dest='qual_small', action='store_true',
                        help="Include image quality metrics on small images")
    parser.add_argument('--qual_full', dest='qual_full', action='store_true',
                        help="Include image quality metrics on full images")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--eval_full', dest='evalnum', action='store_const', const="full")
    group.add_argument('--eval_short', dest='evalnum', action='store_const', const="short")
    group.add_argument('--eval_blitz', dest='evalnum', action='store_const', const="blitz")
    parser.set_defaults(evalnum=None)

    parser.add_argument('--add_postprocessing_noise', type=bool, default=True,
        help="Flag for adding postprocessing noise in rank statistics eval")  # noqa: E128
    parser.add_argument('--postprocessing_noise_factor', type=float, default=1e-6,
        help="Factor for scaling postprocessing noise in rank statistics eval")  # noqa: E128

    args = parser.parse_args()

    if args.evalnum is None and (args.rank_small or args.rank_full or args.qual_small or args.qual_full):
        raise RuntimeError("You asked for evaluation to occur, but did not pass in '--eval_full', '--eval_short', or '--eval_blitz' to specify length of evaluation")

    # training_weights = np.arange(12,2,-3)
    # training_weights = training_weights / training_weights.sum()
    # training_weights = None
    # training_weights = [0.87, 0.06, 0.03, 0.03]
    # training_weights = np.arange(24,2,-7)
    # training_weights = weights / weights.sum()
    training_weights = [0.4, 0.3, 0.2, 0.1]
    # training_weights_12x = np.arange(36,2,-11)
    # training_weights = training_weights_12x / training_weights_12x.sum()
    print(f"training_weights for data loading are {training_weights}")

    mode = args.mode
    arch = args.architecture
    log_folder = args.log_folder
    steps_per_epoch = args.steps_per_epoch
    batch_size = args.batch_size
    val_size = args.val_size
    num_samples = args.num_samples
    train_years = args.train_years
    val_years = args.val_years
    num_batches = args.num_batches
    filters_disc = args.filters_disc
    filters_gen = args.filters_gen
    lr_disc = args.learning_rate_disc
    lr_gen = args.learning_rate_gen
    kl_weight = args.kl_weight
    noise_channels = args.noise_channels
    latent_variables = args.latent_variables
    add_noise = args.add_postprocessing_noise
    noise_factor = args.postprocessing_noise_factor

    num_epochs = int(num_samples/(steps_per_epoch * batch_size))
    epoch = 1

    # create log folder and model save/load subfolder if they don't exist
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    model_weights_root = os.path.join(log_folder, "models")
    Path(model_weights_root).mkdir(parents=True, exist_ok=True)

    if args.problem_type == "normal":
        downsample = False
        input_channels = 9
    elif args.problem_type == "superresolution":
        downsample = True
        input_channels = 1
    else:
        raise Exception("no such problem type, try again!")

    if args.do_training:
        # initialize GAN
        print(f"val years is {val_years}")
        model, batch_gen_train, batch_gen_valid, _, _ = \
            train.setup_model(mode=mode,
                              arch=arch,
                              train_years=train_years,
                              val_years=val_years,
                              val_size=val_size,
                              downsample=downsample,
                              weights=training_weights,
                              input_channels=input_channels,
                              latent_variables=latent_variables,
                              batch_size=batch_size,
                              filters_gen=filters_gen,
                              filters_disc=filters_disc,
                              noise_channels=noise_channels,
                              lr_disc=lr_disc,
                              lr_gen=lr_gen,
                              kl_weight=kl_weight)

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
            if mode in ("GAN", "VAEGAN"):
                log = pd.DataFrame(
                    columns=["training_samples",
                             "disc_loss", "disc_loss_real",
                             "disc_loss_fake", "disc_loss_gp",
                             "gen_loss"])
            elif mode == "det":
                log = pd.DataFrame(columns=["training_samples",
                                            "loss"])

        plot_fname = os.path.join(log_folder, "progress.pdf")
        rank_small_fname = os.path.join(log_folder, "rank-small.txt")
        rank_full_fname = os.path.join(log_folder, "rank-full.txt")
        qual_small_fname = os.path.join(log_folder, "qual-small.txt")
        qual_full_fname = os.path.join(log_folder, "qual-full.txt")

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
            else:
                assert False
            log.to_csv(log_file, index=False, float_format="%.6f")

            # Save model weights each epoch
            gen_weights_file = os.path.join(model_weights_root, "gen_weights-{:07d}.h5".format(training_samples))
            model.gen.save_weights(gen_weights_file)

    else:
        print("Training skipped...")

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
    if args.rank_small:
        evaluation.rank_metrics_by_time(mode=mode,
                                        arch=arch,
                                        val_years=val_years,
                                        log_fn=rank_small_fname,
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
                                        log_fn=rank_full_fname,
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

    if args.qual_small:
        evaluation.quality_metrics_by_time(mode=mode,
                                           arch=arch,
                                           val_years=val_years,
                                           log_fn=qual_small_fname,
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
                                           log_fn=qual_full_fname,
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
