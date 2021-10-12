import argparse
import json
import os

import matplotlib; matplotlib.use("Agg")  # noqa: E702

import numpy as np
import pandas as pd

import train
import evaluation


path = os.path.dirname(os.path.abspath(__file__))

# TODO: remove application?
# Simplify Log path / save weights root?
# Add check_every argument
# Plots

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
                        help="GAN, detGAN, VAEGAN")
    parser.add_argument('--problem_type', type=str, default="normal",
        help="normal (IFS to NIMROD), superresolution (NIMROD to NIMROD)")  # noqa: E128
    parser.add_argument('--train_years', type=int, nargs='+',
                        default=[2016, 2017, 2018],
                        help="Training years")
    parser.add_argument('--val_years', type=int, nargs='+', default=2019,
      help="Validation years -- cannot pass a list if using create_fixed_dataset")  # noqa: E128
    parser.add_argument('--val_size', type=int, default=8,
                        help='Num val examples')
    parser.add_argument('--load_weights_root', type=str, default="",
                        help="Network weights file root to load")
    parser.add_argument('--save_weights_root', type=str, default="",
                        help="Network weights file root to save")
    parser.add_argument('--log_path', type=str, default="",
                        help="Log files path")
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
    parser.add_argument('--learning_rate_disc', type=float, default=1e-5,
                        help="Learning rate used for discriminator optimizer")
    parser.add_argument('--learning_rate_gen', type=float, default=1e-5,
                        help="Learning rate used for generator optimizer")
    parser.add_argument('--no_train', dest='do_training', action='store_false',
                        help="Do NOT carry out training, only perform eval")
    parser.add_argument('--eval_small', dest='eval_small', action='store_true',
                        help="Include CRPS/rank evaluation on small images")
    parser.add_argument('--eval_full', dest='eval_full', action='store_true',
                        help="Include CRPS/rank evaluation on full images")
    parser.add_argument('--qual_small', dest='qual_small', action='store_true',
                        help="Include image quality metrics on small images")
    parser.add_argument('--qual_full', dest='qual_full', action='store_true',
                        help="Include image quality metrics on full images")
    parser.set_defaults(do_training=True)
    parser.set_defaults(eval_small=False)
    parser.set_defaults(eval_full=False)
    parser.set_defaults(qual_small=False)
    parser.set_defaults(qual_full=False)
    parser.add_argument('--add_postprocessing_noise', type=bool, default=True,
        help="Flag for adding postprocessing noise in rank statistics eval")  # noqa: E128
    parser.add_argument('--postprocessing_noise_factor', type=float, default=1e-6,
        help="Factor for scaling postprocessing noise in rank statistics eval")  # noqa: E128

    args = parser.parse_args()
    mode = args.mode

    assert args.mode in ("GAN", "detGAN", "VAEGAN")

    # weights = np.arange(12,2,-3)
    # weights = weights / weights.sum()
    # weights = None
    # weights = [0.87, 0.06, 0.03, 0.03]
    # weights = np.arange(24,2,-7)
    # weights = weights / weights.sum()
    weights = [0.4, 0.3, 0.2, 0.1]
    # weights_12x = np.arange(36,2,-11)
    # weights = weights_12x / weights_12x.sum()

    print(f"weights for data loading are {weights}")

    load_weights_root = args.load_weights_root
    save_weights_root = args.save_weights_root
    log_path = args.log_path
    steps_per_epoch = args.steps_per_epoch
    batch_size = args.batch_size
    val_size = args.val_size
    num_samples = args.num_samples
    train_years = args.train_years
    val_years = args.val_years
    num_batches = args.num_batches
    filters_disc = args.filters_disc
    filters_gen = args.filters_gen
    if mode in ("GAN", "VAEGAN"):
        lr_disc = args.learning_rate_disc
        lr_gen = args.learning_rate_gen
    elif mode == "detGAN":
        # FOR NOW re-use same name to simplify eval script calls
        lr_gen = args.learning_rate_gen
    noise_channels = args.noise_channels
    add_noise = args.add_postprocessing_noise

    num_epochs = int(num_samples/(steps_per_epoch * batch_size))
    epoch = 1

    # noise_dim = (10,10) + (noise_channels,)

    # number of constant fields
    constant_fields = 2

    if not save_weights_root:
        save_weights_root = path + "/../models"

    if args.problem_type == "normal":
        downsample = False
        input_channels = 9
    elif args.problem_type == "superresolution":
        downsample = True
        input_channels = 1
    else:
        raise Exception("no such problem type, try again!")

    if args.do_training:
        if mode == "GAN":
            # initialize GAN
            print(f"val years is {val_years}")
            model, batch_gen_train, batch_gen_valid, _, noise_shapes, _ = \
                train.setup_gan(train_years,
                                val_years,
                                val_size=val_size,
                                downsample=downsample,
                                weights=weights,
                                input_channels=input_channels,
                                constant_fields=constant_fields,
                                batch_size=batch_size,
                                filters_gen=filters_gen,
                                filters_disc=filters_disc,
                                noise_channels=noise_channels,
                                lr_disc=lr_disc,
                                lr_gen=lr_gen)

        elif mode == "detGAN":
            # initialize deterministic model
            model, batch_gen_train, batch_gen_valid, _, _ = \
                train.setup_deterministic(train_years,
                                          val_years,
                                          val_size=val_size,
                                          downsample=downsample,
                                          weights=weights,
                                          input_channels=input_channels,
                                          constant_fields=constant_fields,
                                          steps_per_epoch=steps_per_epoch,
                                          batch_size=batch_size,
                                          filters_gen=filters_gen,
                                          lr=lr_gen)

        else:
            assert False, "other modes not implemented yet"

        if load_weights_root:  # load weights and run status
            model.load(model.filenames_from_root(load_weights_root))
            with open(load_weights_root + "-run_status.json", 'r') as f:
                run_status = json.load(f)
            training_samples = run_status["training_samples"]

            if log_path:
                log_file = "{}/log.txt".format(log_path)
                log = pd.read_csv(log_file)

        else:  # initialize run status
            training_samples = 0

            if log_path:
                log_file = "{}/log.txt".format(log_path)
                # TODO: does VAEGAN need more?
                if mode == "GAN":
                    log = pd.DataFrame(
                        columns=["training_samples",
                                 "disc_loss", "disc_loss_real",
                                 "disc_loss_fake", "disc_loss_gp",
                                 "gen_loss"])
                elif mode == "detGAN":
                    log = pd.DataFrame(columns=["training_samples",
                                                "loss"])
                else:
                    assert False

        plot_fn = "{}/progress_{}.pdf".format(log_path, mode) if log_path \
            else path + "/../figures/progress.pdf"

        eval_fn_small = "{}/eval-small.txt".format(log_path) if log_path \
            else path + "/../figures/eval-small.txt"

        eval_fn_full = "{}/eval-full.txt".format(log_path) if log_path \
            else path + "/../figures/eval-full.txt"

        qual_fn_small = "{}/qual-small.txt".format(log_path) if log_path \
            else path + "/../figures/qual-small.txt"

        qual_fn_full = "{}/qual-full.txt".format(log_path) if log_path \
            else path + "/../figures/qual-full.txt"

        while (training_samples < num_samples):  # main training loop

            print("Epoch {}/{}".format(epoch, num_epochs))

            # train for some number of batches
            if mode == "GAN":
                loss_log = train.train_gan(model,
                                           batch_gen_train,
                                           batch_gen_valid,
                                           noise_shapes,
                                           epoch,
                                           steps_per_epoch,
                                           num_epochs=1,
                                           plot_samples=val_size,
                                           plot_fn=plot_fn)
            elif mode == "detGAN":
                loss_log = train.train_deterministic(model,
                                                     batch_gen_train,
                                                     batch_gen_valid,
                                                     epoch,
                                                     steps_per_epoch,
                                                     num_epochs=1,
                                                     plot_samples=val_size,
                                                     plot_fn=plot_fn)
            else:
                assert False

            loss_log = np.mean(loss_log, axis=0)
            training_samples += steps_per_epoch * batch_size
            epoch += 1

            # save results
            model.save(save_weights_root)
            run_status = {
                "training_samples": training_samples,
            }
            with open(save_weights_root+"-run_status.json", 'w') as f:
                json.dump(run_status, f)

            if log_path:  # log losses and generator weights for evaluation
                if mode == "GAN":
                    log = log.append(pd.DataFrame(data={
                        "training_samples": [training_samples],
                        "disc_loss": [loss_log[0]],
                        "disc_loss_real": [loss_log[1]],
                        "disc_loss_fake": [loss_log[2]],
                        "disc_loss_gp": [loss_log[3]],
                        "gen_loss": [loss_log[4]]
                    }))
                elif mode == "detGAN":
                    log = log.append(pd.DataFrame(data={
                        "training_samples": [training_samples],
                        "loss": [loss_log],
                    }))
                else:
                    assert False
                log.to_csv(log_file, index=False, float_format="%.6f")

                if mode == "GAN":
                    gen_weights_file = "{}/gen_weights-{:07d}.h5".format(
                        log_path, training_samples)

                    model.gen.save_weights(gen_weights_file)
                elif mode == "detGAN":
                    gen_weights_file = "{}/gen_det_weights-{:07d}.h5".format(
                        log_path, training_samples)

                    model.gen_det.save_weights(gen_weights_file)
                else:
                    assert False
    else:
        print("Training skipped...")

    # evaluate model performance
    if args.eval_small:
        evaluation.rank_metrics_by_time(mode,
                                        val_years,
                                        out_fn=eval_fn_small,
                                        weights_dir=log_path,
                                        check_every=1,
                                        N_range=None,
                                        downsample=downsample,
                                        weights=weights,
                                        add_noise=add_noise,
                                        load_full_image=False,
                                        model_number=None,
                                        batch_size=batch_size,
                                        num_batches=num_batches,
                                        filters_gen=filters_gen,
                                        filters_disc=filters_disc,
                                        input_channels=input_channels,
                                        constant_fields=constant_fields,
                                        noise_channels=noise_channels,
                                        rank_samples=100,
                                        lr_disc=lr_disc,
                                        lr_gen=lr_gen)

    if args.eval_full:
        evaluation.rank_metrics_by_time(mode,
                                        val_years,
                                        out_fn=eval_fn_full,
                                        weights_dir=log_path,
                                        check_every=1,
                                        N_range=None,
                                        downsample=downsample,
                                        weights=weights,
                                        add_noise=add_noise,
                                        load_full_image=True,
                                        model_number=None,
                                        batch_size=1,  # memory issues
                                        num_batches=num_batches,
                                        filters_gen=filters_gen,
                                        filters_disc=filters_disc,
                                        input_channels=input_channels,
                                        constant_fields=constant_fields,
                                        noise_channels=noise_channels,
                                        rank_samples=100,
                                        lr_disc=lr_disc,
                                        lr_gen=lr_gen)

    if args.qual_small:
        evaluation.quality_metrics_by_time(mode,
                                           val_years,
                                           out_fn=qual_fn_small,
                                           weights_dir=log_path,
                                           check_every=1,
                                           downsample=downsample,
                                           weights=weights,
                                           load_full_image=False,
                                           batch_size=batch_size,
                                           num_batches=num_batches,
                                           filters_gen=filters_gen,
                                           filters_disc=filters_disc,
                                           input_channels=input_channels,
                                           constant_fields=constant_fields,
                                           noise_channels=noise_channels,
                                           lr_disc=lr_disc,
                                           lr_gen=lr_gen)

    if args.qual_full:
        evaluation.quality_metrics_by_time(mode,
                                           val_years,
                                           out_fn=qual_fn_full,
                                           weights_dir=log_path,
                                           check_every=1,
                                           downsample=downsample,
                                           weights=weights,
                                           load_full_image=True,
                                           batch_size=1,  # memory issues
                                           num_batches=num_batches,
                                           filters_gen=filters_gen,
                                           filters_disc=filters_disc,
                                           input_channels=input_channels,
                                           constant_fields=constant_fields,
                                           noise_channels=noise_channels,
                                           lr_disc=lr_disc,
                                           lr_gen=lr_gen)
