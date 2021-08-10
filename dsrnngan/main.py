import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
#from tensorflow.keras.optimizers import SGD

import train
import eval
import plots


path = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help="ensemble, plot, deterministic")
    parser.add_argument('--problem_type', type=str, help="normal (IFS>NIMROD), easy (NIMROD>NIMROD)", default="normal")
    parser.add_argument('--application', type=str, default='IFS')
    parser.add_argument('--train_years', type=int, nargs='+', default=[2018],
                        help="Training years")
    parser.add_argument('--val_years', type=int, nargs='+', default=2019,
                        help="Validation years -- cannot pass a list if using create_fixed_dataset")
    parser.add_argument('--val_size', type=int, default=8,
                        help='Num val examples')
    parser.add_argument('--load_weights_root', type=str, default="",
        help="Network weights file root to load")
    parser.add_argument('--save_weights_root', type=str, default="",
        help="Network weights file root to save")
    parser.add_argument('--log_path', type=str, default="",
        help="Log files path")
    parser.add_argument('--num_samples', type=int, default=400000,
        help="Training samples")
    parser.add_argument('--steps_per_epoch', type=int, default=200,
        help="Batches per epoch")
    parser.add_argument('--batch_size', type=int, default=16,
        help="Batch size")
    parser.add_argument('--num_batches', type=int, default=64,
        help="Number of batches for eval metrics")
    parser.add_argument('--filters_gen', type=int, default=64,
        help="Number of filters used in generator")
    parser.add_argument('--filters_disc', type=int, default=64,
        help="Number of filters used in discriminator")
    parser.add_argument('--noise_channels', type=int, default=8,
        help="Dimensions of noise passed to generator")
    parser.add_argument('--learning_rate_disc', type=float, default=1e-4,
        help="Learning rate used for discriminator optimizer")
    parser.add_argument('--learning_rate_gen', type=float, default=1e-4,
        help="Learning rate used for generator optimizer")
    parser.add_argument('--add_postprocessing_noise', type=bool, default=True,
        help="Flag for adding postprocessing noise in rank statistics eval")
    parser.add_argument('--load_full_image', type=bool, default=False,
        help="Flag for evaluating and predicting on full-size image")
        
    args = parser.parse_args()
    mode = args.mode
    
    #weights = np.arange(12,2,-3)
    #weights = weights / weights.sum()
    #weights = None
    #weights = [0.87, 0.06, 0.03, 0.03]
    weights = np.arange(24,2,-7)
    weights = weights / weights.sum()
    print(f"weights for data loading are {weights}")
    if mode=="ensemble":

        load_weights_root = args.load_weights_root
        save_weights_root = args.save_weights_root
        log_path = args.log_path
        steps_per_epoch = args.steps_per_epoch
        batch_size = args.batch_size
        val_size = args.val_size
        num_samples = args.num_samples
        train_years = args.train_years
        val_years = args.val_years
        application = args.application
        num_batches = args.num_batches
        filters_disc = args.filters_disc
        filters_gen = args.filters_gen
        lr_disc = args.learning_rate_disc
        lr_gen = args.learning_rate_gen
        noise_channels = args.noise_channels
        problem_type = args.problem_type
        add_noise = args.add_postprocessing_noise
        load_full_image = args.load_full_image
        
        num_epochs = int(num_samples/(steps_per_epoch * batch_size))
        epoch = 1
        #noise_dim = (10,10) + (noise_channels,)
        
        ##number of constant fields
        constant_fields = 2

        if not save_weights_root:
            save_weights_root = path + "/../models"

        if problem_type == "normal":
            downsample = False
            input_channels = 9
        elif problem_type == "easy":
            downsample = True
            input_channels = 1
        else:
            raise Exception("no such problem type, try again!")
    
        # initialize GAN
        print(f"val years is {val_years}")
        (wgan, batch_gen_train, batch_gen_valid, _, noise_shapes, _) = \
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

        if load_weights_root: # load weights and run status
            wgan.load(wgan.filenames_from_root(load_weights_root))
            with open(load_weights_root+"-run_status.json", 'r') as f:
                run_status = json.load(f)
            training_samples = run_status["training_samples"]

            if log_path:
                log_file = "{}/log-{}.txt".format(log_path,application)
                log = pd.read_csv(log_file)

        else: # initialize run status
            training_samples = 0

            if log_path:
                log_file = "{}/log-{}.txt".format(log_path,application)
                log = pd.DataFrame(columns=["training_samples", 
                    "disc_loss", "disc_loss_real", "disc_loss_fake",
                    "disc_loss_gp", "gen_loss"])

        plot_fn = "{}/progress_{}-{}.pdf".format(log_path,mode,application) if log_path \
            else path+"/../figures/progress.pdf"

        eval_fn = "{}/eval-{}.txt".format(log_path, application) if log_path \
            else path+"/../figures/eval.txt"

        qual_fn = "{}/qual-{}.txt".format(log_path, application) if log_path \
            else path+"/../figures/eval.txt"


        while (training_samples < num_samples): # main training loop
            print("Epoch {}/{}".format(epoch, num_epochs))

            # train for some number of batches
            loss_log = train.train_gan(wgan, 
                                       batch_gen_train, 
                                       batch_gen_valid, 
                                       noise_shapes, 
                                       epoch, 
                                       steps_per_epoch, 
                                       num_epochs=1, 
                                       plot_samples=val_size, 
                                       plot_fn=plot_fn)

            loss_log = np.mean(loss_log, axis=0)
            training_samples += steps_per_epoch * batch_size
            epoch += 1

            # save results
            wgan.save(save_weights_root)
            run_status = {
                "application": application,
                "training_samples": training_samples,
            }
            with open(save_weights_root+"-run_status.json", 'w') as f:
                json.dump(run_status, f)

            if log_path: # log losses and generator weights for evaluation
                log = log.append(pd.DataFrame(data={
                    "training_samples": [training_samples],
                    "disc_loss": [loss_log[0]],
                    "disc_loss_real": [loss_log[1]],
                    "disc_loss_fake": [loss_log[2]],
                    "disc_loss_gp": [loss_log[3]],
                    "gen_loss": [loss_log[4]]
                }))
                log.to_csv(log_file, index=False, float_format="%.6f")
                        
                gen_weights_file = "{}/gen_weights-{}-{:07d}.h5".format(
                    log_path, application, training_samples)
                wgan.gen.save_weights(gen_weights_file)
       
        #evaluate model performance        
        eval.rank_metrics_by_time(mode, 
                                  val_years, 
                                  application, 
                                  out_fn=eval_fn, 
                                  weights_dir=log_path, 
                                  check_every=1, 
                                  N_range=None, 
                                  downsample=downsample,
                                  weights=weights,
                                  add_noise=add_noise,
                                  load_full_image=load_full_image,
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
    
        eval.quality_metrics_by_time(mode, 
                                     train_years, 
                                     val_years, 
                                     application, 
                                     out_fn=qual_fn,
                                     weights_dir=log_path, 
                                     check_every=1, 
                                     downsample=downsample,
                                     weights=weights,
                                     batch_size=batch_size, 
                                     num_batches=num_batches, 
                                     filters_gen=filters_gen, 
                                     filters_disc=filters_disc,
                                     input_channels=input_channels,
                                     constant_fields=constant_fields,
                                     noise_channels=noise_channels,  
                                     lr_disc=lr_disc, 
                                     lr_gen=lr_gen)

    elif mode == "deterministic":
        load_weights_root = args.load_weights_root
        save_weights_root = args.save_weights_root
        log_path = args.log_path
        steps_per_epoch = args.steps_per_epoch
        batch_size = args.batch_size
        val_size = args.val_size
        num_samples = args.num_samples
        train_years = args.train_years
        val_years = args.val_years
        application = args.application
        num_batches = args.num_batches
        filters_gen = args.filters_gen
        learning_rate = args.learning_rate_gen
        problem_type = args.problem_type
        add_noise = args.add_postprocessing_noise
        load_full_image = args.load_full_image
        num_epochs = int(num_samples/(steps_per_epoch * batch_size))
        epoch = 1
        
        ##number of constant fields
        constant_fields = 2

        if not save_weights_root:
            save_weights_root = path + "/../models"
        
        if problem_type == "normal":
            downsample = False
            input_channels = 9
        elif problem_type == "easy":
            downsample = True
            input_channels = 1
        else:
            raise Exception("no such problem type, try again!")

        # initialize deterministic model
        (det_model, batch_gen_train, batch_gen_valid, _, _) = \
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
                                      lr=learning_rate)
   

        if load_weights_root: # load weights and run status
            det_model.load(det_model.filenames_from_root(load_weights_root))
            with open(load_weights_root+"-run_status.json", 'r') as f:
                run_status = json.load(f)
            training_samples = run_status["training_samples"]

            if log_path:
                log_file = "{}/log-{}.txt".format(log_path,application)
                log = pd.read_csv(log_file)

        else: # initialize run status
            training_samples = 0

            if log_path:
                log_file = "{}/log-{}.txt".format(log_path,application)
                log = pd.DataFrame(columns=["training_samples", "loss"])

        plot_fn = "{}/progress_{}-{}.pdf".format(log_path,mode,application) if log_path \
            else path+"/../figures/progress.pdf"

        eval_fn = "{}/eval-{}.txt".format(log_path, application) if log_path \
            else path+"/../figures/eval.txt"
        
        qual_fn = "{}/qual-{}.txt".format(log_path, application) if log_path \
            else path+"/../figures/eval.txt"

        while (training_samples < num_samples): # main training loop
            
            print("Epoch {}/{}".format(epoch, num_epochs))
            
            # train for some number of batches
            loss_log = train.train_deterministic(det_model, 
                                                 batch_gen_train, 
                                                 batch_gen_valid, 
                                                 epoch, 
                                                 steps_per_epoch, 
                                                 1, 
                                                 plot_samples=val_size,
                                                 plot_fn=plot_fn)
            loss_log = np.mean(loss_log)
            training_samples += steps_per_epoch * batch_size
            epoch += 1

            # save results
            det_model.save(save_weights_root)
            run_status = {
                "application": application,
                "training_samples": training_samples,
            }
            with open(save_weights_root+"-run_status.json", 'w') as f:
                json.dump(run_status, f)

            if log_path: # log losses and generator weights for evaluation
                log = log.append(pd.DataFrame(data={
                    "training_samples": [training_samples],
                    "loss": [loss_log],
                }))
                log.to_csv(log_file, index=False, float_format="%.6f")
                        
                gen_det_weights_file = "{}/gen_det_weights-{}-{:07d}.h5".format(
                    log_path, application, training_samples)
                det_model.gen_det.save_weights(gen_det_weights_file)

        #evaluate model performance              
        eval.rank_metrics_by_time(mode, 
                                  train_years, 
                                  val_years, 
                                  application, 
                                  out_fn=eval_fn,
                                  weights_dir=log_path, 
                                  check_every=1, 
                                  N_range=None,
                                  downsample=downsample,
                                  weights=weights,
                                  add_noise=add_noise,
                                  load_full_image=load_full_image,
                                  model_number=None,
                                  batch_size=batch_size, 
                                  num_batches=num_batches, 
                                  filters_gen=filters_gen,
                                  input_channels=input_channels,
                                  constant_fields=constant_fields,
                                  rank_samples=1, 
                                  lr_gen=learning_rate)
        
        eval.quality_metrics_by_time(mode, 
                                     train_years, 
                                     val_years, 
                                     application, 
                                     out_fn=qual_fn, 
                                     weights_dir=log_path, 
                                     check_every=1, 
                                     downsample=downsample,
                                     weights=weights,
                                     batch_size=batch_size, 
                                     num_batches=num_batches, 
                                     filters_gen=filters_gen,
                                     input_channels=input_channels,
                                     constant_fields=constant_fields,
                                     lr_gen=learning_rate)
