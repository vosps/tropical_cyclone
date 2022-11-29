# Tropical Cyclone Downscaling

This repo contains code for the U-NET, VAEGAN and WGAN all used to downscale tropical cyclone rainfall from 100 km resolution to 10 km resolution. Trained models are also included.

## Input Data

This version of the models require input data to be in 100 km resolution arrays of shape (nstorms,nlats,nlons) centred around the tropical cyclone.

## WGAN model

Running the code

1. Training the model

First, set up your model parameters in the configuration (.yaml) file.
An example is provided in the main directory. We recommend copying this 
file to somewhere on your local machine before training.

Run the following to start the training:

python main.py --config <path/to/config/file>

There are a number of options you can use at this point. These will 
evaluate your model after it has finished training:

--rank flags will run CRPS/rank based evalution
--qual flags will run image quality based evalution
--plot_ranks will plot rank histograms
	     N.B. you must run rank based eval first
--plot_roc will plot a Receiver Operating Characteristic curve (and by
	   default also a curve for ecPoint for comparison)
	   
These flags all have a small and a full option for operating on the small 
image dataset and full image dataset, respectively.

If you choose to run --rank, --qual or --plot_roc, you must
also specify if you want to do this for all model iterations or 
just a selection. Do this using 

- --eval_full	  (all model iterations)
- --eval_short	  (a subset of 16 model iterations)
- --eval_blitz	  (a subset of 4 model iterations)

Two things to note:
- These three options work well with 100 * 3200 training samples that we 
have been working with. If these numbers change, you may want to update
them accordingly.
- Calculating everything, for all model iterations, will take a long 
time. Possibly weeks. You have been warned. 

As an example, to train a model and evaluate it fully but for only a 
select few model iterations, on full images, you could run:

python main.py --config <path/to/config/file> --eval_blitz --rank_full
--qual_full --plot_ranks_full --plot_roc_full

2. If you've already trained your model, and you just want to run some 
evaluation, use the --no_train flag, for example:

python main.py --config <path/to/config/file> --no_train --eval_full 
--plot_roc_small

3. To generate plots of the output from a trained model, use predict.py
This requires a path to the directory where the weights of the model are 
stored, and will read in the setup parameters directly. Use the following
arguments:

Necessary arguments:
- --log_folder	    <path/to/model/directory>
- --model_number    model iteration you want to use to predict

Optional arguments:
- --num_predictions   number of different input images to predict on
- --num_samples	    number of predictions to draw from ensemble
		    N.B. if you run mode == 'det' we have hardcoded this
		    to 1 for obvious reasons.

There are also the following optional arguments:
- --predict_full_image  to predict on the full image dataset
- --include_Lanczos     	includes Lanczos interpolation prediction
- --include_ecPoint     	includes ecPoint-based prediction
- --include_RainFARM    	includes RainFARM prediction
- --include_deterministic includes deterministic prediction for comparison.
			N.B. Probably don't use this for comparing against
			a deterministic model. 

For example:

python predict.py --log_folder <path/to/model> --model_number 0006400 
--num_predictions 7 --predict_full_image --include_ecPoint

4. If, for whatever reason, you don't want to go through main.py but you
want to do some evaluation on specific models only, you can use the scripts

- run_eval.py
- run_qual.py
- run_roc.py

You'll have to open them and hardcode file paths and model numbes at the
top. I also made a plot_comparisons.py at one point but I've deleted it so
now you have to use predict.py. Sorry.

5. If you want to change model architectures, add them to the models.py
file. You can then use the config.yaml file to call a different model
architecture. We recommend not deleting the old one just yet.

6. run_benchmarks.py will generate benchmark scores for CRPS, RMSE, MAE
and RAPSD for the specified benchmark models (input arguments). We have
set this up to use the same evaluation setup as the NN models but note
the benchmark evaluation uses 100 ensemble members (NN default is 10 for
time & memory reasons). So, for a proper comparison you *must* evaluate
your model and specify 100 ensemble members.

For example:
python run_benchmarks.py --log_folder /path/to/model --include_Lanczos 
--include_ecPoint --include_RainFARM --include_ecPoint_mean 
--include_constant --include_zeros

7. This is research code so please let us know if something is wrong and
also note that it definitely isn't perfect :)
