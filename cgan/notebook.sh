#!/bin/bash

#SBATCH --qos=priority
#SBATCH --partition dmm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=jupyter_launch

# Don't change the names of the output files:
#SBATCH --output=jupyter-%j.log
#SBATCH --error=jupyter-%j.err

# Some initial setup
# module purge
module load anaconda/5.0.0_py3 # this is where the Jupyter client comes from

# optional: if you have a Conda env, activate it here:
conda activate alpine

# Load an R module, if you're running R notebooks. See README.md for setup instructions
#module load R/3.6.3
# Load a Julia module, if you're running Julia notebooks
#module load julia/1.3.0 
# Load this if you want to export Notebooks to PDF:
#module load texlive

# set a random port for the notebook, in case multiple notebooks are
# on the same compute node.
NOTEBOOKPORT=`shuf -i 18000-18500 -n 1`

# set a random port for tunneling, in case multiple connections are happening
# on the same login node.
TUNNELPORT=`shuf -i 18501-19000 -n 1`

# Set up a reverse SSH tunnel from the compute node back to the submitting host (login01 or login02)
# This is the machine we will connect to with SSH forward tunneling from our client.
ssh -R$TUNNELPORT:localhost:$NOTEBOOKPORT $SLURM_SUBMIT_HOST -N -f

echo "FWDSSH='ssh -L8888:localhost:$TUNNELPORT $(whoami)@$SLURM_SUBMIT_HOST.pik-potsdam.de -N'"

# Start the notebook
srun -n1 jupyter-notebook --no-browser --no-mathjax --port=$NOTEBOOKPORT

wait
