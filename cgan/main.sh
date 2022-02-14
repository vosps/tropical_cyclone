#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cgan
#SBATCH --partition mlcnu
source ~/.bashrc
cd $SLURM_SUBMIT_DIR
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOBID
echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST
module load lang/python/anaconda/3.7-2019.03
module load lang/cdo/1.9.8-gcc
module load lib/cudnn/11.2
module load lang/cuda/10.1.105
conda activate alpine
echo alpine environment activated
echo running cgan
python main.py --config config-example.yaml --eval_blitz
# python main.py --config config-example.yaml --no_train --eval_blitz 
# srun --nodes=1 --ntasks-per-node=1 --mem=16gb --partition mlcnu --time=03:00:00 --pty bash -i
# srun --nodes=1 --ntasks-per-node=1 --gres=gpu:2 --mem=32gb --partition mlcnu --time=03:00:00 --pty bash -i
# srun --nodes=1 --ntasks-per-node=1 --mem=16gb --gres=gpu:1 --partition mlcnu --time=03:00:00 --pty bash -i
# srun --nodes=1 --ntasks-per-node=1 --mem=16gb --partition dmm --time=03:00:00 --pty bash -i
# srun --nodes=2 --ntasks-per-node=1 --mem=32gb --partition dmm --time=03:00:00 --pty bash -i
# module load cray-netcdf