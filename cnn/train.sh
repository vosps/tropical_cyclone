#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-71:00:00
#SBATCH --mem=32gb
#SBATCH --gres=gpu:A100:1
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

# nvidia-smi --query-gpu=gpu_name,driver_version,memory.free,memory.total --format=csv
# A5000 A40 on magma 
# rtx_3090 A100
# A100 works on mlcnu

source activate /user/work/al18709/.conda/envs/jungle

# poetry shell
nvidia-smi
# module load lang/cuda/11.2-cudnn-8.1
# module load lang/cuda/10.2.89
# module load lib/cudnn/8.0
nvcc -V
echo alpine environment activated
echo running cgan
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
python train.py
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"