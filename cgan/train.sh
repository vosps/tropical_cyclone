#!/bin/bash
#PBS -l walltime=6:00:00
#PBS -l select=1:ngpus=1:mem=16gb
#PBS -j oe
cd $PBS_O_WORKDIR
module load lang/python/anaconda/3.7-2019.03
module load lang/cdo/1.9.8-gcc
module load lib/cudnn/11.2
module load lang/cuda/11.2.2
source activate alpine
python main.py --config config-example.yaml