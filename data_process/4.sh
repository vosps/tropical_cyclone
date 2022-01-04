#!/bin/bash
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=32:ompthreads=4:mem=64gb
#PBS -j oe
cd $PBS_O_WORKDIR
module load lang/python/anaconda/3.7-2019.03
module load lang/cdo/1.9.8-gcc
source activate alpine
python 4_make_Xy_p.py