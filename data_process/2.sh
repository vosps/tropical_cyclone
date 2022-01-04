#!/bin/bash
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=64:ompthreads=4:mem=128gb
#PBS -j oe
cd $PBS_O_WORKDIR
module load lang/python/anaconda/3.7-2019.03
module load lang/cdo/1.9.8-gcc
source activate alpine
python 2_extract_tc_p.py