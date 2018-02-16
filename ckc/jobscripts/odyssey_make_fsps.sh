#!/bin/bash

#SBATCH -J c3k_to_fsps
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 04:00:00 # Runtime 
#SBATCH -p conroy-intel # Partition to submit to
#SBATCH --mem-per-cpu=2000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -o /n/regal/conroy_lab/bdjohnson/run_ckc/logs/c3k_to_fsps_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/regal/conroy_lab/bdjohnson/run_ckc/logs/c3k_to_fsps_%A_%a.err # Standard err goes to this file

source activate pro
cd /n/regal/conroy_lab/bdjohnson/run_ckc/

seddir=sed_r500
mkdir $seddir
python ckc_to_fsps.py --zindex=${SLURM_ARRAY_TASK_ID} --ck_vers=c3k_v1.3 --spec_type=lores \
       --seddir=${seddir} --sedname=${seddir} --verbose=False
