#!/bin/bash


#SBATCH -J c3k_downsample
#SBATCH -n 20 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 6:00:00 # Runtime 
#SBATCH -p conroy # Partition to submit to
#SBATCH --mem-per-cpu=2000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -o /n/regal/conroy_lab/bdjohnson/ckc/ckc/h5/logs/c3k_to_h5_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/regal/conroy_lab/bdjohnson/ckc/ckc/h5/logs/c3k_to_h5_%A_%a.err # Standard err goes to this file

ncpu=$SLURM_JOB_CPUS_PER_NODE

source activate pympi
cd /n/regal/conroy_lab/bdjohnson/ckc/ckc/
python downsample_h5.py --config R5K --np $ncpu --hname 'h5/c3k_v1.3_feh{:+3.2f}.full.h5'
