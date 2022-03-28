#!/bin/bash
#SBATCH -J jobname
#SBATCH -p ampere
#SBATCH -A PULLAN-MHI-SL2-GPU
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
##SBATCH --requeue

source /usr/local/software/turbostream/ts3610_a100/bashrc_module_ts3610_a100

# run steady
mpirun -npernode 1 -np 1 turbostream input.hdf5 output 1 > log.txt

# write out probe data file for dbslice
python ../../turbigen/post_process_turbostream.py output_avg.hdf5
