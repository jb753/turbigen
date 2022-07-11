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

export TURBIGEN_ROOT='/rds/project/gp10006/rds-gp10006-pullan-mhi/jb753/turbigen'
export PYTHONPATH="$TURBIGEN_ROOT:$PYTHONPATH"

# run steady
turbostream input.hdf5 output 1 > log.txt

# write out probe data file for dbslice
python -m turbigen.post_process_turbostream output_avg.hdf5

# remove extraneous files
rm stopit *.xdmf input.hdf5 #output.hdf5 log.txt
