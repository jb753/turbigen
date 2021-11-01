#!/bin/bash
#SBATCH -J turbogen
#SBATCH -p ampere
#SBATCH -A PULLAN-MHI-SL2-GPU
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
##SBATCH --requeue

# . /etc/profile.d/modules.sh
# module purge
# module load rhel8/default-amp
source /usr/local/software/turbostream/ts3610_a100/bashrc_module_ts3610_a100

# run steady
mpirun -npernode 1 -np 1 turbostream input_1.hdf5 output_1 1 > log_1.txt

# convert to unsteady
python ../../convert_unsteady.py output_1.hdf5 input_2.hdf5

# run unsteady
mpirun -npernode 1 -np 1 turbostream input_2.hdf5 output_2 1 > log_2.txt

