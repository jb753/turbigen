#!/bin/bash
#SBATCH -J jobname
#SBATCH -p ampere
#SBATCH -A PULLAN-MHI-SL2-GPU
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00


export TURBIGEN_ROOT='/rds/project/gp10006/rds-gp10006-pullan-mhi/jb753/turbigen'
export PYTHONPATH=$PYTHONPATH:$TURBIGEN_ROOT
source /usr/local/software/turbostream/ts3610_a100/bashrc_module_ts3610_a100
python -u -c 'from turbigen import submit, turbostream; submit._run_search(turbostream.write_grid_from_params)' &> log_tabu.txt

# cd opt_row_1
# python -u $TURBIGEN_ROOT/turbigen/post_process_turbostream output_avg.hdf5 --plot
