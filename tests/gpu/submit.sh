#!/bin/bash
#SBATCH -J turbigen_gpu_test
#SBATCH -p ampere
#SBATCH -A brind-sl3-gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00

# Find all configs in the gpu test directory
# and run turbigen on each one
for config in tests/gpu/*.yaml ; do
	echo "Running $config"
	uv run turbigen --no-job $config
done
