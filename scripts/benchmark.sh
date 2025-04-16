#!/bin/bash
# Script to run benchmark tests for emb solver

# Setup environment
if [[ $(hostname) =~ "cpu" ]]; then 
    echo "On HPC, loading modules"
    module purge
    module load rhel8/default-icl &> /dev/null
    unset I_MPI_PMI_LIBRARY
else
    echo "Not on HPC, no modules"
    export OMPI_MCA_btl_vader_single_copy_mechanism=none
fi
export OMP_NUM_THREADS=1  # Only one thread for each MPI process
unset PYTHONDONTWRITEBYTECODE  # Allow Python to write .pyc files

# Clear up files
mkdir -p plots
rm -f plots/bench.dat

# Run the benchmark for different nproc
for size in 8 6 4 2 1; do
# for size in 8 6 ; do
    mpirun --allow-run-as-root -np $size python scripts/benchmark.py
done

cat plots/bench.dat

# Make the plot
echo "Making the plots"
python scripts/plot_benchmark.py
echo "Done."
