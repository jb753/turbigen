#!/bin/bash

if [[ $(hostname) =~ "cpu" ]]; then 
    echo "On HPC, loading modules"
    module purge
    module load rhel8/default-icl &> /dev/null
    unset I_MPI_PMI_LIBRARY
else
    echo "Not on HPC, no modules"
    export OMPI_MCA_btl_vader_single_copy_mechanism=none
fi


export OMP_NUM_THREADS=1
unset PYTHONDONTWRITEBYTECODE

make compile

rm -f tests/bench.dat

for size in 8 4 2 1 ; do
    mpirun --allow-run-as-root -np $size python tests/benchmark.py
done
