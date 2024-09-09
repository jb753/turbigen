#!/bin/bash

module purge
module load rhel8/default-icl &> /dev/null


unset I_MPI_PMI_LIBRARY
export OMP_NUM_THREADS=1
unset PYTHONDONTWRITEBYTECODE

make compile-intel

rm -f tests/bench.dat

# for size in 8 4 2 1 ; do
for size in 8; do
    mpirun -np $size python tests/benchmark.py &> /dev/null
done
