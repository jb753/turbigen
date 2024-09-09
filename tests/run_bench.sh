#!/bin/bash

export OMP_NUM_THREADS=1

SECONDS=0
python tests/benchmark.py &> /dev/null
SERIAL=$SECONDS
echo np=1 elapsed=$SERIAL


for size in 2 4 8; do
    SECONDS=0
    mpirun -np $size python tests/benchmark.py &> /dev/null
    PARA=$SECONDS
    echo np=$size elapsed=$PARA, speedup=$(echo "scale=2; $SERIAL/$PARA" | bc -l)
done
