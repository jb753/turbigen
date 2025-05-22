#!/bin/bash 
# Profile emb solver
export OMP_NUM_THREADS=1
LINE_PROFILE=1 turbigen examples/axial_turbine.yaml -I
