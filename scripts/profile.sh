#!/bin/bash 
# Profile emb solver
PYFILE="src/turbigen/solvers/emb.py"
sed -i '/def run_slave(/i @profile' "$PYFILE"
kernprof -l turbigen examples/turbine_cascade.yaml
mkdir -p plots
python -m line_profiler -rmt "turbigen.lprof" > plots/profile.txt
sed -i '/@profile/d' "$PYFILE"
