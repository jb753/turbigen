#!/bin/bash 
# Profile emb solver
sed -i '/def run_slave(/i @profile' turbigen/solvers/emb.py
kernprof -l turbigen profile.yaml
mkdir -p plots
python -m line_profiler -rmt "turbigen.lprof" > plots/profile.txt
sed -i '/@profile/d'  turbigen/solvers/emb.py
