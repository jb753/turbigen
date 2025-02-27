#!/bin/bash 
sed -i '/def run_slave(/i @profile' turbigen/solvers/embsolve.py
kernprof -l turbigen tests/back-to-back/stage.yaml
python -m line_profiler -rmt "turbigen.lprof" > plots/profile.txt
