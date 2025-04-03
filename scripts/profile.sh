#!/bin/bash 
sed -i '/def run_slave(/i @profile' turbigen/solvers/emb.py
kernprof -l turbigen profile.yaml
python -m line_profiler -rmt "turbigen.lprof" > profile.txt
sed -i '/@profile/{N;/@profile\ndef run_slave(/d;}'  turbigen/solvers/emb.py
