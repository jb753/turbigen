#!/bin/bash 
# Profile emb solver
PYFILE="src/turbigen/solvers/emb.py"
export OMP_NUM_THREADS=1
sed -i '/def run_slave(/i @profile' "$PYFILE"
uv pip uninstall turbigen && uv pip install --no-cache .
kernprof -l turbigen examples/axial_turbine.yaml -I
mkdir -p plots
python -m line_profiler -rmt "turbigen.lprof" > plots/profile.txt
sed -i '/@profile/d' "$PYFILE"
uv pip uninstall turbigen && uv pip install --no-cache .
