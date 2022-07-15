#!/usr/bin/env python2
# usage: run_turbostream.py input_json
#
# From a parameters JSON file, start Turbostream on a free GPU, and then
# post-process the flow solution to get an output metadata JSON
#
# All of the Python2 stuff is called in this script

import sys, os, subprocess
from turbigen import turbostream, submit, post_process_turbostream

# Get argument
try:
    json_file_path = sys.argv[1]
except IndexError:
    raise Exception("No input file specified.")
    sys.exit(1)

# Work out some file names
workdir, json_file_name = os.path.split(os.path.abspath(json_file_path))
basedir = os.path.dirname(workdir)

gpu_id = 0

# Change to working dir
os.chdir(workdir)

# Read the parameters
param = submit.ParameterSet.from_json(json_file_name)

# Write the grid
input_file_name = "input.hdf5"
turbostream.write_grid_from_params(param, input_file_name)

print('written input hdf5')

output_prefix = "output"
# Start Turbostream
cmd_str = "CUDA_VISIBLE_DEVICES=%d turbostream %s %s 1 > log.txt" % (
    gpu_id,
    input_file_name,
    output_prefix,
)
subprocess.Popen(cmd_str, shell=True).wait()

# Post process
post_process_turbostream.post_process(output_prefix + "_avg.hdf5")

# Remove extraneous files
spare_files = [
    "stopit",
    output_prefix + ".xdmf",
    output_prefix + "_avg.xdmf",
    input_file_name,
]
for f in spare_files:
    try:
        os.remove(os.path.join(".", f))
    except OSError:
        pass
