#!/usr/bin/env pvpython
# Usage: convert_ts3_to_ts4.py INPUT_HDF5 OUTPUT_STEM
import os
import sys
import ts.process.ts3_reader
import ts.process.pre.vtk_adaptor

if not len(sys.argv) == 3:
    print("Usage: convert_ts3_to_ts4.py INPUT_HDF5 OUTPUT_STEM")
    exit(1)

# Make all paths absolute
input_hdf5, output_stem = sys.argv[1:3]
input_hdf5_full = os.path.abspath(input_hdf5)
output_stem_full = os.path.abspath(output_stem)

# Check for input and output files
if not os.path.exists(input_hdf5_full):
    print("Input %s not found" % input_hdf5_full)
    exit(1)
output_dir, _ = os.path.split(output_stem_full)
if not os.path.exists(output_dir):
    print("Output dir %s not found" % output_dir)
    exit(1)

# Read TS3 file
reader = ts.process.ts3_reader.TS3Reader()
vtk_grid = reader.read(input_hdf5_full, True)

# Write TS4
g = ts.process.pre.vtk_adaptor.create_ts_grid(vtk_grid)
g.write_hdf5(f"{output_stem_full}.hdf5")
