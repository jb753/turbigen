#!/usr/bin/env pvpython
# Usage: convert_ts3_to_ts4.py INPUT_HDF5 OUTPUT_STEM
import os
import numpy
import sys
import vtk
import ts.process.ts3_reader
import ts.process.pre.vtk_adaptor
import ts.process.pre.vtk_util
import ts.util

if not len(sys.argv) == 3:
    print("Usage: convert_ts3_to_ts4.py INPUT_HDF5 OUTPUT_STEM")
    exit(1)


def select_by_string(vtk_grid, string):

    obj_name_key = vtk.vtkCompositeDataSet.NAME()

    # Clear selection
    ts.process.pre.vtk_util.clear_selection_block(vtk_grid)

    # Prepare to iterate
    mb_root = ts.process.pre.vtk_util.get_root_block(vtk_grid)
    iter = mb_root.NewIterator()
    iter.VisitOnlyLeavesOff()
    iter.InitTraversal()

    # Loop over things
    found_indices = []
    while not iter.IsDoneWithTraversal():

        meta = iter.GetCurrentMetaData()
        flat_index = iter.GetCurrentFlatIndex()

        # Compare object name to the search string
        obj_name = meta.Get(obj_name_key)
        if string in obj_name:
            found_indices.append(flat_index)
            print(f"Selected {obj_name} at index {flat_index}")

        iter.GoToNextItem()

    # Filter the grid using found ids
    vtk_grid = ts.process.pre.extract.filter(vtk_grid, found_indices, False)

    return vtk_grid


def set_group(vtk_grid, group_id):

    # Loop through selected bcells
    for bcell in ts.process.pre.vtk_util.get_selected_bcells(vtk_grid):

        # Copy the bcell and add the group array
        obj_copy = ts.process.pre.vtk_util.copy_bcell(bcell.obj, copy_arrays=False)
        ncell = obj_copy.GetNumberOfCells()
        group_array = numpy.zeros(ncell, ts.util.format.np_int) + group_id
        obj_copy.CellData.append(group_array, "group")

        # Update grid with the copied bcell
        ts.process.pre.vtk_util.update_bcell(
            vtk_grid, bcell.domain, bcell.id, bcell.bcond_kind, obj_copy
        )

        print(f"Set group={group_id}")

    return vtk_grid


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

# Set groups for potential unsteady forcing
# Find the inlet and set to group 0
vtk_grid = select_by_string(vtk_grid, "Inlet")
vtk_grid = set_group(vtk_grid, 0)

# Find the inlet and set to group 0
vtk_grid = select_by_string(vtk_grid, "Outlet")
vtk_grid = set_group(vtk_grid, 1)

# Write TS4
g = ts.process.pre.vtk_adaptor.create_ts_grid(vtk_grid)
g.write_hdf5(f"{output_stem_full}.hdf5")
