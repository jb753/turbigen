"""Demonstrate how to read in saved cuts for further analysis.

Using the `write_cuts` post-processing function, we can save traverse or blade surface cuts of the flow field at the
completion of the turbigen run. This script reads in both types of cut and explains the file format.
"""

import numpy as np

# Read in traverse data
TRAVERSE_NPZ = "scripts/cut_traverse_2.npz"
print("Reading traverse data.")
traverse = np.load(TRAVERSE_NPZ)

# The type of fluid (Perfect or Real) is given by the 'class' key
print(f"Fluid type is: {traverse['class']}")

# Fluid properties and other scalars are stored as scalars
print(f'Number of blades Nb={traverse["Nb"]}')
print(f'Fluid properties ga={traverse["gamma"]}, cp={traverse["cp"]}')

# The 2D data is stored in a 3D array data:
# The properties on the first axis are listed in 'data_rows'
data = traverse["data"]
data_rows = traverse["data_rows"]
shape = data.shape
print(f"data shape = {shape} where:")
print(f"axis 0 indexes over {shape[0]} properties at each grid point")
print(f"the properties are: {data_rows}")
print(f"axis 1 indexes over {shape[1]} triangles")
print(f"axis 2 indexes over {shape[2]} vertices of each triangle")

# We can do a similar thing with blade surface data
SURF_NPZ = "scripts/cut_blade_00.npz"
print("Reading blade surface data.")
surf = np.load(SURF_NPZ)

# The 2D structured data is stored in a 3D array data
data = surf["data"]
data_rows = surf["data_rows"]
shape = data.shape
print(f"data shape = {shape} where:")
print(f"axis 0 indexes over {shape[0]} properties at each grid point")
print(f"the properties are: {data_rows}")
print(f"axis 1 indexes over {shape[1]} chordwise positions")
print(f"axis 2 indexes over {shape[2]} spanwise positions")
