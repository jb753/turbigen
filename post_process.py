"""Read a Turbostream flow solution and calculate quantities of interest.

Call by running in termianl:

python post_process.py path/to/output.hdf5

"""

# Import some modules we will need
import sys # System commands
from ts import ts_tstream_reader, ts_tstream_cut  # Turbostream libraries
import matplotlib.pyplot as plt  # Plotting library

# Arbitrary reference temperatures for entropy calculation
Pref = 1e5
Tref = 300.0

# Get output file name from command line arguments
output_hdf5 = sys.argv[1]  # The second argument

print("POST-PROCESSING %s\n" % output_hdf5)

# Load the grid from hdf5 file
tsr = ts_tstream_reader.TstreamReader()  # Reader class
g = tsr.read(output_hdf5)  # Use the class to read into grid object

# The grid object g contains everything we might want to know about the flow
# solution. We need to pull out relevant bits of the data we need.

# The gas properties are stored as "application variables" - scalar values that
# are the same everywhere
cp = g.get_av("cp")  # Specific heat capacity
ga = g.get_av("ga")  # Specific heat ratio
rgas = cp * (1.0 - 1.0 / ga)  # Gas constant

# The grid is made of two blocks of 3D matrices
# id 0 - stator row
# id 1 - rotor row
bid_stator = 0
bid_rotor = 1
bid_all = [bid_stator, bid_rotor]

# g.get_block(bid) will extract just one block id for further processing
#
# We store them in a list comprehension for easy access
blks = [g.get_block(bid) for bid in bid_all]

# We need to know the number of grid points in each block to look at the flow
# field. So use a list comprehension to loop over all blocks and get the 
# attributes
#
# ni for streamwise
# nj for radial
# nk for pitchwise
ni = [blk.ni for blk in blks]
nj = [blk.nj for blk in blks]
nk = [blk.nk for blk in blks]
# Later, we can recall, for example, how many radial points are in the
# rotor block by using nj[bid_rotor]

# Suppose we want to take a two-dimensional slice through the flow field in the
# x-rt plane at a constant radius. This means we want to "cut" the flow at a
# fixed j-index, but include all i and k values.

# i streamwise, j spanwise, k pitchwise
# start and stop like range()
c = ts_tstream_cut.TstreamStructuredCut()  # Initialise a blank cut
c.read_from_grid(
    g,  # Take flow properties from grid object we loader earlier
    Pref,  # Reference pressure for entropy (arbitray)
    Tref,   # Reference temperature for entropy (arbitray)
    bid_stator,  # Cut in the stator block
    ist=0,  # Start cutting at first streamwise index
    ien=ni[bid_stator],  # Stop cutting at last streamwise index
    jst=0,  # Start cutting at first radial index
    jen=1,  # Stop cutting at second radial index (i.e. a slice one point thick)
    kst=0,  # Start cutting at first pitchwise location
    ken=nk[bid_stator],  # Stop cutting at last pitchwise location.
)

# Now plot contours of density
fig, ax = plt.subplots()  # Make a blank figure
ax.contourf(c.x,c.rt,c.ro)  # Filled contour plot of density on x-rt plane
ax.axis('equal')  # Real aspect ratio
plt.show()  # Render figure to screen
