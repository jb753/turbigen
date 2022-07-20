"""This file contains functions for reading TS probe data."""
import numpy as np
import sys, os, json
from ts import ts_tstream_reader, ts_tstream_cut
import matplotlib.pyplot as plt

Pref = 1e5
Tref = 300.0

if __name__ == "__main__":

    # Get output file name from command line arguments
    output_hdf5 = sys.argv[1]
    basedir = os.path.dirname(output_hdf5)
    run_name = os.path.split(os.path.abspath(basedir))[-1]
    print("POST-PROCESSING %s\n" % output_hdf5)

    # Load the grid
    tsr = ts_tstream_reader.TstreamReader()
    g = tsr.read(output_hdf5)

    # Gas properties
    cp = g.get_av("cp")  # Specific heat capacity
    ga = g.get_av("ga")  # Specific heat ratio
    rgas = cp * (1.0 - 1.0 / ga)  # Gas constant

    # Collect numbers of grid points
    bids = np.array(g.get_block_ids())
    blks = [g.get_block(bid) for bid in bids]
    ni = [blk.ni for blk in blks]
    nj = [blk.nj for blk in blks]
    nk = [blk.nk for blk in blks]

    # Block ids
    bid_stator = 0
    bid_rotor = 1

    # Take cut at midspan
    # i streamwise, j spanwise, k pitchwise
    # start and stop like range()
    c = ts_tstream_cut.TstreamStructuredCut()
    c.read_from_grid(
        g,
        Pref,
        Tref,
        bid_stator,
        ist=0,
        ien=ni[0],  # All spanwise
        jst=0,
        jen=1,  # First radial
        kst=0,
        ken=nk[0],  # All pitchwise
    )

    fig, ax = plt.subplots()
    ax.contourf(c.x,c.rt,c.ro)
    ax.axis('equal')
    plt.savefig('contour_ro.pdf')
