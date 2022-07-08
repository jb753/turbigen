"""This script reads a steady TS solution and creates an unsteady input file."""
import numpy as np
from ts import ts_tstream_reader, ts_tstream_steady_to_unsteady, ts_tstream_type
from ts import ts_tstream_load_balance, ts_tstream_patch_kind
import sys

# Set parameters for unsteady data output

# Number of rotor blade passing periods to run for
# Change me so that the computaion reaches a periodic state
ncycle = 128

# Time steps per cycle
nstep_cycle = 72

# Number of time steps between probes
nstep_save_probe = 1


def convert(input_file, output_file):

    # Read in the converged steady solution
    tsr = ts_tstream_reader.TstreamReader()
    g = tsr.read(input_file)
    bids = g.get_block_ids()

    # Add probes at mid-span
    for bid in bids:
        b = g.get_block(bid)
        jmid = b.nj // 2
        p = ts_tstream_type.TstreamPatch()
        p.kind = ts_tstream_patch_kind.probe
        p.bid = bid
        p.ist = 0
        p.ien = b.ni
        p.jst = jmid
        p.jen = jmid + 1
        p.kst = 0
        p.ken = b.nk
        p.nxbid = 0
        p.nxpid = 0
        p.idir = 0
        p.jdir = 1
        p.kdir = 2
        p.pid = g.add_patch(bid, p)
        g.set_pv("probe_append", ts_tstream_type.int, p.bid, p.pid, 1)

    # Get numbers of blades and circumferential extents
    nb = np.array([g.get_bv("nblade", bid) for bid in bids])
    dt = 2.0 * np.pi / nb

    # Duplicate to 8th annulus
    sect = 8.0
    dt_sect = 2.0 * np.pi / sect

    # Get blade numbers for the sector
    dup = np.round(dt_sect / dt).astype(int)
    scale = dt_sect / dup / dt
    nb_sect = (dup * sect).astype(int)

    # Set frequency based on vane passing
    rpm = g.get_bv("rpm", bids[-1])
    freq = rpm / 60.0 * nb_sect[0]
    print("frequency f=%.1f Hz" % freq)

    # Hard-code the periodic patch connections (sorry)
    periodic = {}
    periodic[0] = (0, 0)
    periodic[1] = (0, 2)
    periodic[2] = (1, 0)
    periodic[3] = (1, 2)

    # Duplicate the grid to form the sector
    g2 = ts_tstream_steady_to_unsteady.steady_to_unsteady(
        g, dup, scale, periodic
    )

    # variables for unsteady run
    g2.set_av("ncycle", ts_tstream_type.int, ncycle)
    g2.set_av("frequency", ts_tstream_type.float, freq)

    g2.set_av("nstep_cycle", ts_tstream_type.int, nstep_cycle)
    g2.set_av("nstep_inner", ts_tstream_type.int, 100)

    # disable saving of snapshots
    g2.set_av("nstep_save", ts_tstream_type.int, 999999)

    # Which time step to start saving probes
    nstep_save_start = int((ncycle - nb_sect[-1] // 8) * nstep_cycle)

    # Save probes and average for last few period
    g2.set_av("nstep_save_start", ts_tstream_type.int, nstep_save_start)
    g2.set_av("nstep_save_start_probe", ts_tstream_type.int, nstep_save_start)

    # save probes every  nth step
    g2.set_av("nstep_save_probe", ts_tstream_type.int, nstep_save_probe)

    # other configuration variables
    g2.set_av("dts_conv", ts_tstream_type.float, 0.0001)
    g2.set_av("facsafe", ts_tstream_type.float, 0.2)
    g2.set_av("dts", ts_tstream_type.int, 1)

    # No multigrid, set blade numbers
    for bid in g2.get_block_ids():
        g2.set_bv("fmgrid", ts_tstream_type.float, bid, 0.0)

    # use mixing lengths and flow guess from steady calculation
    g2.set_av("restart", ts_tstream_type.int, 1)
    g2.set_av("poisson_restart", ts_tstream_type.int, 1)
    g2.set_av("poisson_nstep", ts_tstream_type.int, 0)

    # load balance for
    ts_tstream_load_balance.load_balance(g2, 1)

    # Reset spurious application variable
    g2.set_av("if_ale", ts_tstream_type.int, 0)
    g2.set_av("smooth_scale_dts_option", ts_tstream_type.int, 1)

    # write out unsteady input file
    g2.write_hdf5(output_file)

    print("Old blade counts [%d %d]" % tuple(nb))
    print("Scaled blade counts [%d %d] x %d" % (dup[0], dup[1], sect))
    print("Scaling factors [%f %f]" % tuple(scale))


if __name__ == "__main__":

    convert(sys.argv[1], sys.argv[2])
