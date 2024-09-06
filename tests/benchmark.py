"""Viscous test cases."""

import turbigen.solvers.embsolve
import turbigen.compflow_native as cf
import turbigen.grid
import turbigen.clusterfunc
import turbigen.util
import numpy as np
from timeit import default_timer as timer
import sys
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
import pytest

# Check our MPI rank
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# Jump to solver slave process if not first rank
if rank > 0:
    from turbigen.solvers import embsolve
    embsolve.run_slave()
    sys.exit(0)


def make_cylinder():

    # Geometry
    L = 0.4
    rm = 1.0
    dr = 0.1

    r1 = rm - dr / 2.0
    r2 = rm + dr / 2.0

    ni = 161
    nj = 81
    nk = 65
    ntot = ni*nj*nk

    Nb = np.round(2*np.pi * rm / dr).astype(int)
    pitch = 2.0 * np.pi / Nb

    xv = np.linspace(0, L, ni)
    rv = np.linspace(r1, r2, nj)
    tv = np.linspace(0.0, pitch, nk)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

    # Split into blocks
    blocks = []
    nblock = 1
    istb = [ni // nblock * iblock for iblock in range(nblock)]
    ienb = [ni // nblock * (iblock + 1) + 1 for iblock in range(nblock)]
    ienb[-1] = ni

    for iblock in range(nblock):

        # Special case for only one block
        if nblock == 1:
            patches = [
                turbigen.grid.InletPatch(i=0),
                turbigen.grid.OutletPatch(i=-1),
            ]

        # First block has an inlet
        elif iblock == 0:
            patches = [
                turbigen.grid.InletPatch(i=0),
                turbigen.grid.PeriodicPatch(i=-1),
            ]

        # Last block has outlet
        elif iblock == (nblock - 1):
            patches = [
                turbigen.grid.PeriodicPatch(i=0),
                turbigen.grid.OutletPatch(i=-1),
            ]

        # Middle blocks are both periodic
        else:
            patches = [
                turbigen.grid.PeriodicPatch(i=0),
                turbigen.grid.PeriodicPatch(i=-1),
            ]

        patches.extend(
            [
                turbigen.grid.PeriodicPatch(k=0),
                turbigen.grid.PeriodicPatch(k=-1),
            ]
        )

        block = turbigen.grid.PerfectBlock.from_coordinates(
            xrt[:, istb[iblock] : ienb[iblock], :, :], Nb, patches
        )
        block.label = f"b{iblock}"

        blocks.append(block)


    g = turbigen.grid.Grid(blocks)
    g.check_coordinates()

    # Boundary conditions
    cp = 1005.
    ga = 1.4
    mu = 1.84e-5
    Po1 = 1e5
    To1 = 300.
    Ma1 = 0.4
    V1 = cf.V_cpTo_from_Ma(Ma1, ga) * np.sqrt(cp * To1)
    P1 = Po1 / cf.Po_P_from_Ma(Ma1, ga)
    T1 = To1 / cf.To_T_from_Ma(Ma1, ga)
    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1.set_P_T(Po1, To1)
    So1.set_Tu0(To1)
    g.apply_inlet(So1, 0., 0.)
    g.calculate_wall_distance()
    g.apply_outlet(P1)
    g.match_patches()

    # Initial guess
    for b in g:
        b.Vx = V1
        b.Vr = 0.0
        b.Vt = 0.0
        b.cp = cp
        b.gamma = ga
        b.mu = mu
        b.Omega = 0.0
        b.set_P_T(P1, T1)
        b.set_Tu0(To1)


    print(f'Ncell/10^6 = {g.ncell/1e6}')

    conf = turbigen.solvers.embsolve.Config(n_step=201, n_step_avg=10)
    turbigen.solvers.embsolve.run(g, conf)


if __name__ == "__main__":

    make_cylinder()
