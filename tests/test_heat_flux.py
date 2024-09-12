"""Run a quasi-1D nozzle in the native solver."""

import turbigen.solvers.embsolve
import turbigen.compflow_native as cf
import turbigen.grid
import turbigen.util
import numpy as np
from timeit import default_timer as timer
from copy import copy
import sys
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
import pytest

def make_grid(
    L_h=4.0,
    skew=0.0,
    htr=0.95,
    Ma1=0.3,
    rpm=0.
):
    """Generate the grid."""

    # Geometry
    h = 0.1
    L = h * L_h
    rm = 0.5 * h * (1.0 + htr) / (1.0 - htr)
    rh = rm - 0.5 * h
    rt = rm + 0.5 * h

    # Boundary conditions
    ga = 1.4
    cp = 1005.0
    mu = 1.8e-3
    Beta = 0.0
    Po1 = 1e5
    To1 = 300.0
    To1a = To1
    TR = 2.
    To1b = To1*TR

    # Rotating reference frame
    Omega = rpm / 60. * np.pi * 2.
    U = Omega*rm

    # Set inlet Ma to get inlet static state
    rgas = cp * (ga - 1.0) / ga
    V = cf.V_cpTo_from_Ma(Ma1, ga) * np.sqrt(cp * To1)
    P1 = Po1 / cf.Po_P_from_Ma(Ma1, ga)
    T1a = To1a / cf.To_T_from_Ma(Ma1, ga)
    T1b = To1b / cf.To_T_from_Ma(Ma1, ga)


    # Relative flow angle
    Alpha = 0.
    Vt = V * np.sin(np.radians(Alpha))
    Vx = V * np.cos(np.radians(Alpha))
    Vt_rel = Vt - U
    Alpha_rel = np.degrees(np.arctan2(Vt_rel, V))

    # Numbers of grid points
    AR_pitch = 1.
    AR_merid = 2.
    nj = 33
    nk = 17
    ni = int(nj * L_h/ AR_merid)

    # Use pitchwise aspect ratio to find cell spacing, pitch and Nb
    pitch = h / (nj - 1) * (nk - 1) * AR_pitch
    Nb = int(2.0 * np.pi * rm / pitch)
    dt = 2.0 * np.pi / float(Nb)

    # Make the coordinates
    tv = np.linspace(0.0, dt, nk)
    xv = np.linspace(0.0, L, ni)
    rv = np.linspace(rh, rt, nj)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

    # Sentinel of None means skew along flow direction
    if skew is None:
        skew_max = 30.
        skew = np.clip(-Alpha_rel,-skew_max, skew_max)

    # Apply skew
    xrt[2] += xrt[0] * np.tan(np.radians(skew)) / xrt[1]

    jmid = nj//2

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
                turbigen.grid.InletPatch(i=0, j=(0,jmid)),
                turbigen.grid.InletPatch(i=0, j=(jmid,-1)),
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
                turbigen.grid.InviscidPatch(k=0),
                turbigen.grid.InviscidPatch(k=-1),
                turbigen.grid.InviscidPatch(j=0),
                turbigen.grid.InviscidPatch(j=-1),
            ]
        )

        block = turbigen.grid.PerfectBlock.from_coordinates(
            xrt[:, istb[iblock] : ienb[iblock], :, :], Nb, patches
        )
        block.label = f"b{iblock}"

        blocks.append(block)

    # Make the grid object
    g = turbigen.grid.Grid(blocks)
    g.check_coordinates()

    # Boundary conditions
    So1a = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1b = So1a.copy()
    So1a.set_P_T(Po1, To1a)
    So1b.set_P_T(Po1, To1b)
    So1 = [So1a, So1b]


    for patch, state in zip(g.inlet_patches, So1):
        patch.state = state
        patch.Alpha = 0.
        patch.Beta = 0.

    g.calculate_wall_distance()
    g.apply_outlet(P1)

    # Fluid props
    for b in g:
        b.cp = cp
        b.gamma = ga
        b.mu = mu
        b.Omega = Omega

    # Initial guess
    for ib, b in enumerate(g):
        b.Vx = Vx
        b.Vr = 0.0
        b.Vt = Vt
        T1 = np.empty(b.shape)
        T1[:,:jmid,:] = T1a
        T1[:,jmid:,:] = T1b
        b.set_P_T(P1, T1)

    g.match_patches()

    return g


settings = {
    "n_step": 5000,
    "n_step_avg": 500,
    "nstep_damp": -1,
    "xllim_pitch": 10.0,
    "plot_conv": True,
}
conf = turbigen.solvers.embsolve.Config(**settings)

def not_test_heat_flux():
    """"""

    g = make_grid()

    np.set_printoptions(precision=2)

    turbigen.solvers.embsolve.run(g, conf)

    # Check To conservation
    C1 = g[0][0,:,:]
    C2 = g[0][-1,:,:]
    C1m, A, _ = C1.mix_out()
    C2m, _, _ = C2.mix_out()
    print(C1m.To, C2m.To)

    fig, ax = plt.subplots()
    ax.axis('equal')
    b = g[0]
    C = b[:,:,b.nk//2]
    cm = ax.contourf(C.x, C.r, C.To)
    plt.colorbar(cm)
    plt.show()


if __name__ == "__main__":
    not_test_heat_flux()

